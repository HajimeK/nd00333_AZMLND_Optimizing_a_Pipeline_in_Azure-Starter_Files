# AzureにおけるML Pipelineの最適化

## 概要
このプロジェクトは、Udacity Azure MLNanodegreeの一部です。
このプロジェクトでは、Python SDKと提供されているScikit-learnモデルを使用してAzureMLパイプラインを構築および最適化します。
次に、このモデルはAzureAutoMLの実行と比較されます。

## サマリ
ここでは、<a href='https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv'>Bank Marketing</a>のデータセットを使用します。このデータセットには、銀行サービスに関心を持っている個人に関するデータが含まれています。銀行のサービスに加入している個人のパターンを特定するのがここでの課題です。

<img src="https://video.udacity-data.com/topher/2020/September/5f639574_creating-and-optimizing-an-ml-pipeline/creating-and-optimizing-an-ml-pipeline.png"></img>

このプロジェクトでは、MLパイプラインを作成してハイパーパラメータの最適化を図り、結果を比較しました。

- カスタムコード化されたモデル（標準のScikit-learnロジスティック回帰）をハイパーパラメーターをHyperDriveを使用して最適化
- AutoMLを使用して同じデータセットでモデルを構築および最適化

もっとも性能の良かったモデルは、AutoMLによって得られたVotingEnsempleモデルで、91.64%の"Accuracy"を達成しました。Scikit-Learnの89.58%よりも若干良い結果が得られています。

## Scikit-learn Pipeline

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">Scikit-LearnのLogistic Regressionモデル</a>を採用し、Hyperdriveによりハイパーパラメータのチューニングを行いました。

計算リソースの有効活用の面でのメリットを図るため、ランダム サンプリングを使用して最初の検索を行った後、早期終了の検索空間を絞り込んで結果を改善します。

```
# Specify parameter sampler
ps = RandomParameterSampling( {
    "--C": uniform(0.1, 1.0),
    "--max_iter": choice(range(1, 100))
    }
)
```
このサンプリングアルゴリズムでは、不連続値のセットまたは連続した範囲の分布からパラメーター値が選択されます。
C, max_iterはLogistic Regressionのパラメータです。
Cは、0.1から1.0の間の均等に分布される値を返します。
max_iterは、1から100までの範囲の整数値をとります。

ここで早期終了のために、ここではBanditPolicyにより、パフォーマンスが低く見込みのないパラメータについては早期に最適値の探索を停止します。BanditPolicyにより、余裕期間の基準に基づいた早期終了ポリシーと、評価の頻度と遅延間隔を定義します。

```
policy = BanditPolicy(slack_factor=0.1, evaluation_interval=2)
```

slack_factor: 最高の実行トレーニングの実行に関して許容される余裕時間。 この係数は、余裕期間を比率として指定します。
evaluation_interval:任意。 ポリシーを適用する頻度。 トレーニング スクリプトによってログに記録されるたびに、主要メトリックは 1 間隔としてカウントされます。

## AutoML

AutoMLによって使用されるhyperparameterは以下のように定義されます。

```
automl_config = AutoMLConfig(
    experiment_timeout_minutes = 30,
    task = 'classification',
    primary_metric = 'accuracy',
    training_data = pd.concat([X,y], axis = 1),
    #valudation_data = validation_data,
    label_column_name = 'y',
    n_cross_validations = 5)
```

The configuation parameters:
```
- experiement will time out after 30 minutes
- classification models
- Primary metric is "accuracy" among  others for classification ("https://docs.microsoft.com/ja-jp/azure/machine-learning/how-to-configure-auto-train#primary-metric")
- Data is split into the 5 cross validation data sets
```

![](2021-07-05-19-39-09.png)


## Pipeline比較

もっとも性能の良かったモデルは、AutoMLによって得られた*VotingEnsemble*モデルで、91.64%の"Accuracy"を達成しました。Scikit-Learnの89.58%よりも若干良い結果が得られています。

AutoMLによる選択では*VotingEnsemble*モデルでは*XGBoost*アルゴリズムが用いたEnsemble手法です。このモデルは私が実施した他の実験でもパラメータを調整することで他のEnsemble手法の*Random Forest*よりもよい結果を得られています[<a href="https://github.com/HajimeK/machine-learning/blob/master/projects/capstone/report.pdf">Ref</a>]。



## Future work

AutoMLで比較的容易に複数モデルを検証し、HyperDriveで使用した個別モデル相当もしくは以上の結果を得られることを見てきました。
データの妥当性の評価と今後より良い結果を得ること2面から評価を行うことが必要と考えています。

データの妥当性の評価については、今回はRandomなデータ分割を行っていますが、データの偏りや異常値などの評価を行わずにデータセットを採用しています。*clean_data*をアップデートして、こうしたデータの*feature engineering*についても実施する必要があると考えています。

また、「今後より良い結果を得ること」という観点からは、*AutoML*で得られた*VotingEnsemble*のハイパーパラメータをチューニングして*AutoML*の結果の評価の妥当性をさらに評価します。*Azure ML Studio*で重要な特徴量の情報が得られます。こうした情報を使用し、PCAとの比較の中でアルゴリズムを評価し、学習結果を評価する必要があります。