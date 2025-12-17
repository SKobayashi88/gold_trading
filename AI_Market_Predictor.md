AI Market Predictor (Institutional Edition)

Project Specification: Final Trinity Build

1. プロジェクト概要 (Executive Summary)

AI Market Predictor は、個人投資家が機関投資家（ヘッジファンド）の意思決定プロセスを完全再現するための統合投資分析プラットフォームである。
最新の生成AIによる戦略立案・リスク監査に加え、数学的な資金管理（マネーマネジメント）機能を統合し、「何を、いつ、いくらで売買するか」というトレードの全工程をデータドリブン化する。

2. コア・アーキテクチャ: The Hedge Fund Trinity

本システムは、実際のヘッジファンド運用体制を模した**「三位一体 (Trinity)」**の役割分担で構成される。

Role

Agent / Engine

役割 (Responsibility)

アウトプット

CIO (Chief Investment Officer)

OpenAI GPT-5.1

「攻め」



市場機会の発見と、論理的な売買戦略の立案。

戦略シナリオ (Bull/Bear)



売買ルール (JSON)

CRO (Chief Risk Officer)

Gemini 3.0 Flash

「守り」



戦略の脆弱性診断、過学習の指摘、マクロリスク警告。

監査レポート (Pass/Fail)



リスク警告

PM (Portfolio Manager)

Quantitative Engine

「調整」



勝率とボラティリティに基づいた、最適なポジションサイズの算出。

適正ロット数



ケリー基準値

3. 実装機能詳細

A. 市場分析・戦略フェーズ (Intelligence & Strategy)

マクロ/カレンダー分析: Geminiによる経済指標カレンダー生成とファンダメンタルズ分析。

高度テクニカル分析: scipyによるチャートパターン（ダブルトップ等）の数学的特定。

戦略生成: OpenAIによるバックテスト可能な戦略コード（JSON）の生成。

B. 検証フェーズ (Validation Lab)

Mark-to-Market Backtest: 含み損益ベースでの厳密な資産曲線シミュレーション。

Walk-Forward Optimization: 期間をスライドさせた検証によるカーブフィッティング排除。

コスト管理: スプレッド・手数料を考慮した実戦的損益計算。

C. 資金管理フェーズ (Money Management) [NEW]

Kelly Criterion Calculator: バックテストの「勝率」と「ペイオフレシオ（リスクリワード）」から、数学的に資産を最大化する賭け率（ケリー値）を算出。

Volatility Targeting: 現在の市場ボラティリティに基づき、リスクが一定になるようポジションサイズを動的に調整（ボラティリティが高い時はロットを落とす）。

4. データ・技術スタック

Frontend: Streamlit

AI Models: OpenAI GPT-4o/5.1, Google Gemini 3.0/2.0

Data Sources: yfinance, duckduckgo-search, CFTC

Storage: SQLite3 (全トレード・分析履歴の永続化)

5. 結論

本システムは、単なる予測ツールではない。
CIOの「アイデア」、CROの「批判的思考」、PMの「資金規律」をデスクトップ上に再現し、感情を排した完全な投資意思決定を実現するオペレーティング・システムである。