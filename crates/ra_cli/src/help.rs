//! FIXME: write short doc here

pub const GLOBAL_HELP: &str = "ra-cli

USAGE:
    ra_cli <SUBCOMMAND>

FLAGS:
    -h, --help        Prints help information

SUBCOMMANDS:
    analysis-bench
    analysis-stats
    highlight
    parse
    symbols";

pub const ANALYSIS_BENCH_HELP: &str = "ra_cli-analysis-bench

USAGE:
    ra_cli analysis-bench [FLAGS] [OPTIONS] [PATH]

FLAGS:
    -h, --help        Prints help information
    -v, --verbose

OPTIONS:
    --complete <PATH:LINE:COLUMN>    Compute completions at this location
    --highlight <PATH>               Hightlight this file

ARGS:
    <PATH>    Project to analyse";

pub const ANALYSIS_STATS_HELP: &str = "ra-cli-analysis-stats

USAGE:
    ra_cli analysis-stats [FLAGS] [OPTIONS] [PATH]

FLAGS:
    -h, --help            Prints help information
        --memory-usage
    -v, --verbose
    -q, --quiet

OPTIONS:
    -o <ONLY>

ARGS:
    <PATH>";

pub const HIGHLIGHT_HELP: &str = "ra-cli-highlight

USAGE:
    ra_cli highlight [FLAGS]

FLAGS:
    -h, --help       Prints help information
    -r, --rainbow";

pub const SYMBOLS_HELP: &str = "ra-cli-symbols

USAGE:
    ra_cli highlight [FLAGS]

FLAGS:
    -h, --help    Prints help inforamtion";

pub const PARSE_HELP: &str = "ra-cli-parse

USAGE:
    ra_cli parse [FLAGS]

FLAGS:
    -h, --help       Prints help inforamtion
        --no-dump";
