pub fn print_global_help() {
    println!(
        "ra-cli

USAGE:
    ra_cli <SUBCOMMAND>

FLAGS:
    -h, --help        Prints help information

SUBCOMMANDS:
    analysis-bench
    analysis-stats
    highlight
    parse
    symbols"
    )
}

pub fn print_analysis_bench_help() {
    println!(
        "ra_cli-analysis-bench

USAGE:
    ra_cli analysis-bench [FLAGS] [OPTIONS] [PATH]

FLAGS:
    -h, --help        Prints help information
    -v, --verbose
    
OPTIONS:
    --complete <PATH:LINE:COLUMN>    Compute completions at this location
    --highlight <PATH>               Hightlight this file
    
ARGS:
    <PATH>    Project to analyse"
    )
}

pub fn print_analysis_stats_help() {
    println!(
        "ra-cli-analysis-stats

USAGE:
    ra_cli analysis-stats [FLAGS] [OPTIONS] [PATH]
    
FLAGS:
    -h, --help            Prints help information
        --memory-usage
    -v, --verbose
    
OPTIONS:
    -o <ONLY>
    
ARGS:
    <PATH>"
    )
}

pub fn print_highlight_help() {
    println!(
        "ra-cli-highlight
    
USAGE:
    ra_cli highlight [FLAGS]
    
FLAGS:
    -h, --help       Prints help information
    -r, --rainbow"
    )
}

pub fn print_symbols_help() {
    println!(
        "ra-cli-symbols
    
USAGE:
    ra_cli highlight [FLAGS]
    
FLAGS:
    -h, --help    Prints help inforamtion"
    )
}

pub fn print_parse_help() {
    println!(
        "ra-cli-parse
    
USAGE:
    ra_cli parse [FLAGS]
    
FLAGS:
    -h, --help       Prints help inforamtion
        --no-dump"
    )
}
