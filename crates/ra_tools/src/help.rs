pub fn print_global_help() {
    println!(
        "tasks

USAGE:
    ra_tools <SUBCOMMAND>

FLAGS:
    -h, --help       Prints help information

SUBCOMMANDS:
    format
    format-hook
    fuzz-tests
    gen-syntax
    gen-tests
    install-ra
    lint"
    )
}

pub fn print_install_ra_help() {
    println!(
        "ra_tools-install-ra

USAGE:
    ra_tools.exe install-ra [FLAGS]

FLAGS:
        --client-code
    -h, --help           Prints help information
        --jemalloc
        --server"
    )
}

pub fn print_no_param_subcommand_help(subcommand: &str) {
    println!(
        "ra_tools-{}

USAGE:
    ra_tools {}

FLAGS:
    -h, --help       Prints help information",
        subcommand, subcommand
    );
}

pub fn print_install_ra_conflict() {
    println!(
        "error: The argument `--server` cannot be used with `--client-code`
                    
For more information try --help"
    )
}
