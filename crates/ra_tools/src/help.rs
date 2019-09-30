//! FIXME: write short doc here

pub const GLOBAL_HELP: &str = "tasks

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
    lint";

pub const INSTALL_RA_HELP: &str = "ra_tools-install-ra

USAGE:
    ra_tools.exe install-ra [FLAGS]

FLAGS:
        --client-code
    -h, --help           Prints help information
        --jemalloc
        --server";

pub fn print_no_param_subcommand_help(subcommand: &str) {
    eprintln!(
        "ra_tools-{}

USAGE:
    ra_tools {}

FLAGS:
    -h, --help       Prints help information",
        subcommand, subcommand
    );
}

pub const INSTALL_RA_CONFLICT: &str =
    "error: The argument `--server` cannot be used with `--client-code`
                    
For more information try --help";
