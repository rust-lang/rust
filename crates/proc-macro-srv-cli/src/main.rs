//! A standalone binary for `proc-macro-srv`.

use proc_macro_srv::cli;

fn main() -> std::io::Result<()> {
    cli::run()
}
