//! A standalone binary for `proc-macro-srv`.

use proc_macro_srv::cli;

fn main() {
    cli::run().unwrap();
}
