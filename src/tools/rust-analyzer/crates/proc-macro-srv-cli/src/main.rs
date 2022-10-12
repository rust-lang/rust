//! A standalone binary for `proc-macro-srv`.

use proc_macro_srv::cli;

fn main() -> std::io::Result<()> {
    let v = std::env::var("RUST_ANALYZER_INTERNALS_DO_NOT_USE");
    match v.as_deref() {
        Ok("this is unstable") => {
            // very well, if you must
        }
        _ => {
            eprintln!("If you're rust-analyzer, you can use this tool by exporting RUST_ANALYZER_INTERNALS_DO_NOT_USE='this is unstable'.");
            eprintln!("If not, you probably shouldn't use this tool. But do what you want: I'm an error message, not a cop.");
            std::process::exit(122);
        }
    }

    cli::run()
}
