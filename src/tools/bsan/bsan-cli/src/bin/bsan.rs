#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_session;

use rustc_session::{EarlyDiagCtxt, config::ErrorOutputType};
use std::env;

const BSAN_BUG_REPORT_URL: &str = "https://github.com/BorrowSanitizer/rust/issues/new";

fn main() {
    env_logger::init();
    let early_dcx = EarlyDiagCtxt::new(ErrorOutputType::default());
    let args = rustc_driver::args::raw_args(&early_dcx)
        .unwrap_or_else(|_| std::process::exit(rustc_driver::EXIT_FAILURE));

    rustc_driver::install_ice_hook(BSAN_BUG_REPORT_URL, |_| ());

    // If this flag is set, then bsan will act like rustc
    let args = if env::var("BSAN_BE_RUSTC").is_ok() {
        args
    } else {
        let mut rustc_args = vec![];
        // bsan is being invoked through RUSTC_WRAPPER
        for arg in args.iter().skip(1) {
            if arg == "--" {
                break;
            } else {
                rustc_args.push(arg.to_string());
            }
        }
        rustc_args
    };

    let internal_features =
        rustc_driver::install_ice_hook(rustc_driver::DEFAULT_BUG_REPORT_URL, |_| ());
    bsan::run_compiler(args, &mut bsan::BSanCallBacks {}, internal_features)
}
