//! A standalone binary for `proc-macro-srv`.
//! Driver for proc macro server
#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]
#[cfg(feature = "in-rust-tree")]
extern crate rustc_driver as _;

use std::io;

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

    run()
}

#[cfg(not(any(feature = "sysroot-abi", rust_analyzer)))]
fn run() -> io::Result<()> {
    eprintln!("proc-macro-srv-cli requires the `sysroot-abi` feature to be enabled");
    std::process::exit(70);
}

#[cfg(any(feature = "sysroot-abi", rust_analyzer))]
fn run() -> io::Result<()> {
    use proc_macro_api::msg::{self, Message};

    let read_request = |buf: &mut String| msg::Request::read(&mut io::stdin().lock(), buf);

    let write_response = |msg: msg::Response| msg.write(&mut io::stdout().lock());

    let mut srv = proc_macro_srv::ProcMacroSrv::default();
    let mut buf = String::new();

    while let Some(req) = read_request(&mut buf)? {
        let res = match req {
            msg::Request::ListMacros { dylib_path } => {
                msg::Response::ListMacros(srv.list_macros(&dylib_path))
            }
            msg::Request::ExpandMacro(task) => match srv.span_mode() {
                msg::SpanMode::Id => {
                    msg::Response::ExpandMacro(srv.expand(*task).map(|(it, _)| it))
                }
                msg::SpanMode::RustAnalyzer => msg::Response::ExpandMacroExtended(
                    srv.expand(*task).map(|(tree, span_data_table)| msg::ExpandMacroExtended {
                        tree,
                        span_data_table,
                    }),
                ),
            },
            msg::Request::ApiVersionCheck {} => {
                msg::Response::ApiVersionCheck(proc_macro_api::msg::CURRENT_API_VERSION)
            }
            msg::Request::SetConfig(config) => {
                srv.set_span_mode(config.span_mode);
                msg::Response::SetConfig(config)
            }
        };
        write_response(res)?
    }

    Ok(())
}
