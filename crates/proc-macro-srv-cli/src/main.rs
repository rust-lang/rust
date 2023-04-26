//! A standalone binary for `proc-macro-srv`.
//! Driver for proc macro server
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

#[cfg(not(feature = "sysroot-abi"))]
fn run() -> io::Result<()> {
    panic!("proc-macro-srv-cli requires the `sysroot-abi` feature to be enabled");
}

#[cfg(feature = "sysroot-abi")]
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
            msg::Request::ExpandMacro(task) => msg::Response::ExpandMacro(srv.expand(task)),
            msg::Request::ApiVersionCheck {} => {
                msg::Response::ApiVersionCheck(proc_macro_api::msg::CURRENT_API_VERSION)
            }
        };
        write_response(res)?
    }

    Ok(())
}
