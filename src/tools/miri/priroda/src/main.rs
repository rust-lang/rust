#![feature(rustc_private)]

extern crate rustc_abi;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

mod debugger;
mod frontend;

use debugger::PrirodaContext;
use miri::*;
use rustc_driver::Compilation;
use rustc_hir::attrs::CrateType;
use rustc_interface::interface;
use rustc_middle::ty::TyCtxt;
use rustc_session::EarlyDiagCtxt;
use rustc_session::config::ErrorOutputType;

fn find_sysroot() -> String {
    std::env::var("MIRI_SYSROOT")
        .expect("set MIRI_SYSROOT to the path from `cargo miri setup --print-sysroot`")
}

fn main() {
    let early_dcx = EarlyDiagCtxt::new(ErrorOutputType::default());
    rustc_driver::init_rustc_env_logger(&early_dcx);

    let mut args: Vec<String> = std::env::args().collect();
    let frontend = Frontend::parse_from_args(&mut args);

    args.splice(1..1, miri::MIRI_DEFAULT_ARGS.iter().map(ToString::to_string));

    let sysroot_flag = String::from("--sysroot");
    if !args.contains(&sysroot_flag) {
        args.push(sysroot_flag);
        args.push(find_sysroot());
    }
    // FIXME: handle the same `-Z` flags that Miri accepts.
    rustc_driver::run_compiler(&args, &mut PrirodaCompilerCalls::new(frontend));
}

/// Frontend selected by Priroda-specific CLI flags.
#[derive(Clone, Copy)]
enum Frontend {
    Cli,
    Dap { port: Option<u16> },
}

impl Frontend {
    /// Remove Priroda-only flags before forwarding the remaining arguments to rustc.
    fn parse_from_args(args: &mut Vec<String>) -> Self {
        let mut frontend = Frontend::Cli;
        let mut rustc_args = Vec::with_capacity(args.len());
        let mut parsing_priroda_args = true;

        let mut arg_iter = std::mem::take(args).into_iter();
        if let Some(program) = arg_iter.next() {
            rustc_args.push(program);
        }

        while let Some(arg) = arg_iter.next() {
            if parsing_priroda_args {
                if arg == "--dap" {
                    if matches!(frontend, Frontend::Cli) {
                        frontend = Frontend::Dap { port: None };
                    }
                    continue;
                }

                if arg == "--port" {
                    let port_str = arg_iter
                        .next()
                        .unwrap_or_else(|| Self::fatal_arg_error("--port requires a value"));
                    frontend = Frontend::Dap { port: Some(Self::parse_port(&port_str)) };
                    continue;
                }

                if let Some(port_str) = arg.strip_prefix("--port=") {
                    frontend = Frontend::Dap { port: Some(Self::parse_port(port_str)) };
                    continue;
                }

                if arg == "--" {
                    parsing_priroda_args = false;
                }
            }

            rustc_args.push(arg);
        }

        *args = rustc_args;
        frontend
    }

    fn parse_port(port: &str) -> u16 {
        port.parse()
            .unwrap_or_else(|_| Self::fatal_arg_error("--port requires a valid u16 port number"))
    }

    fn fatal_arg_error(message: &str) -> ! {
        eprintln!("priroda: {message}");
        std::process::exit(1);
    }
}

struct PrirodaCompilerCalls {
    frontend: Frontend,
}

impl PrirodaCompilerCalls {
    fn new(frontend: Frontend) -> Self {
        Self { frontend }
    }
}

impl rustc_driver::Callbacks for PrirodaCompilerCalls {
    fn after_analysis<'tcx>(&mut self, _: &interface::Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        tcx.dcx().emit_stashed_diagnostics();
        tcx.dcx().abort_if_errors();

        if !tcx.crate_types().contains(&CrateType::Executable) {
            // FIXME: support non-bin crates by listing functions and letting users call them with manually entered arguments.
            tcx.dcx().fatal("priroda only makes sense on bin crates");
        }

        let ecx = create_ecx(tcx);

        let mut session = PrirodaContext::new(ecx);
        let result = match self.frontend {
            Frontend::Cli => frontend::Cli {}.run_cli_loop(&mut session),
            Frontend::Dap { port } => frontend::Dap { port }.run_dap_loop(&mut session),
        };

        match result.report_err() {
            Ok(()) => {}
            Err(err) =>
                if let Some((return_code, _leak_check)) = report_result(&session.ecx, err) {
                    std::process::exit(return_code);
                },
        }

        Compilation::Stop
    }
}

fn create_ecx<'tcx>(tcx: TyCtxt<'tcx>) -> MiriInterpCx<'tcx> {
    let (entry_id, entry_type) = miri::entry_fn(tcx);
    // FIXME: share Miri launcher configuration so interpreted programs receive
    // their program name, arguments, environment snapshot, and `MIRI_CWD`.
    let config = MiriConfig::default();
    // FIXME: report interpreter initialization failures instead of panicking.
    miri::create_ecx(tcx, entry_id, entry_type, &config, None).unwrap()
}
