#![feature(rustc_private)]

use std::task::Poll;

extern crate miri;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_hir_analysis;
extern crate rustc_interface;
extern crate rustc_log;
extern crate rustc_middle;
extern crate rustc_session;

use std::io::{self, Write};

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

    args.splice(1..1, miri::MIRI_DEFAULT_ARGS.iter().map(ToString::to_string));

    let sysroot_flag = String::from("--sysroot");
    if !args.contains(&sysroot_flag) {
        args.push(sysroot_flag);
        args.push(find_sysroot());
    }
    //TODO: handle the same `-Z` flags that Miri accepts.
    rustc_driver::run_compiler(&args, &mut PrirodaCompilerCalls::new());
}

struct PrirodaCompilerCalls;

impl PrirodaCompilerCalls {
    fn new() -> Self {
        Self
    }
}

impl rustc_driver::Callbacks for PrirodaCompilerCalls {
    fn after_analysis<'tcx>(&mut self, _: &interface::Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        tcx.dcx().emit_stashed_diagnostics();
        tcx.dcx().abort_if_errors();

        if !tcx.crate_types().contains(&CrateType::Executable) {
            //TODO: support non-bin crates by listing functions and letting users call them with manually entered arguments.
            tcx.dcx().fatal("priroda only makes sense on bin crates");
        }

        let ecx = create_ecx(tcx);

        let mut session = PrirodaContext::new(ecx);
        let result = run_cli_loop(&mut session);

        match result.report_err() {
            Ok(()) => {}
            Err(err) =>
                if let Some((return_code, _leak_check)) = report_result(&session.ecx, err) {
                    //TODO: print the evaluated program's exit code and return to the debugger prompt instead of exiting Priroda.
                    if return_code != 0 {
                        std::process::exit(return_code);
                    }
                },
        }

        Compilation::Stop
    }
}

fn create_ecx<'tcx>(tcx: TyCtxt<'tcx>) -> MiriInterpCx<'tcx> {
    let (entry_id, entry_type) = miri::entry_fn(tcx);
    let config = MiriConfig::default();
    miri::create_ecx(tcx, entry_id, entry_type, &config, None).unwrap()
}

pub struct PrirodaContext<'tcx> {
    ecx: MiriInterpCx<'tcx>,
}

impl<'tcx> PrirodaContext<'tcx> {
    fn new(ecx: MiriInterpCx<'tcx>) -> Self {
        Self { ecx }
    }

    // TODO: replace the bool with a StepResult enum once we distinguish
    // running, finished, breakpoint stops, and other debugger states.
    pub fn step(&mut self) -> InterpResult<'tcx, bool> {
        if !self.ecx.step()? {
            match self.ecx.run_on_stack_empty()? {
                Poll::Pending => return interp_ok(true),
                Poll::Ready(()) => {
                    self.ecx.terminate_active_thread(TlsAllocAction::Deallocate)?;
                    return interp_ok(false);
                }
            }
        }

        interp_ok(true)
    }
    pub fn print_location(&self) {
        let span = self.ecx.machine.current_user_relevant_span();
        let location = self.ecx.tcx.sess.source_map().span_to_diagnostic_string(span);
        // TODO: skip noisy std/runtime spans and avoid printing `no-location`
        // once the basic command loop is solid.
        println!("{location}");
        io::stdout().flush().unwrap();
    }
    fn run_command(&mut self, command: SessionCommand) -> InterpResult<'tcx, bool> {
        match command {
            SessionCommand::Step => self.step(),
        }
    }
}

enum SessionCommand {
    Step,
}

fn parse_command(input: &str) -> Option<SessionCommand> {
    match input.trim() {
        "" | "s" | "step" => Some(SessionCommand::Step),
        _ => None,
    }
}

fn run_cli_loop<'tcx>(session: &mut PrirodaContext<'tcx>) -> InterpResult<'tcx> {
    loop {
        print!("(priroda) ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        // TODO: handle EOF explicitly so scripted input can stop the CLI instead
        // of being treated like an empty Enter step.
        io::stdin().read_line(&mut input).unwrap();

        if let Some(command) = parse_command(&input) {
            match session.run_command(command)? {
                false => {
                    println!("program finished");
                    return interp_ok(());
                }
                true => session.print_location(),
            }
        } else {
            println!("no command");
        }
    }
}
