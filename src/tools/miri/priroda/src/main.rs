#![feature(rustc_private)]

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

use std::collections::HashSet;
use std::io::{self, Write};
use std::path::PathBuf;

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
                    // TODO: translate Miri termination into a Priroda execution-state enum so
                    // the CLI loop can distinguish whole-program exit from individual thread
                    // completion, print the exit code, and return to the debugger prompt.
                    println!("program finished with exit code {return_code}");
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

struct SourceLocation {
    local_path: Option<PathBuf>,
    display: String,
    line: usize,
    column: usize,
}

#[derive(Eq, Hash, PartialEq)]
struct Breakpoint {
    path: PathBuf,
    line: usize,
}

pub struct PrirodaContext<'tcx> {
    ecx: MiriInterpCx<'tcx>,
    breakpoints: HashSet<Breakpoint>,
    current_location: Option<SourceLocation>,
    last_location: Option<SourceLocation>,
}

fn normalize_path(path: PathBuf) -> PathBuf {
    path.canonicalize().unwrap_or(path)
}

impl<'tcx> PrirodaContext<'tcx> {
    fn new(ecx: MiriInterpCx<'tcx>) -> Self {
        Self { ecx, breakpoints: HashSet::new(), current_location: None, last_location: None }
    }

    // TODO: return a StepResult enum once we distinguish breakpoint stops,
    // program exit, and other debugger states.
    pub fn step(&mut self) -> InterpResult<'tcx> {
        // state inspection should happen only after a successful step
        self.ecx.miri_step()?;
        self.last_location = self.current_location.take();
        self.current_location = self.resolve_current_location();
        interp_ok(())
    }

    pub fn continue_execution(&mut self) -> InterpResult<'tcx> {
        loop {
            self.step()?;
            if self.is_at_breakpoint() {
                return interp_ok(());
            }
        }
    }

    fn set_breakpoint(&mut self, path: PathBuf, line: usize) {
        self.breakpoints.insert(Breakpoint { path: normalize_path(path), line });
    }

    fn is_at_breakpoint(&self) -> bool {
        // TODO: avoid repeated stops when one source line maps to multiple MIR statements.
        let Some(location) = &self.current_location else {
            return false;
        };

        let Some(path) = &location.local_path else {
            return false;
        };

        self.breakpoints.contains(&Breakpoint { path: path.clone(), line: location.line })
    }

    fn resolve_current_location(&self) -> Option<SourceLocation> {
        // TODO: resolve macro-backed lines such as `println!` and `assert_eq!` for breakpoints.
        let span = self.ecx.machine.current_user_relevant_span();
        if span.is_dummy() {
            return None;
        }

        let source_map = self.ecx.tcx.sess.source_map();
        let loc = source_map.lookup_char_pos(span.lo());

        Some(SourceLocation {
            // TODO: cache normalized source paths; this runs after every MIR step.
            local_path: loc.file.name.clone().into_local_path().map(normalize_path),
            display: source_map.span_to_diagnostic_string(span),
            line: loc.line,
            column: loc.col_display + 1,
        })
    }

    pub fn print_location(&self) {
        // TODO: skip noisy std/runtime spans and avoid printing `no-location`
        // once the basic command loop is solid.
        match &self.current_location {
            Some(location) => println!("{}", location.display),
            None => println!("no-location"),
        }
        io::stdout().flush().unwrap();
    }
    fn run_command(&mut self, command: SessionCommand) -> InterpResult<'tcx> {
        match command {
            SessionCommand::Step => self.step(),
            SessionCommand::Quit => unreachable!("quit is handled by the CLI loop"),
            SessionCommand::Continue => self.continue_execution(),
            SessionCommand::Breakpoint(path, line) => {
                // TODO: print a breakpoint confirmation instead of treating this like an execution step.
                self.set_breakpoint(path, line);
                interp_ok(())
            }
        }
    }
}

enum SessionCommand {
    Step,
    Quit,
    Continue,
    Breakpoint(PathBuf, usize),
}

fn parse_breakpoint(input: &str) -> Option<SessionCommand> {
    // TODO: reject empty paths and line 0 with a useful `usage: break <path>:<line>` error.
    let (path, line) = input.rsplit_once(':')?;
    let line = line.parse().ok()?;

    Some(SessionCommand::Breakpoint(PathBuf::from(path), line))
}

fn parse_command(input: &str) -> Option<SessionCommand> {
    let input = input.trim();
    let mut parts = input.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or("");
    let args = parts.next().unwrap_or("").trim();

    match command {
        "" | "s" | "step" => Some(SessionCommand::Step),
        "q" | "quit" => Some(SessionCommand::Quit),
        "c" | "continue" => Some(SessionCommand::Continue),
        "b" | "break" => parse_breakpoint(args),
        _ => None,
    }
}

fn run_cli_loop<'tcx>(session: &mut PrirodaContext<'tcx>) -> InterpResult<'tcx> {
    loop {
        print!("(priroda) ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input).unwrap();

        if bytes_read == 0 {
            println!("stdin closed, stopping");
            return interp_ok(());
        }

        if let Some(command) = parse_command(&input) {
            match command {
                SessionCommand::Quit => {
                    println!("quitting");
                    return interp_ok(());
                }

                command => {
                    session.run_command(command)?;
                    session.print_location();
                }
            }
        } else {
            println!("no command");
        }
    }
}
