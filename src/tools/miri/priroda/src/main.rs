#![feature(rustc_private)]

extern crate miri;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_hir_analysis;
extern crate rustc_index;
extern crate rustc_interface;
extern crate rustc_log;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::PathBuf;

use miri::*;
use rustc_driver::Compilation;
use rustc_hir::attrs::CrateType;
use rustc_index::IndexVec;
use rustc_interface::interface;
use rustc_middle::mir;
use rustc_middle::mir::{Local, VarDebugInfoContents};
use rustc_middle::ty::TyCtxt;
use rustc_session::EarlyDiagCtxt;
use rustc_session::config::ErrorOutputType;
use rustc_span::source_map::SourceMap;
use rustc_span::{Span, Symbol};

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
    // FIXME: handle the same `-Z` flags that Miri accepts.
    rustc_driver::run_compiler(&args, &mut PrirodaCompilerCalls::new());
}

struct PrirodaCompilerCalls;

impl PrirodaCompilerCalls {
    // FIXME: remove this constructor if PrirodaCompilerCalls remains a unit struct.
    fn new() -> Self {
        Self
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
        let cli = Cli {};
        let result = cli.run_cli_loop(&mut session);

        match result.report_err() {
            Ok(()) => {}
            Err(err) =>
                if let Some((return_code, _leak_check)) = report_result(&session.ecx, err) {
                    // FIXME: translate Miri termination into a Priroda execution-state enum so
                    // the CLI loop can distinguish whole-program exit from individual thread
                    // completion, run Miri-equivalent leak checks, print the exit code, and
                    // return to the debugger prompt.
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
    // FIXME: share Miri launcher configuration so interpreted programs receive
    // their program name, arguments, environment snapshot, and `MIRI_CWD`.
    let config = MiriConfig::default();
    // FIXME: report interpreter initialization failures instead of panicking.
    miri::create_ecx(tcx, entry_id, entry_type, &config, None).unwrap()
}

/// Structured source information for frontends.
struct SourceLocation {
    // storing `span` to use it lazily to compute path.
    span: Span,
    line: usize,
}

impl SourceLocation {
    fn local_path(&self, source_map: &SourceMap) -> Option<PathBuf> {
        let loc = source_map.lookup_char_pos(self.span.lo());
        loc.file.name.clone().into_local_path().map(normalize_path)
    }
}

/// Source-level breakpoints indexed by normalized path, then line.
type BreakpointTable = HashMap<PathBuf, HashSet<usize>>;

/// Owns one interpreter session and its debugger state.
///
/// Frontend rendering should eventually live outside this type.
struct PrirodaContext<'tcx> {
    ecx: MiriInterpCx<'tcx>,
    breakpoints: BreakpointTable,
    current_location: Option<SourceLocation>,
    last_location: Option<SourceLocation>,
}

struct LocalDesc {
    name: Option<Symbol>,
    local: Local,
    ty: String,
}
/// Controls when execution returns to the frontend.
enum ResumeMode {
    /// Stop at the next visible MIR instruction.
    MirInstruction,
    /// Stop at the next source line
    ///
    /// Take `Option` because some cases current state has no mapped to source code location
    SourceLine(Option<(PathBuf, usize)>),
    /// Continue until reaching a breakpoint.
    Continue,
}

/// Describes whether the current MIR instruction should be shown to the user.
enum InstructionVisibility {
    NoInstruction,
    Hidden,
    Visible,
}

/// Describes why execution stopped and returned control to the frontend.
enum StepResult {
    Step,
    Breakpoint,
}

fn normalize_path(path: PathBuf) -> PathBuf {
    path.canonicalize().unwrap_or(path)
}

impl<'tcx> PrirodaContext<'tcx> {
    fn new(ecx: MiriInterpCx<'tcx>) -> Self {
        Self { ecx, breakpoints: HashMap::new(), current_location: None, last_location: None }
    }

    fn local_path(&self, location: &SourceLocation) -> Option<PathBuf> {
        let source_map = self.ecx.tcx.sess.source_map();
        location.local_path(source_map)
    }

    fn current_source_position(&self) -> Option<(PathBuf, usize)> {
        let location = self.current_location.as_ref()?;
        Some((self.local_path(location)?, location.line))
    }

    // Used to treat `continue` like a source-level step for breakpoint checks:
    // several MIR locations can point at one source line, but they should only
    // report that source breakpoint once.
    fn last_source_position(&self) -> Option<(PathBuf, usize)> {
        let location = self.last_location.as_ref()?;
        Some((self.local_path(location)?, location.line))
    }

    /// Step to the next visible MIR instruction.
    fn stepi(&mut self) -> InterpResult<'tcx, StepResult> {
        self.resume(ResumeMode::MirInstruction)
    }
    fn step(&mut self) -> InterpResult<'tcx, StepResult> {
        self.resume(ResumeMode::SourceLine(self.current_source_position()))
    }

    /// Continue execution until reaching a breakpoint or propagating termination.
    fn continue_execution(&mut self) -> InterpResult<'tcx, StepResult> {
        self.resume(ResumeMode::Continue)
    }

    fn set_breakpoint(&mut self, path: PathBuf, line: usize) -> BreakpointSetResult {
        // FIXME: validate breakpoints here so every frontend gets the same behavior.
        // Reject empty paths, missing files, directories, and line 0. Decide whether
        // out-of-range lines should be rejected or kept as pending breakpoints.
        // Report duplicate registrations separately.

        let path = normalize_path(path);
        match self.breakpoints.entry(path.clone()).or_default().insert(line) {
            true => BreakpointSetResult::Added(path, line),
            false => BreakpointSetResult::Duplicate,
        }
    }

    /// Advance execution until the selected resume mode reaches a stopping point.
    fn resume(&mut self, mode: ResumeMode) -> InterpResult<'tcx, StepResult> {
        loop {
            self.advance()?;

            // An explicit breakpoint should stop execution even when the current
            // MIR instruction would normally be hidden during manual stepping.
            if self.is_at_breakpoint() {
                return interp_ok(StepResult::Breakpoint);
            }

            match mode {
                ResumeMode::MirInstruction
                    if matches!(
                        self.current_instruction_visibility(),
                        InstructionVisibility::Visible
                    ) =>
                {
                    return interp_ok(StepResult::Step);
                }

                ResumeMode::SourceLine(ref prev_location) => {
                    match (prev_location, &self.current_location) {
                        // We started from an unmapped source location. Stop at the first mapped source location we can show to the user.
                        (None, Some(_)) => return interp_ok(StepResult::Step),

                        (Some((prev_path, prev_line)), Some(current_location)) => {
                            if let Some(current_path) = self.local_path(current_location) {
                                // A source step stops when the visible source position changes to a different file or line.
                                if *prev_path != current_path || *prev_line != current_location.line
                                {
                                    return interp_ok(StepResult::Step);
                                }
                            }
                        }

                        _ => {}
                    }
                }

                ResumeMode::MirInstruction | ResumeMode::Continue => {}
            }
        }
    }

    /// Advance Miri by one interpreter-loop transition.
    fn advance(&mut self) -> InterpResult<'tcx> {
        // FIXME: use a Miri-owned scheduler-aware debugger step API before
        // claiming support for multi-threaded interpreted programs.

        // State inspection should happen only after a successful step.
        self.ecx.miri_step()?;
        self.last_location = self.current_location.take();
        self.current_location = self.resolve_current_location();
        interp_ok(())
    }

    fn current_instruction_visibility(&self) -> InstructionVisibility {
        // If the active thread has no stack frame, there is no MIR instruction to show.
        let Some(frame) = self.ecx.active_thread_stack().last() else {
            return InstructionVisibility::NoInstruction;
        };

        // `Right(span)` means the frame has source context but no precise MIR program-counter location.
        let Either::Left(location) = frame.current_loc() else {
            return InstructionVisibility::NoInstruction;
        };

        let basic_block = &frame.body().basic_blocks[location.block];

        // `statement_index == statements.len()` points at the block terminator.
        // Terminators affect control flow, so they are always visible.
        let Some(statement) = basic_block.statements.get(location.statement_index) else {
            return InstructionVisibility::Visible;
        };

        // Hide bookkeeping-only MIR statements during manual stepping.
        match statement.kind {
            mir::StatementKind::StorageLive(_)
            | mir::StatementKind::StorageDead(_)
            | mir::StatementKind::Nop => InstructionVisibility::Hidden,
            _ => InstructionVisibility::Visible,
        }
    }

    fn is_at_breakpoint(&self) -> bool {
        let Some(bp) = self.current_breakpoint() else {
            return false;
        };

        // If the previous interpreter step had the same source position, this
        // is another MIR location for the breakpoint we just reported.
        self.last_source_position().as_ref() != Some(&bp)
    }

    fn current_breakpoint(&self) -> Option<(PathBuf, usize)> {
        let (path, line) = self.current_source_position()?;
        let lines = self.breakpoints.get(&path)?;

        if lines.contains(&line) { Some((path, line)) } else { None }
    }

    fn resolve_current_location(&self) -> Option<SourceLocation> {
        // FIXME: resolve macro-backed lines such as `println!` and `assert_eq!`
        // through `span.source_callsite()` before matching breakpoints.
        let span = self.ecx.machine.current_user_relevant_span();
        if span.is_dummy() {
            return None;
        }

        let source_map = self.ecx.tcx.sess.source_map();
        let loc = source_map.lookup_char_pos(span.lo());

        Some(SourceLocation { span, line: loc.line })
    }

    fn run_command(&mut self, command: DebuggerCommand) -> InterpResult<'tcx, CommandResult> {
        match command {
            DebuggerCommand::StepI => self.stepi().map(CommandResult::ExecutionStopped),
            DebuggerCommand::Step => self.step().map(CommandResult::ExecutionStopped),
            DebuggerCommand::Continue =>
                self.continue_execution().map(CommandResult::ExecutionStopped),
            DebuggerCommand::Breakpoint(path, line) =>
                interp_ok(CommandResult::BreakpointResult(self.set_breakpoint(path, line))),
            DebuggerCommand::ListLocals => interp_ok(CommandResult::Locals(self.list_locals())),
            DebuggerCommand::TerminateSession => interp_ok(CommandResult::TerminateSession),
        }
    }

    /// Returns structured descriptions for locals in the innermost stack frame.
    ///
    /// Starts from all MIR locals, then enriches them with source names from
    /// `var_debug_info` when a debug entry maps directly to a whole local.
    fn list_locals(&self) -> Vec<LocalDesc> {
        let Some(frame) = self.ecx.active_thread_stack().last() else {
            return Vec::new();
        };

        self.local_desc_map(frame).into_iter().collect()
    }

    fn local_desc_map(
        &self,
        frame: &Frame<'tcx, Provenance, FrameExtra<'tcx>>,
    ) -> IndexVec<Local, LocalDesc> {
        // Initialize one description per MIR local so the table can be indexed by Local.
        let mut locals: IndexVec<Local, LocalDesc> = frame
            .body()
            .local_decls
            .iter_enumerated()
            .map(|(id, local_decl)| {
                LocalDesc { name: None, local: id, ty: local_decl.ty.to_string() }
            })
            .collect();

        // FIXME: Some debug-info entries do not have a backing MIR local, for example
        // because the source variable was optimized out or is represented as a
        // projection. This local-indexed table cannot represent those entries yet;
        // the final locals list should become a `Vec<LocalDesc>` with `id : Option<Local>`, `id`
        // could be renamed to `local`.

        // Attach source names from debug info when the debug entry maps directly to a whole MIR local.
        for var_debug_info in &frame.body().var_debug_info {
            if let VarDebugInfoContents::Place(place) = var_debug_info.value
                && let Some(local) = place.as_local()
                && locals[local].name.is_none()
            {
                locals[local].name = Some(var_debug_info.name);
            }
        }

        locals
    }
}

enum DebuggerCommand {
    StepI,
    Step,
    TerminateSession,
    Continue,
    Breakpoint(PathBuf, usize),
    ListLocals,
}

enum BreakpointSetResult {
    Added(PathBuf, usize),
    Duplicate,
    // FIXME: add pending breakpoint support later if needed.
}

enum CommandResult {
    ExecutionStopped(StepResult),
    BreakpointResult(BreakpointSetResult),
    Locals(Vec<LocalDesc>),
    // FIXME: distinguish terminating the debugger session from disconnecting a
    // frontend and terminating the interpreted program once multiple frontends exist.
    TerminateSession,
}

struct Cli;

impl Cli {
    pub fn run_cli_loop<'tcx>(&self, session: &mut PrirodaContext<'tcx>) -> InterpResult<'tcx> {
        loop {
            print!("(priroda) ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            let bytes_read = io::stdin().read_line(&mut input).unwrap();

            if bytes_read == 0 {
                println!("stdin closed, stopping");
                return interp_ok(());
            }

            if let Some(command) = self.parse_command(&input) {
                match session.run_command(command)? {
                    CommandResult::ExecutionStopped(result) => {
                        if matches!(result, StepResult::Breakpoint) {
                            println!("Hit breakpoint");
                        }
                        self.print_location(session);
                    }
                    CommandResult::BreakpointResult(res) =>
                        match res {
                            BreakpointSetResult::Added(path, line) =>
                                println!("breakpoint added: {}:{}", path.display(), line),

                            BreakpointSetResult::Duplicate => println!("Duplicate breakpoint"),
                        },
                    CommandResult::Locals(locals_desc) =>
                        if locals_desc.is_empty() {
                            println!("no locals");
                        } else {
                            for local_desc in &locals_desc {
                                let mut name_str = "None".to_string();
                                if let Some(name) = local_desc.name {
                                    name_str = name.to_string();
                                }
                                println!(
                                    "Name: {}, Id: _{}, Ty: {}",
                                    name_str,
                                    local_desc.local.index(),
                                    local_desc.ty,
                                );
                            }
                        },
                    CommandResult::TerminateSession => {
                        println!("quitting");
                        return interp_ok(());
                    }
                }
            } else {
                println!("no command");
            }

            io::stdout().flush().unwrap();
        }
    }

    fn parse_command(&self, input: &str) -> Option<DebuggerCommand> {
        // TODO: look at the Spanned crate for how to easily produce errors in
        // rustc's style while manually parsing text input.
        // FIXME: we need to distinguish malformed input from the unknown commands by returning useful
        // command error that describes if it malformed or non exist command
        let input = input.trim();
        let mut parts = input.splitn(2, char::is_whitespace);
        let command = parts.next().unwrap_or("");
        let args = parts.next().unwrap_or("").trim();

        match command {
            // FIXME: empty line should repats last command user typed not exeute specific command.
            "" | "si" | "stepi" => Some(DebuggerCommand::StepI),
            "s" | "step" => Some(DebuggerCommand::Step),
            "q" | "quit" => Some(DebuggerCommand::TerminateSession),
            "c" | "continue" => Some(DebuggerCommand::Continue),
            "b" | "break" => self.parse_breakpoint(args),
            "l" | "locals" => Some(DebuggerCommand::ListLocals),
            _ => None,
        }
    }

    fn print_location(&self, session: &PrirodaContext) {
        match &session.current_location {
            Some(location) =>
                if let Some(path) = session.local_path(location) {
                    println!("{}:{}", path.display(), location.line);
                } else {
                    let source_map = session.ecx.tcx.sess.source_map();
                    println!("{}", source_map.span_to_diagnostic_string(location.span));
                },
            None => println!("no-location"),
        }
        io::stdout().flush().unwrap();
    }

    fn parse_breakpoint(&self, input: &str) -> Option<DebuggerCommand> {
        // FIXME: return a typed CommandError so malformed breakpoint input is
        // distinguishable from an unknown command. Semantic validation belongs
        // in PrirodaContext::set_breakpoint so non-CLI frontends cannot bypass it.
        let (path, line) = input.rsplit_once(':')?;
        let line = line.parse().ok()?;

        Some(DebuggerCommand::Breakpoint(PathBuf::from(path), line))
    }
}
