use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::PathBuf;

use miri::Immediate::Uninit;
use miri::{InterpErrorInfo, InterpErrorKind, TerminationInfo, *};
use rustc_abi::{FIRST_VARIANT, FieldIdx, Size};
use rustc_hir::def::CtorKind;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::mir::{self, Local, ProjectionElem, VarDebugInfoContents, VarDebugInfoFragment};
use rustc_middle::ty::{self, TyKind};
use rustc_span::source_map::SourceMap;
use rustc_span::{Span, Symbol};

/// Structured source information for frontends.
pub(super) struct SourceLocation {
    // Keep the span so each frontend can resolve paths with its own rendering
    // rules instead of forcing every caller to use one path representation.
    pub(super) span: Span,
    pub(super) line: usize,
    pub(super) column: usize,
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
pub(super) struct PrirodaContext<'tcx> {
    pub(super) ecx: MiriInterpCx<'tcx>,
    breakpoints: BreakpointTable,
    pub(super) current_location: Option<SourceLocation>,
    last_location: Option<SourceLocation>,
    // FIXME: add restart and other post-exit commands, similar to GDB and
    // old Priroda, instead of only replaying the saved exit code.
    exit_code: Option<i32>,
}

pub(super) enum StorageProj {
    Field(usize),
    Deref,
    Downcast(Symbol),
    Variant(usize),
    Unsupported(String),
}

impl StorageProj {
    pub(super) fn render(&self) -> String {
        match self {
            StorageProj::Field(field_idx) => format!(".{field_idx}"),
            StorageProj::Deref => ".*".to_string(),
            StorageProj::Downcast(name) => format!(" as {name}"),
            StorageProj::Variant(variant_idx) => format!(" as variant#{variant_idx}"),
            StorageProj::Unsupported(unsop) => format!(".<unsupported:{unsop}>"),
        }
    }
}

pub(super) struct LocalDesc {
    /// Source variable name from `VarDebugInfo`, if this row has one.
    pub(super) source_name: Option<Symbol>,

    /// Source-side projection from `VarDebugInfo::composite`, e.g. `.field` in source fragment `x.field`.
    pub(super) source_projection: Option<Vec<Symbol>>,

    /// MIR storage local that backs this description, if any.
    pub(super) local: Option<Local>,

    /// rendered/debug MIR place projection for now
    pub(super) storage_projection: Vec<StorageProj>,

    /// Display-rendered type for this description.
    pub(super) ty: String,

    /// Run-time state for now; will be expanded later
    pub(super) value: String,
}

impl LocalDesc {
    pub(super) fn source_projection_str(&self) -> String {
        self.source_projection
            .as_ref()
            .map(|fields| fields.iter().map(|field| field.to_string()).collect::<String>())
            .unwrap_or_default()
    }

    pub(super) fn storage_projection_str(&self) -> String {
        self.storage_projection.iter().map(StorageProj::render).collect::<String>()
    }
}

/// Controls when execution returns to the frontend.
enum ResumeMode {
    /// Stop at the next visible MIR instruction.
    MirInstruction,
    /// Stop at the next source line.
    ///
    /// `None` means the current interpreter position has no source location, so
    /// the first mapped source location is good enough to report.
    SourceLine(Option<(PathBuf, usize)>),
    /// Step over the source position `start_position`, entered from a stack of
    /// depth `start_stack_depth`.
    ///
    /// Execution keeps going while it is deeper than `start_stack_depth` (i.e.
    /// inside a call made from the stepped-over line), and stops once it is back
    /// at that depth or shallower and the displayed source position has changed.
    StepOver { start_position: Option<(PathBuf, usize)>, start_stack_depth: usize },
    /// Step out of the current user frame, stopping once execution returns to a
    /// shallower user-frame depth.
    StepOut { start_position: Option<(PathBuf, usize)>, start_user_frame_depth: usize },
    /// Stop at the first mapped source location from a user-relevant frame.
    ///
    /// This is the DAP entry-stop primitive: it skips over interpreter startup
    /// and Miri-internal frames until there is a location an editor can show.
    FirstUserSourceLocation,
    /// Continue until reaching a breakpoint.
    Continue,
}

/// Describes whether the current MIR instruction should be shown to the user.
enum InstructionVisibility {
    NoInstruction,
    Hidden,
    Visible,
}

impl ResumeMode {
    fn skipped_breakpoint(&self) -> Option<&(PathBuf, usize)> {
        match self {
            ResumeMode::SourceLine(Some(position))
            | ResumeMode::StepOver { start_position: Some(position), .. }
            | ResumeMode::StepOut { start_position: Some(position), .. } => Some(position),
            _ => None,
        }
    }
}

/// Describes why execution stopped and returned control to the frontend.
pub(super) enum StepResult {
    Step,
    Breakpoint,
    Exception { message: String },
}

pub(super) enum ExecutionResult {
    Stopped(StepResult),
    ProgramExited { code: i32 },
    Rejected { message: &'static str },
}

fn normalize_path(path: PathBuf) -> PathBuf {
    path.canonicalize().unwrap_or(path)
}

impl<'tcx> PrirodaContext<'tcx> {
    pub(super) fn new(ecx: MiriInterpCx<'tcx>) -> Self {
        Self {
            ecx,
            breakpoints: HashMap::new(),
            current_location: None,
            last_location: None,
            exit_code: None,
        }
    }

    pub(super) fn local_path(&self, location: &SourceLocation) -> Option<PathBuf> {
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

    fn already_finished(&self) -> Option<ExecutionResult> {
        self.exit_code.map(|code| ExecutionResult::ProgramExited { code })
    }

    /// Step to the next visible MIR instruction.
    fn stepi(&mut self) -> InterpResult<'tcx, ExecutionResult> {
        if let Some(result) = self.already_finished() {
            return interp_ok(result);
        }
        self.resume(ResumeMode::MirInstruction)
    }

    /// Step until the displayed source file or line changes.
    ///
    /// This is the CLI source-level step; it shares its stepping semantics with
    /// [`Self::step_in_source`].
    pub(super) fn step(&mut self) -> InterpResult<'tcx, ExecutionResult> {
        self.step_in_source()
    }

    /// Step into the next source location.
    ///
    /// This can enter calls that have a distinct displayed source position,
    /// while `next` uses [`Self::step_over_source`].
    pub(super) fn step_in_source(&mut self) -> InterpResult<'tcx, ExecutionResult> {
        if let Some(result) = self.already_finished() {
            return interp_ok(result);
        }
        self.resume(ResumeMode::SourceLine(self.current_source_position()))
    }

    /// Step over the current source position, not stopping inside any call it makes.
    ///
    /// Records the current source position and stack depth before advancing,
    /// then keeps stepping until execution is back at that depth (or shallower)
    /// and the displayed source position has changed.
    pub(super) fn step_over_source(&mut self) -> InterpResult<'tcx, ExecutionResult> {
        if let Some(result) = self.already_finished() {
            return interp_ok(result);
        }
        let start_position = self.current_source_position();
        let start_stack_depth = self.active_thread_stack_depth();
        self.resume(ResumeMode::StepOver { start_position, start_stack_depth })
    }

    /// Number of frames on the active thread's stack.
    fn active_thread_stack_depth(&self) -> usize {
        self.ecx.active_thread_stack().len()
    }

    /// Step out of the current user frame.
    ///
    /// Records the current user-frame depth and runs until execution reaches a
    /// source location in a shallower user frame.
    pub(super) fn step_out_source(&mut self) -> InterpResult<'tcx, ExecutionResult> {
        if let Some(result) = self.already_finished() {
            return interp_ok(result);
        }
        let start_user_frame_depth = self.active_user_frame_depth();
        if start_user_frame_depth <= 1 {
            return interp_ok(ExecutionResult::Rejected {
                message: "stepOut is not meaningful in the outermost user frame",
            });
        }
        let start_position = self.current_source_position();
        self.resume(ResumeMode::StepOut { start_position, start_user_frame_depth })
    }

    /// Run until the initial editor-visible stop point.
    pub(super) fn stop_at_first_user_location(&mut self) -> InterpResult<'tcx, ExecutionResult> {
        self.resume(ResumeMode::FirstUserSourceLocation)
    }

    /// Return the active frame name while DAP still reports only one frame.
    pub(super) fn current_frame_name(&self) -> Option<String> {
        let frame = self.ecx.active_thread_stack().last()?;
        Some(frame.instance().to_string())
    }

    /// Continue execution until reaching a breakpoint or propagating termination.
    pub(super) fn continue_execution(&mut self) -> InterpResult<'tcx, ExecutionResult> {
        if let Some(result) = self.already_finished() {
            return interp_ok(result);
        }
        self.resume(ResumeMode::Continue)
    }

    pub(super) fn finish_session(&mut self) -> InterpResult<'tcx, ()> {
        interp_ok(())
    }

    pub(super) fn set_breakpoint(&mut self, path: PathBuf, line: usize) -> BreakpointSetResult {
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

    fn program_exit(err: &InterpErrorInfo<'tcx>) -> Option<i32> {
        let InterpErrorKind::MachineStop(info) = err.kind() else {
            return None;
        };
        // FIXME: Preserve `TerminationInfo::Exit::leak_check` and run Miri's
        // leak/thread-leak diagnostics once Priroda grows a proper post-exit
        // finalization path. For now, program exit only records the debuggee exit code.
        let Some(TerminationInfo::Exit { code, .. }) = info.downcast_ref::<TerminationInfo>()
        else {
            return None;
        };
        Some(*code)
    }

    fn stop_at_exception(&mut self, err: InterpErrorInfo<'tcx>) -> StepResult {
        let message = err.kind().to_string();
        self.last_location = self.current_location.take();
        self.current_location = self.resolve_current_location();
        StepResult::Exception { message }
    }

    /// Advance execution until the selected resume mode reaches a stopping point.
    fn resume(&mut self, mode: ResumeMode) -> InterpResult<'tcx, ExecutionResult> {
        loop {
            // Program exits are not debugger exceptions. Preserve all other
            // interpreter errors as stopped debugger events.
            if let Err(err) = self.advance().report_err() {
                if let Some(code) = Self::program_exit(&err) {
                    self.exit_code = Some(code);
                    return interp_ok(ExecutionResult::ProgramExited { code });
                }
                return interp_ok(ExecutionResult::Stopped(self.stop_at_exception(err)));
            }

            // An explicit breakpoint should stop execution even when the current
            // MIR instruction would normally be hidden during manual stepping.
            if self.is_at_breakpoint(mode.skipped_breakpoint()) {
                return interp_ok(ExecutionResult::Stopped(StepResult::Breakpoint));
            }

            match mode {
                ResumeMode::MirInstruction
                    if matches!(
                        self.current_instruction_visibility(),
                        InstructionVisibility::Visible
                    ) =>
                {
                    return interp_ok(ExecutionResult::Stopped(StepResult::Step));
                }

                ResumeMode::SourceLine(ref prev_location) => {
                    match (prev_location, &self.current_location) {
                        // We started from an unmapped location; stop once there
                        // is a source position the frontend can display.
                        (None, Some(_)) =>
                            return interp_ok(ExecutionResult::Stopped(StepResult::Step)),

                        (Some((prev_path, prev_line)), Some(current_location)) => {
                            if let Some(current_path) = self.local_path(current_location) {
                                // A source step stops when the displayed source
                                // position changes to a different file or line.
                                if *prev_path != current_path || *prev_line != current_location.line
                                {
                                    return interp_ok(ExecutionResult::Stopped(StepResult::Step));
                                }
                            }
                        }

                        _ => {}
                    }
                }

                ResumeMode::StepOver { ref start_position, start_stack_depth } => {
                    // While deeper than where we started, we are inside a call
                    // made from the stepped-over line; keep going.
                    if self.active_thread_stack_depth() > start_stack_depth {
                        continue;
                    }

                    // Back at (or shallower than) the starting depth: stop once
                    // the displayed source position has changed.
                    match (start_position, &self.current_location) {
                        // We started from an unmapped location; stop once there
                        // is a source position the frontend can display.
                        (None, Some(_)) =>
                            return interp_ok(ExecutionResult::Stopped(StepResult::Step)),
                        (Some((start_path, start_line)), Some(current_location)) => {
                            // A source step stops when the displayed source
                            // position changes to a different file or line.
                            if let Some(current_path) = self.local_path(current_location)
                                && (*start_path != current_path
                                    || *start_line != current_location.line)
                            {
                                // Return spans can point at a function header. Keep walking when
                                // that would move `next` backwards within the same frame.
                                if self.active_thread_stack_depth() == start_stack_depth
                                    && *start_path == current_path
                                    && current_location.line < *start_line
                                {
                                    continue;
                                }
                                return interp_ok(ExecutionResult::Stopped(StepResult::Step));
                            }
                        }
                        _ => {}
                    }
                }

                ResumeMode::StepOut { start_user_frame_depth, .. }
                    if self.active_user_frame_depth() < start_user_frame_depth
                        && self.current_location.is_some() =>
                {
                    return interp_ok(ExecutionResult::Stopped(StepResult::Step));
                }

                ResumeMode::FirstUserSourceLocation
                    if self.current_location.is_some() && self.has_user_relevant_frame() =>
                {
                    return interp_ok(ExecutionResult::Stopped(StepResult::Step));
                }

                ResumeMode::MirInstruction
                | ResumeMode::FirstUserSourceLocation
                | ResumeMode::StepOut { .. }
                | ResumeMode::Continue => {}
            }
        }
    }

    fn has_user_relevant_frame(&self) -> bool {
        self.active_user_frame_depth() > 0
    }

    fn active_user_frame_depth(&self) -> usize {
        // Walk the whole stack, not just the top frame: during interpreter
        // startup the user's `main` can sit under Miri-internal frames that
        // have no source span, so checking only `last()` would miss it.
        self.ecx
            .active_thread_stack()
            .iter()
            .filter(|frame| frame.extra.user_relevance == u8::MAX)
            .count()
    }

    /// Advance Miri by one interpreter-loop transition.
    fn advance(&mut self) -> InterpResult<'tcx> {
        // FIXME: use a Miri-owned scheduler-aware debugger step API before
        // claiming support for multi-threaded interpreted programs.

        // State inspection should happen only after a successful step.
        self.ecx.step_current_thread()?;
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

    fn is_at_breakpoint(&self, skipped_breakpoint: Option<&(PathBuf, usize)>) -> bool {
        let Some(bp) = self.current_breakpoint() else {
            return false;
        };
        if skipped_breakpoint == Some(&bp) {
            return false;
        }

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
        let span = self.ecx.machine.current_user_relevant_span();
        if span.is_dummy() {
            return None;
        }

        let span = span.source_callsite();
        let source_map = self.ecx.tcx.sess.source_map();
        let loc = source_map.lookup_char_pos(span.lo());

        Some(SourceLocation { span, line: loc.line, column: loc.col_display + 1 })
    }

    pub(super) fn run_command(
        &mut self,
        command: DebuggerCommand,
    ) -> InterpResult<'tcx, CommandResult> {
        match command {
            DebuggerCommand::StepI => self.stepi().map(CommandResult::Execution),
            DebuggerCommand::Step => self.step().map(CommandResult::Execution),
            DebuggerCommand::Next => self.step_over_source().map(CommandResult::Execution),
            DebuggerCommand::StepOut => self.step_out_source().map(CommandResult::Execution),
            DebuggerCommand::Continue => self.continue_execution().map(CommandResult::Execution),
            DebuggerCommand::Breakpoint(path, line) =>
                interp_ok(CommandResult::BreakpointResult(self.set_breakpoint(path, line))),
            DebuggerCommand::ListLocals => interp_ok(CommandResult::Locals(self.list_locals())),
            DebuggerCommand::Print(local) =>
                interp_ok(CommandResult::SingleLocal(self.get_local(local))),
            DebuggerCommand::Follow(alloc_id, offset) =>
                self.follow_alloc(alloc_id, offset).map(CommandResult::Memory),
            DebuggerCommand::TerminateSession =>
                self.finish_session().map(|()| CommandResult::TerminateSession),
        }
    }

    fn follow_alloc(&self, alloc_id: AllocId, offset: usize) -> InterpResult<'tcx, String> {
        let alloc = self.ecx.get_alloc_raw(alloc_id)?;
        if offset > alloc.len() {
            return Err(miri::err_unsup_format!(
                "allocation offset {offset} is outside {alloc_id}"
            ))
            .into();
        }

        let memory = self.render_alloc_bytes(alloc_id, offset..alloc.len())?;
        interp_ok(format!("Allocation {alloc_id}+{offset}: {memory}"))
    }

    fn get_local(&self, local: usize) -> Option<LocalDesc> {
        let frame = self.ecx.active_thread_stack().last()?;

        self.make_mir_local_desc(frame, local)
    }

    /// Returns structured descriptions for locals in the innermost stack frame.
    ///
    /// Starts from all MIR locals, then enriches them with source names from
    /// `var_debug_info` when a debug entry maps directly to a whole local.
    pub(super) fn list_locals(&self) -> Vec<LocalDesc> {
        let Some(frame) = self.ecx.active_thread_stack().last() else {
            return Vec::new();
        };

        self.build_local_descs(frame)
    }

    /// Renders the current byte range of an indirect MIR value.
    ///
    /// Initialized bytes are shown in hexadecimal, uninitialized bytes as `??`,
    /// and complete pointer-sized provenance as pointer markers.
    fn render_mplace_bytes(&self, mplace: &MPlaceTy<'tcx>) -> InterpResult<'tcx, String> {
        let Some((size, _)) = self.ecx.size_and_align_of_val(mplace)? else {
            // Extern types cannot currently be executed as by-value locals,
            // so this path cannot yet be covered by a Priroda UI fixture.
            // FIXME: Add coverage once Priroda supports printing dereferenced places.
            return interp_ok("<unsupported-unsized>".to_string());
        };

        let size = size.bytes_usize();
        if size == 0 {
            return interp_ok("[]".to_string());
        }

        let (alloc_id, offset, _) =
            self.ecx.ptr_get_alloc_id(mplace.ptr(), size.try_into().unwrap())?;
        let offset = offset.bytes_usize();
        let range = offset..offset.strict_add(size);

        self.render_alloc_bytes(alloc_id, range)
    }

    /// Render a raw allocation range without requiring a typed memory place.
    ///
    /// This is also used by the future-facing `follow` command, where we have a
    /// pointer target but do not yet know the target's type or size.
    fn render_alloc_bytes(
        &self,
        alloc_id: AllocId,
        range: Range<usize>,
    ) -> InterpResult<'tcx, String> {
        let alloc = self.ecx.get_alloc_raw(alloc_id)?;

        let mut rendered = Vec::with_capacity(range.len());

        let ptr_size = self.ecx.tcx.data_layout.pointer_size();

        for chunk in alloc.init_mask().range_as_init_chunks(range.into()) {
            let chunk_range = chunk.range();
            let chunk_range = chunk_range.start.bytes_usize()..chunk_range.end.bytes_usize();

            if chunk.is_init() {
                let ptr_size = ptr_size.bytes_usize();
                let mut cursor = chunk_range.start;

                while cursor < chunk_range.end {
                    // Full pointer provenance is rendered as a pointer marker. Bytewise
                    // provenance fragments are intentionally left as raw bytes here: they do
                    // not represent a complete pointer-sized value.
                    if let Some(prov) = alloc.provenance().get_ptr(Size::from_bytes(cursor))
                        && cursor + ptr_size <= chunk_range.end
                    {
                        let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(
                            cursor..cursor + ptr_size,
                        );
                        let offset = read_target_uint(self.ecx.tcx.data_layout.endian, bytes)
                            .map_err(|err| {
                                miri::err_unsup_format!("invalid pointer representation: {err}")
                            })?;

                        let offset = Size::from_bytes(offset);
                        rendered.push(format!("{:?}", Pointer::new(Some(prov), offset)));

                        cursor += ptr_size;
                    } else {
                        let byte = alloc
                            .inspect_with_uninit_and_ptr_outside_interpreter(cursor..cursor + 1)[0];

                        rendered.push(format!("{byte:02x}"));
                        cursor += 1;
                    }
                }
            } else {
                rendered.extend(std::iter::repeat_n("__".to_string(), chunk_range.len()));
            }
        }

        interp_ok(format!("[{}]", rendered.join(" ")))
    }

    /// Render an evaluated operand using Rust-source-shaped containers with raw leaves.
    ///
    /// The operand is produced from live interpreter state, usually via `local_to_op`
    /// for a whole MIR local or `eval_place_to_op` for a projected debug-info place.
    ///
    /// This intentionally does not call user `Debug` / `Display`, and it does not
    /// try to make every scalar leaf pretty yet. Unsupported cases and leaf values
    /// fall back to `render_op`, preserving the old raw byte/provenance renderer.
    ///
    /// FIXME: teach the leaf renderer about simple Rust scalars (`bool`, integers,
    /// chars, raw pointers/references) once the source-shaped container output is
    /// stable enough to stop depending on byte dumps for every field.
    ///
    /// FIXME: decide how much dereferencing belongs in this renderer. References
    /// currently stay as raw pointer leaves; following them may belong in the
    /// existing `follow` command instead of automatic local rendering.
    fn render_source_shaped_op(&self, op: OpTy<'tcx>) -> String {
        self.render_source_shaped_op_inner(op, 0)
    }

    /// Recursive worker for `render_source_shaped_op`.
    ///
    /// The depth limit keeps cyclic/reference-heavy values from making debugger
    /// output explode once more container kinds are added. At the limit, the raw
    /// renderer remains the ground truth.
    ///
    /// FIXME: replace this fixed recursion limit with a value-size/output-budget
    /// policy so large acyclic values and deeply nested values degrade more
    /// predictably.
    fn render_source_shaped_op_inner(&self, op: OpTy<'tcx>, depth: usize) -> String {
        const MAX_SOURCE_SHAPE_DEPTH: usize = 8;

        if depth >= MAX_SOURCE_SHAPE_DEPTH {
            return self.render_op(op);
        }

        match op.layout.ty.kind() {
            // Empty enums have no active variant to format. Unions do not record
            // which field is currently active, so choosing one would be misleading.
            //
            // FIXME: support unions only with an explicit user-selected field or
            // another source of active-field information. Guessing from layout
            // bytes would make debugger output look more certain than it is.
            ty::Adt(def, _) if def.variants().is_empty() || def.is_union() => self.render_op(op),

            ty::Adt(def, _) => {
                // Enums need their runtime discriminant and a downcasted layout
                // view before fields can be projected. Structs use their sole
                // variant directly. Keep the display name tied to the same choice.
                let (variant_idx, down, name) = if def.is_enum() {
                    let Some(variant_idx) = self.ecx.read_discriminant(&op).discard_err() else {
                        // FIXME: expose this as an explicit render error when
                        // Priroda grows structured value states. Falling back to
                        // bytes keeps today's UI usable but hides why the enum
                        // could not be source-shaped.
                        return self.render_op(op);
                    };
                    let Some(down) = self.ecx.project_downcast(&op, variant_idx).discard_err()
                    else {
                        // FIXME: distinguish invalid/uninitialized discriminants
                        // from projection bugs in the rendered output once locals
                        // can carry structured diagnostics.
                        return self.render_op(op);
                    };
                    let variant_def = &def.variants()[variant_idx];
                    (
                        variant_idx,
                        down,
                        format!("{}::{}", self.ecx.tcx.item_name(def.did()), variant_def.name),
                    )
                } else {
                    let variant_idx = FIRST_VARIANT;
                    let variant_def = &def.variants()[variant_idx];
                    (variant_idx, op.clone(), variant_def.name.to_string())
                };

                let variant_def = &def.variants()[variant_idx];

                let mut fields = Vec::with_capacity(variant_def.fields.len());
                for i in 0..variant_def.fields.len() {
                    let field_idx = FieldIdx::from_usize(i);
                    // `project_field` avoids manual offset math and works for both
                    // immediate and memory-backed operands through `Projectable`.
                    let Some(field_op) = self.ecx.project_field(&down, field_idx).discard_err()
                    else {
                        // FIXME: preserve the successfully rendered fields and
                        // mark only this field as unavailable once the value model
                        // can represent partial render failures.
                        return self.render_op(op);
                    };
                    fields.push(self.render_source_shaped_op_inner(field_op, depth + 1));
                }

                // Match Rust constructor spelling:
                // - `Const`: unit structs/variants, e.g. `UnitStruct`, `Enum::Unit`
                // - `Fn`: tuple structs/variants, e.g. `Pair(a, b)` or `EmptyTuple()`
                // - `None`: braced structs/variants, including the empty `{}` case
                match variant_def.ctor_kind() {
                    Some(CtorKind::Const) => name,
                    Some(CtorKind::Fn) => format!("{name}({})", fields.join(", ")),
                    None if fields.is_empty() => format!("{name} {{}}"),
                    None => {
                        let fields = variant_def
                            .fields
                            .iter()
                            .zip(fields)
                            .map(|(field_def, value)| format!("{}: {value}", field_def.name))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{name} {{ {fields} }}")
                    }
                }
            }

            ty::Tuple(args) => {
                let mut fields = Vec::with_capacity(args.len());
                for i in 0..args.len() {
                    // Tuples have no field names in source, so preserve their
                    // source field order and render children positionally.
                    let Some(field_op) =
                        self.ecx.project_field(&op, FieldIdx::from_usize(i)).discard_err()
                    else {
                        // FIXME: render tuple fields independently so one
                        // projection failure does not throw away the whole
                        // source-shaped tuple.
                        return self.render_op(op);
                    };
                    fields.push(self.render_source_shaped_op_inner(field_op, depth + 1));
                }

                if fields.len() == 1 {
                    format!("({},)", fields[0])
                } else {
                    format!("({})", fields.join(", "))
                }
            }

            ty::Array(_, _) | ty::Slice(_) => {
                // `project_array_fields` uses the dynamic length for slices. That
                // avoids the classic mistake of treating slice layout as a fixed
                // zero-length array.
                let Some(mut iter) = self.ecx.project_array_fields(&op).discard_err() else {
                    // FIXME: when slice metadata is invalid, show that as a slice
                    // length problem instead of silently falling back to raw bytes.
                    return self.render_op(op);
                };

                let mut fields = Vec::new();
                // FIXME: add an output budget/truncation policy before rendering
                // very large arrays or slices in full.
                loop {
                    match iter.next(&self.ecx).discard_err() {
                        Some(Some((_idx, field_op))) =>
                            fields.push(self.render_source_shaped_op_inner(field_op, depth + 1)),
                        Some(None) => break,
                        // FIXME: keep already-rendered elements and mark the
                        // failed index once partial render errors are supported.
                        None => return self.render_op(op),
                    }
                }

                format!("[{}]", fields.join(", "))
            }

            // FIXME: consider source-shaped special cases for strings, closures,
            // generators/coroutines, trait objects, and SIMD/vector-like types.
            // Until then these stay on the raw renderer path.
            _ => self.render_op(op),
        }
    }

    /// Render an evaluated operand using the same raw representation for
    /// whole locals and projected MIR places.
    fn render_op(&self, op: OpTy<'tcx>) -> String {
        match op.as_mplace_or_imm() {
            Either::Right(imm) => format!("{imm}"),

            Either::Left(mplace) =>
                match self.render_mplace_bytes(&mplace).report_err() {
                    Ok(bytes) => bytes,
                    Err(err) => format!("<error: {}>", err.to_string()),
                },
        }
    }

    /// Render the source-side path from composite debug info, such as `.field`.
    fn render_source_projection(
        fragment: Option<&VarDebugInfoFragment<'tcx>>,
    ) -> Option<Vec<Symbol>> {
        let VarDebugInfoFragment { ty, projection } = fragment?;

        // Walk the source-side projection from the original
        // composite variable type. Each `Field` element stores the
        // resulting field type, so resolve the field name from the
        // current base type before advancing to `field_ty`.
        let mut projection_ty = ty;

        Some(
            projection
                .iter()
                .map(|elem| {
                    match elem {
                        ProjectionElem::Field(field_idx, field_ty) => {
                            let rendered = match projection_ty.kind() {
                                TyKind::Adt(adt_def, _args) if adt_def.is_struct() => {
                                    let variant = adt_def.non_enum_variant();
                                    let field = &variant.fields[*field_idx];
                                    Symbol::intern(&format!(".{}", field.name))
                                }

                                TyKind::Tuple(_) =>
                                    Symbol::intern(&format!(".{}", field_idx.index())),

                                _ => Symbol::intern(".<unexpected>"),
                            };

                            projection_ty = field_ty;

                            rendered
                        }
                        // `VarDebugInfoFragment::projection` is expected to be
                        // field-only. If that ever changes, keep the unexpected
                        // segment visible instead of silently rendering a
                        // misleading source path.
                        other => Symbol::intern(&format!(".<unsupported:{other:?}>")),
                    }
                })
                .collect(),
        )
    }

    /// Render the MIR storage-side path that backs a debug-info local.
    fn render_storage_projection(projection: &[mir::PlaceElem<'tcx>]) -> Vec<StorageProj> {
        projection
            .iter()
            .map(|projection_elem| {
                match projection_elem {
                    ProjectionElem::Field(field_idx, _) => StorageProj::Field(field_idx.index()),
                    ProjectionElem::Deref => StorageProj::Deref,
                    ProjectionElem::Downcast(Some(name), _) => StorageProj::Downcast(*name),
                    ProjectionElem::Downcast(None, variant_idx) =>
                        StorageProj::Variant(variant_idx.index()),
                    other => StorageProj::Unsupported(format!("{other:?}")),
                }
            })
            .collect()
    }

    /// Builds the baseline debugger row for one MIR local without scanning debug info.
    fn make_mir_local_desc(
        &self,
        frame: &Frame<'tcx, Provenance, FrameExtra<'tcx>>,
        local: usize,
    ) -> Option<LocalDesc> {
        let local = mir::Local::from_usize(local);
        let local_decl = frame.body().local_decls.get(local)?;

        // Create LocalDesc for MIR local before processing debug info.
        // Debug-info enrichment is layered on by build_local_descs.
        let mut local_desc = LocalDesc {
            source_name: None,
            source_projection: None,
            local: Some(local),
            storage_projection: Vec::new(),
            ty: local_decl.ty.to_string(),
            value: "<unsupported>".to_string(),
        };

        match &frame.locals[local].as_mplace_or_imm() {
            None => {
                local_desc.value = "<dead>".to_string();
            }
            Some(Either::Right(Uninit)) => local_desc.value = "<uninit>".to_string(),

            Some(Either::Left(_) | Either::Right(_)) => {
                let op = self
                    .ecx
                    .local_to_op(local, None)
                    .expect("this error can only occur in CTFE on generic code");
                local_desc.value = self.render_source_shaped_op(op);
            }
        };

        Some(local_desc)
    }

    fn build_local_descs(
        &self,
        frame: &Frame<'tcx, Provenance, FrameExtra<'tcx>>,
    ) -> Vec<LocalDesc> {
        let local_decls = &frame.body().local_decls;

        let mut local_descs: Vec<LocalDesc> = Vec::with_capacity(local_decls.len());

        // Start with one baseline row for every MIR local, then layer debug info on top.
        for (local_idx, _) in local_decls.iter_enumerated() {
            local_descs.push(self.make_mir_local_desc(frame, local_idx.index()).unwrap());
        }

        // FIXME: Finish classifying `var_debug_info` by keeping the source path
        // and MIR storage path separate:
        //
        // - source side: `var_debug_info.name` plus
        //   `var_debug_info.composite.projection`
        // - storage side: `VarDebugInfoContents::Place(place).local` plus
        //   `place.projection`
        //
        // Already handled by the `place.as_local()` path below:
        // - whole source variable -> whole MIR local:
        //   `composite = None`, `Place(_N)` with empty projection.
        // - source fragment -> whole MIR local:
        //   `composite = Some(source_proj)`, `Place(_N)` with empty projection.
        //
        // Remaining cases to represent or explicitly defer:
        // - whole source variable -> projected MIR storage:
        //   `composite = None`, `Place(_N.proj)`.
        // - source fragment -> projected MIR storage:
        //   `composite = Some(source_proj)`, `Place(_N.storage_proj)`.
        // - source variable/fragment -> constant:
        //   `Const(...)`, with no MIR local id.
        // - optimized-out/debug-only/unsupported shapes:
        //   explicit deferred state, not silent discard.
        //
        // Final output should be produced by walking `Vec<LocalDesc>`,
        // then append explicit deferred/debug-info-only rows where needed.
        // Related: SROA can split a source local like `_slice: ExtraSlice` into
        // field locals whose debug paths should be printed as `_slice._slice`
        // and `_slice._extra`, not as two separate locals both named `_slice`.

        // Whole-place debug entries enrich the direct storage-local description.
        // Projected places are evaluated from their original MIR Place and use
        // the same raw renderer as ordinary locals.
        for var_debug_info in &frame.body().var_debug_info {
            if let VarDebugInfoContents::Place(place) = &var_debug_info.value {
                if let Some(local_idx) = place.as_local()
                    && local_descs[local_idx.index()].source_name.is_none()
                {
                    let local_idx = local_idx.index();
                    local_descs[local_idx].source_projection =
                        Self::render_source_projection(var_debug_info.composite.as_deref());
                    local_descs[local_idx].source_name = Some(var_debug_info.name);
                } else if !place.projection.is_empty() {
                    let storage_projection = Self::render_storage_projection(place.projection);
                    let source_projection =
                        Self::render_source_projection(var_debug_info.composite.as_deref());
                    let value = self
                        .ecx
                        .eval_place_to_op(*place, None)
                        .map(|op| self.render_source_shaped_op(op))
                        .unwrap_or_else(|err| format!("<error: {}>", err.to_string()));

                    local_descs.push(LocalDesc {
                        source_name: Some(var_debug_info.name),
                        source_projection,
                        local: Some(place.local),
                        storage_projection,
                        ty: place.ty(local_decls, self.ecx.tcx.tcx).ty.to_string(),
                        value,
                    });
                }
            }
        }

        local_descs
    }
}

pub(super) enum DebuggerCommand {
    StepI,
    Step,
    Next,
    StepOut,
    TerminateSession,
    Continue,
    Breakpoint(PathBuf, usize),
    ListLocals,
    Print(usize),
    Follow(AllocId, usize),
}

pub(super) enum BreakpointSetResult {
    Added(PathBuf, usize),
    Duplicate,
    // FIXME: add pending breakpoint support later if needed.
}

pub(super) enum CommandResult {
    Execution(ExecutionResult),
    BreakpointResult(BreakpointSetResult),
    Locals(Vec<LocalDesc>),
    SingleLocal(Option<LocalDesc>),
    Memory(String),
    // FIXME: distinguish terminating the debugger session from disconnecting a
    // frontend and terminating the interpreted program once multiple frontends exist.
    TerminateSession,
}
