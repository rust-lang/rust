//! Annotation pass for move/copy operations.
//!
//! This pass modifies the source scopes of statements containing `Operand::Move` and `Operand::Copy`
//! to make them appear as if they were inlined from `compiler_move()` and `compiler_copy()` intrinsic
//! functions. This creates the illusion that moves/copies are function calls in debuggers and
//! profilers, making them visible for performance analysis.
//!
//! The pass leverages the existing inlining infrastructure by creating synthetic `SourceScopeData`
//! with the `inlined` field set to point to the appropriate intrinsic function.

use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt, TypingEnv};
use rustc_session::config::DebugInfo;
use rustc_span::sym;

/// Default minimum size in bytes for move/copy operations to be annotated. Set to 64+1 bytes
/// (typical cache line size) to focus on potentially expensive operations.
const DEFAULT_ANNOTATE_MOVES_SIZE_LIMIT: u64 = 65;

/// Bundle up parameters into a structure to make repeated calling neater
struct Params<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    source_scopes: &'a mut IndexVec<SourceScope, SourceScopeData<'tcx>>,
    local_decls: &'a IndexVec<Local, LocalDecl<'tcx>>,
    typing_env: TypingEnv<'tcx>,
    size_limit: u64,
}

/// MIR transform that annotates move/copy operations for profiler visibility.
pub(crate) struct AnnotateMoves {
    compiler_copy: Option<DefId>,
    compiler_move: Option<DefId>,
}

impl<'tcx> crate::MirPass<'tcx> for AnnotateMoves {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.opts.unstable_opts.annotate_moves.is_enabled()
            && sess.opts.debuginfo != DebugInfo::None
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // Skip promoted MIR bodies to avoid recursion
        if body.source.promoted.is_some() {
            return;
        }

        let typing_env = body.typing_env(tcx);
        let size_limit = tcx
            .sess
            .opts
            .unstable_opts
            .annotate_moves
            .size_limit()
            .unwrap_or(DEFAULT_ANNOTATE_MOVES_SIZE_LIMIT);

        // Common params, including selectively borrowing the bits of Body we need to avoid
        // mut/non-mut aliasing problems.
        let mut params = Params {
            tcx,
            source_scopes: &mut body.source_scopes,
            local_decls: &body.local_decls,
            typing_env,
            size_limit,
        };

        // Storage for Call terminator argument SourceInfo
        let mut call_arg_source_info = Vec::new();

        // Process each basic block
        for (block, block_data) in body.basic_blocks.as_mut().iter_enumerated_mut() {
            for stmt in &mut block_data.statements {
                let source_info = &mut stmt.source_info;

                if let StatementKind::Assign(box (_, rvalue)) = &stmt.kind {
                    // Save the original scope before processing any operands. This prevents
                    // chaining when multiple operands are processed.
                    let original_scope = source_info.scope;

                    match rvalue {
                        Rvalue::Use(op)
                        | Rvalue::Repeat(op, _)
                        | Rvalue::Cast(_, op, _)
                        | Rvalue::UnaryOp(_, op) => {
                            self.annotate_move(&mut params, source_info, original_scope, op);
                        }
                        Rvalue::BinaryOp(_, box (lop, rop)) => {
                            self.annotate_move(&mut params, source_info, original_scope, lop);
                            self.annotate_move(&mut params, source_info, original_scope, rop);
                        }
                        Rvalue::Aggregate(_, ops) => {
                            for op in ops {
                                self.annotate_move(&mut params, source_info, original_scope, op);
                            }
                        }
                        Rvalue::Ref(..)
                        | Rvalue::ThreadLocalRef(..)
                        | Rvalue::RawPtr(..)
                        | Rvalue::NullaryOp(..)
                        | Rvalue::Discriminant(..)
                        | Rvalue::CopyForDeref(..)
                        | Rvalue::ShallowInitBox(..)
                        | Rvalue::WrapUnsafeBinder(..) => {} // No operands to instrument
                    }
                }
            }

            // Process terminator operands
            if let Some(terminator) = &mut block_data.terminator {
                let source_info = &mut terminator.source_info;
                // Save the original scope before processing any operands
                let original_scope = source_info.scope;

                match &terminator.kind {
                    TerminatorKind::Call { func, args, .. }
                    | TerminatorKind::TailCall { func, args, .. } => {
                        self.annotate_move(&mut params, source_info, original_scope, func);

                        // For Call arguments, store SourceInfo separately instead of modifying the
                        // terminator's SourceInfo (which would affect the entire Call)
                        for (index, arg) in args.iter().enumerate() {
                            if let Some(arg_source_info) = self.get_annotated_source_info(
                                &mut params,
                                original_scope,
                                &arg.node,
                            ) {
                                call_arg_source_info.push(((block, index), arg_source_info));
                            }
                        }
                    }
                    TerminatorKind::SwitchInt { discr: op, .. }
                    | TerminatorKind::Assert { cond: op, .. }
                    | TerminatorKind::Yield { value: op, .. } => {
                        self.annotate_move(&mut params, source_info, original_scope, op);
                    }
                    TerminatorKind::InlineAsm { operands, .. } => {
                        for op in &**operands {
                            match op {
                                InlineAsmOperand::In { value, .. }
                                | InlineAsmOperand::InOut { in_value: value, .. } => {
                                    self.annotate_move(
                                        &mut params,
                                        source_info,
                                        original_scope,
                                        value,
                                    );
                                }
                                // Const, SymFn, SymStatic, Out, and Label don't have Operands we care about
                                _ => {}
                            }
                        }
                    }
                    _ => {} // Other terminators don't have operands
                }
            }
        }

        // Store the Call argument SourceInfo in the body (only if we have any)
        body.call_arg_move_source_info = call_arg_source_info;
    }

    fn is_required(&self) -> bool {
        false // Optional optimization/instrumentation pass
    }
}

impl AnnotateMoves {
    pub(crate) fn new<'tcx>(tcx: TyCtxt<'tcx>) -> Self {
        let compiler_copy = tcx.get_diagnostic_item(sym::compiler_copy);
        let compiler_move = tcx.get_diagnostic_item(sym::compiler_move);

        Self { compiler_copy, compiler_move }
    }

    /// Returns annotated SourceInfo for a move/copy operation without modifying anything. Used for
    /// Call arguments where we need to store SourceInfo separately. Returns None if the operand
    /// should not be annotated.
    fn get_annotated_source_info<'tcx>(
        &self,
        params: &mut Params<'_, 'tcx>,
        original_scope: SourceScope,
        op: &Operand<'tcx>,
    ) -> Option<SourceInfo> {
        let (place, profiling_marker) = match op {
            Operand::Move(place) => (place, self.compiler_move?),
            Operand::Copy(place) => (place, self.compiler_copy?),
            _ => return None,
        };

        let Params { tcx, typing_env, local_decls, size_limit, source_scopes } = params;

        let type_size =
            self.should_annotate_operation(*tcx, *typing_env, local_decls, place, *size_limit)?;

        let ty = place.ty(*local_decls, *tcx).ty;
        let callsite_span = source_scopes[original_scope].span;
        let new_scope = self.create_inlined_scope(
            *tcx,
            *typing_env,
            source_scopes,
            original_scope,
            callsite_span,
            profiling_marker,
            ty,
            type_size,
        );

        Some(SourceInfo { span: callsite_span, scope: new_scope })
    }

    /// If this is a Move or Copy of a concrete type, update its debug info to make it look like it
    /// was inlined from `core::profiling::compiler_move`/`compiler_copy`.
    ///
    /// Takes an explicit `original_scope` to use as the parent scope, which prevents chaining when
    /// multiple operands in the same statement are processed.
    ///
    /// The statement's span is NOT modified, so profilers will show the move at its actual source
    /// location rather than at profiling.rs. This provides more useful context about where the move
    /// occurs in the user's code.
    fn annotate_move<'tcx>(
        &self,
        params: &mut Params<'_, 'tcx>,
        source_info: &mut SourceInfo,
        original_scope: SourceScope,
        op: &Operand<'tcx>,
    ) {
        let (place, Some(profiling_marker)) = (match op {
            Operand::Move(place) => (place, self.compiler_move),
            Operand::Copy(place) => (place, self.compiler_copy),
            _ => return,
        }) else {
            return;
        };
        let Params { tcx, typing_env, local_decls, size_limit, source_scopes } = params;

        if let Some(type_size) =
            self.should_annotate_operation(*tcx, *typing_env, local_decls, place, *size_limit)
        {
            let ty = place.ty(*local_decls, *tcx).ty;
            let callsite_span = source_info.span;
            let new_scope = self.create_inlined_scope(
                *tcx,
                *typing_env,
                source_scopes,
                original_scope,
                callsite_span,
                profiling_marker,
                ty,
                type_size,
            );
            source_info.scope = new_scope;
            // Note: We deliberately do NOT modify source_info.span.
            // Keeping the original span means profilers show the actual source location
            // of the move/copy, which is more useful than showing profiling.rs:13.
            // The scope change is sufficient to make the move appear as an inlined call
            // to compiler_move/copy in the profiler.
        }
    }

    /// Determines if an operation should be annotated based on type characteristics.
    /// Returns Some(size) if it should be annotated, None otherwise.
    fn should_annotate_operation<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        local_decls: &rustc_index::IndexVec<Local, LocalDecl<'tcx>>,
        place: &Place<'tcx>,
        size_limit: u64,
    ) -> Option<u64> {
        let ty = place.ty(local_decls, tcx).ty;
        let layout = match tcx.layout_of(typing_env.as_query_input(ty)) {
            Ok(layout) => layout,
            Err(err) => {
                tracing::info!("Failed to get layout of {ty:?}: {err}");
                return None;
            }
        };

        let size = layout.size.bytes();

        // 1. Skip ZST types (no actual move/copy happens)
        if layout.is_zst() {
            return None;
        }

        // 2. Check size threshold (only annotate large moves/copies)
        if size < size_limit {
            return None;
        }

        // 3. Skip scalar/vector types that won't generate memcpy
        match layout.layout.backend_repr {
            rustc_abi::BackendRepr::Scalar(_)
            | rustc_abi::BackendRepr::ScalarPair(_, _)
            | rustc_abi::BackendRepr::SimdVector { .. } => None,
            _ => Some(size),
        }
    }

    /// Creates an inlined scope that makes operations appear to come from
    /// the specified compiler intrinsic function.
    fn create_inlined_scope<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        source_scopes: &mut IndexVec<SourceScope, SourceScopeData<'tcx>>,
        original_scope: SourceScope,
        callsite_span: rustc_span::Span,
        profiling_def_id: DefId,
        ty: Ty<'tcx>,
        type_size: u64,
    ) -> SourceScope {
        // Monomorphize the profiling marker for the actual type being moved/copied + size const
        // parameter compiler_move<T, const SIZE: usize> or compiler_copy<T, const SIZE: usize>
        let size_const = ty::Const::from_target_usize(tcx, type_size);
        let generic_args = tcx.mk_args(&[ty.into(), size_const.into()]);
        let profiling_instance = Instance::expect_resolve(
            tcx,
            typing_env,
            profiling_def_id,
            generic_args,
            callsite_span,
        );

        // Get the profiling marker's definition span to use as the scope's span
        // This ensures the file_start_pos/file_end_pos in the DebugScope match the DIScope's file
        let profiling_span = tcx.def_span(profiling_def_id);

        // Create new inlined scope that makes the operation appear to come from the profiling
        // marker
        let inlined_scope_data = SourceScopeData {
            // Use profiling_span so file bounds match the DIScope (profiling.rs)
            // This prevents DILexicalBlockFile mismatches that would show profiling.rs
            // with incorrect line numbers
            span: profiling_span,
            parent_scope: Some(original_scope),

            // The inlined field shows: (what was inlined, where it was called from)
            // - profiling_instance: the compiler_move/copy function that was "inlined"
            // - callsite_span: where the move/copy actually occurs in the user's code
            inlined: Some((profiling_instance, callsite_span)),

            // Proper inlined scope chaining to maintain debug info hierarchy
            // We need to find the first non-compiler_move inlined scope in the chain
            inlined_parent_scope: {
                let mut scope = original_scope;
                loop {
                    let scope_data = &source_scopes[scope];
                    if let Some((instance, _)) = scope_data.inlined {
                        // Check if this is a compiler_move/copy scope we created
                        if let Some(def_id) = instance.def_id().as_local() {
                            let def_id = Some(def_id.to_def_id());
                            if def_id == self.compiler_move || def_id == self.compiler_copy {
                                // This is one of our scopes, skip it and look at its inlined_parent_scope
                                if let Some(parent) = scope_data.inlined_parent_scope {
                                    scope = parent;
                                    continue;
                                } else {
                                    // No more parents, this is fine
                                    break None;
                                }
                            }
                        }
                        // This is a real inlined scope (not compiler_move/copy), use it
                        break Some(scope);
                    } else {
                        // Not an inlined scope, use its inlined_parent_scope
                        break scope_data.inlined_parent_scope;
                    }
                }
            },

            local_data: ClearCrossCrate::Clear,
        };

        // Add the new scope and return its index
        source_scopes.push(inlined_scope_data)
    }
}
