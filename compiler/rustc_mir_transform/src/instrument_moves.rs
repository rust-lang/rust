//! Instrumentation pass for move/copy operations.
//!
//! This pass modifies the source scopes of statements containing `Operand::Move` and `Operand::Copy`
//! to make them appear as if they were inlined from `compiler_move()` and `compiler_copy()` intrinsic
//! functions. This creates the illusion that moves/copies are function calls in debuggers and
//! profilers, making them visible for performance analysis.
//!
//! The pass leverages the existing inlining infrastructure by creating synthetic `SourceScopeData`
//! with the `inlined` field set to point to the appropriate intrinsic function.

use rustc_index::IndexVec;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt, TypingEnv};
use rustc_session::config::DebugInfo;
use rustc_span::sym;

/// Default minimum size in bytes for move/copy operations to be instrumented. Set to 64+1 bytes
/// (typical cache line size) to focus on potentially expensive operations.
const DEFAULT_INSTRUMENT_MOVES_SIZE_LIMIT: u64 = 65;

#[derive(Copy, Clone, Debug)]
enum Operation {
    Move,
    Copy,
}

/// Bundle up parameters into a structure to make repeated calling neater
struct Params<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    source_scopes: &'a mut IndexVec<SourceScope, SourceScopeData<'tcx>>,
    local_decls: &'a IndexVec<Local, LocalDecl<'tcx>>,
    typing_env: TypingEnv<'tcx>,
    size_limit: u64,
}

/// MIR transform that instruments move/copy operations for profiler visibility.
pub(crate) struct InstrumentMoves;

impl<'tcx> crate::MirPass<'tcx> for InstrumentMoves {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.opts.unstable_opts.instrument_moves && sess.opts.debuginfo != DebugInfo::None
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
            .instrument_moves_size_limit
            .unwrap_or(DEFAULT_INSTRUMENT_MOVES_SIZE_LIMIT);

        // Common params, including selectively borrowing the bits of Body we need to avoid
        // mut/non-mut aliasing problems.
        let mut params = Params {
            tcx,
            source_scopes: &mut body.source_scopes,
            local_decls: &body.local_decls,
            typing_env,
            size_limit,
        };

        // Process each basic block
        for block_data in body.basic_blocks.as_mut() {
            for stmt in &mut block_data.statements {
                let source_info = &mut stmt.source_info;

                if let StatementKind::Assign(box (_, rvalue)) = &stmt.kind {
                    match rvalue {
                        Rvalue::Use(op)
                        | Rvalue::Repeat(op, _)
                        | Rvalue::Cast(_, op, _)
                        | Rvalue::UnaryOp(_, op) => {
                            self.annotate_move(&mut params, source_info, op);
                        }
                        Rvalue::BinaryOp(_, box (lop, rop)) => {
                            self.annotate_move(&mut params, source_info, lop);
                            self.annotate_move(&mut params, source_info, rop);
                        }
                        Rvalue::Aggregate(_, ops) => {
                            for op in ops {
                                self.annotate_move(&mut params, source_info, op);
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
                match &terminator.kind {
                    TerminatorKind::Call { func, args, .. } => {
                        // Instrument the function operand
                        self.annotate_move(&mut params, source_info, func);
                        // Instrument each argument
                        for arg in &*args {
                            self.annotate_move(&mut params, source_info, &arg.node);
                        }
                    }
                    TerminatorKind::SwitchInt { discr, .. } => {
                        self.annotate_move(&mut params, source_info, discr);
                    }
                    _ => {} // Other terminators don't have operands
                }
            }
        }
    }

    fn is_required(&self) -> bool {
        false // Optional optimization/instrumentation pass
    }
}

impl InstrumentMoves {
    /// If this is a Move or Copy of a concrete type, update its debug info to make it look like it
    /// was inlined from `core::profiling::compiler_move`/`compiler_copy`.
    fn annotate_move<'tcx>(
        &self,
        params: &mut Params<'_, 'tcx>,
        source_info: &mut SourceInfo,
        op: &Operand<'tcx>,
    ) {
        let (place, operation) = match op {
            Operand::Move(place) => (place, Operation::Move),
            Operand::Copy(place) => (place, Operation::Copy),
            _ => return,
        };
        let Params { tcx, typing_env, local_decls, size_limit, source_scopes } = params;

        if let Some(type_size) =
            self.should_instrument_operation(*tcx, *typing_env, local_decls, place, *size_limit)
        {
            let ty = place.ty(*local_decls, *tcx).ty;
            source_info.scope = self.create_inlined_scope(
                *tcx,
                *typing_env,
                source_scopes,
                source_info,
                operation,
                ty,
                type_size,
            );
        }
    }

    /// Determines if an operation should be instrumented based on type characteristics.
    /// Returns Some(size) if it should be instrumented, None otherwise.
    fn should_instrument_operation<'tcx>(
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

        // 2. Check size threshold (only instrument large moves/copies)
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
        original_source_info: &SourceInfo,
        operation: Operation,
        ty: Ty<'tcx>,
        type_size: u64,
    ) -> SourceScope {
        let intrinsic_def_id = match operation {
            Operation::Move => tcx.get_diagnostic_item(sym::compiler_move),
            Operation::Copy => tcx.get_diagnostic_item(sym::compiler_copy),
        };

        let Some(intrinsic_def_id) = intrinsic_def_id else {
            // Shouldn't happen, but just return original scope if it does
            return original_source_info.scope;
        };

        // Monomorphize the intrinsic for the actual type being moved/copied + size const parameter
        // compiler_move<T, const SIZE: usize> or compiler_copy<T, const SIZE: usize>
        let size_const = ty::Const::from_target_usize(tcx, type_size);
        let generic_args = tcx.mk_args(&[ty.into(), size_const.into()]);
        let intrinsic_instance = Instance::expect_resolve(
            tcx,
            typing_env,
            intrinsic_def_id,
            generic_args,
            original_source_info.span,
        );

        // Create new inlined scope that makes the operation appear to come from the intrinsic
        let inlined_scope_data = SourceScopeData {
            span: original_source_info.span,
            parent_scope: Some(original_source_info.scope),

            // Pretend this op is inlined from the intrinsic
            inlined: Some((intrinsic_instance, original_source_info.span)),

            // Proper inlined scope chaining to maintain debug info hierarchy
            inlined_parent_scope: {
                let parent_scope = &source_scopes[original_source_info.scope];
                if parent_scope.inlined.is_some() {
                    // If parent is already inlined, chain through it
                    Some(original_source_info.scope)
                } else {
                    // Otherwise, use the parent's inlined_parent_scope
                    parent_scope.inlined_parent_scope
                }
            },

            local_data: ClearCrossCrate::Clear,
        };

        // Add the new scope
        source_scopes.push(inlined_scope_data)
    }
}
