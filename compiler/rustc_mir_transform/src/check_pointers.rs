use rustc_hir::lang_items::LangItem;
use rustc_index::IndexVec;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use tracing::{debug, trace};

/// Details of a pointer check, the condition on which we decide whether to
/// fail the assert and an [AssertKind] that defines the behavior on failure.
pub(crate) struct PointerCheck<'tcx> {
    pub(crate) cond: Operand<'tcx>,
    pub(crate) assert_kind: Box<AssertKind<Operand<'tcx>>>,
}

/// Indicates whether we insert the checks for borrow places of a raw pointer.
/// Concretely places with [MutatingUseContext::Borrow] or
/// [NonMutatingUseContext::SharedBorrow].
#[derive(Copy, Clone)]
pub(crate) enum BorrowCheckMode {
    IncludeBorrows,
    ExcludeBorrows,
}

/// Utility for adding a check for read/write on every sized, raw pointer.
///
/// Visits every read/write access to a [Sized], raw pointer and inserts a
/// new basic block directly before the pointer access. (Read/write accesses
/// are determined by the `PlaceContext` of the MIR visitor.) Then calls
/// `on_finding` to insert the actual logic for a pointer check (e.g. check for
/// alignment). A check can choose to be inserted for (mutable) borrows of
/// raw pointers via the `borrow_check_mode` parameter.
///
/// This utility takes care of the right order of blocks, the only thing a
/// caller must do in `on_finding` is:
/// - Append [Statement]s to `stmts`.
/// - Append [LocalDecl]s to `local_decls`.
/// - Return a [PointerCheck] that contains the condition and an [AssertKind].
///   The AssertKind must be a panic with `#[rustc_nounwind]`. The condition
///   should always return the boolean `is_ok`, so evaluate to true in case of
///   success and fail the check otherwise.
/// This utility will insert a terminator block that asserts on the condition
/// and panics on failure.
pub(crate) fn check_pointers<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    excluded_pointees: &[Ty<'tcx>],
    on_finding: F,
    borrow_check_mode: BorrowCheckMode,
) where
    F: Fn(
        /* tcx: */ TyCtxt<'tcx>,
        /* pointer: */ Place<'tcx>,
        /* pointee_ty: */ Ty<'tcx>,
        /* context: */ PlaceContext,
        /* local_decls: */ &mut IndexVec<Local, LocalDecl<'tcx>>,
        /* stmts: */ &mut Vec<Statement<'tcx>>,
        /* source_info: */ SourceInfo,
    ) -> PointerCheck<'tcx>,
{
    // This pass emits new panics. If for whatever reason we do not have a panic
    // implementation, running this pass may cause otherwise-valid code to not compile.
    if tcx.lang_items().get(LangItem::PanicImpl).is_none() {
        return;
    }

    let typing_env = body.typing_env(tcx);
    let basic_blocks = body.basic_blocks.as_mut();
    let local_decls = &mut body.local_decls;

    // This operation inserts new blocks. Each insertion changes the Location for all
    // statements/blocks after. Iterating or visiting the MIR in order would require updating
    // our current location after every insertion. By iterating backwards, we dodge this issue:
    // The only Locations that an insertion changes have already been handled.
    for block in basic_blocks.indices().rev() {
        for statement_index in (0..basic_blocks[block].statements.len()).rev() {
            let location = Location { block, statement_index };
            let statement = &basic_blocks[block].statements[statement_index];
            let source_info = statement.source_info;

            let mut finder = PointerFinder::new(
                tcx,
                local_decls,
                typing_env,
                excluded_pointees,
                borrow_check_mode,
            );
            finder.visit_statement(statement, location);

            for (local, ty, context) in finder.into_found_pointers() {
                debug!("Inserting check for {:?}", ty);
                let new_block = split_block(basic_blocks, location);

                // Invoke `on_finding` which appends to `local_decls` and the
                // blocks statements. It returns information about the assert
                // we're performing in the Terminator.
                let block_data = &mut basic_blocks[block];
                let pointer_check = on_finding(
                    tcx,
                    local,
                    ty,
                    context,
                    local_decls,
                    &mut block_data.statements,
                    source_info,
                );
                block_data.terminator = Some(Terminator {
                    source_info,
                    kind: TerminatorKind::Assert {
                        cond: pointer_check.cond,
                        expected: true,
                        target: new_block,
                        msg: pointer_check.assert_kind,
                        // This calls a panic function associated with the pointer check, which
                        // is #[rustc_nounwind]. We never want to insert an unwind into unsafe
                        // code, because unwinding could make a failing UB check turn into much
                        // worse UB when we start unwinding.
                        unwind: UnwindAction::Unreachable,
                    },
                });
            }
        }
    }
}

struct PointerFinder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a mut LocalDecls<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    pointers: Vec<(Place<'tcx>, Ty<'tcx>, PlaceContext)>,
    excluded_pointees: &'a [Ty<'tcx>],
    borrow_check_mode: BorrowCheckMode,
}

impl<'a, 'tcx> PointerFinder<'a, 'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        local_decls: &'a mut LocalDecls<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        excluded_pointees: &'a [Ty<'tcx>],
        borrow_check_mode: BorrowCheckMode,
    ) -> Self {
        PointerFinder {
            tcx,
            local_decls,
            typing_env,
            excluded_pointees,
            pointers: Vec::new(),
            borrow_check_mode,
        }
    }

    fn into_found_pointers(self) -> Vec<(Place<'tcx>, Ty<'tcx>, PlaceContext)> {
        self.pointers
    }

    /// Whether or not we should visit a [Place] with [PlaceContext].
    ///
    /// We generally only visit Reads/Writes to a place and only Borrows if
    /// requested.
    fn should_visit_place(&self, context: PlaceContext) -> bool {
        match context {
            PlaceContext::MutatingUse(
                MutatingUseContext::Store
                | MutatingUseContext::Call
                | MutatingUseContext::Yield
                | MutatingUseContext::Drop,
            ) => true,
            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Copy | NonMutatingUseContext::Move,
            ) => true,
            PlaceContext::MutatingUse(MutatingUseContext::Borrow)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::SharedBorrow) => {
                matches!(self.borrow_check_mode, BorrowCheckMode::IncludeBorrows)
            }
            _ => false,
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for PointerFinder<'a, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        if !self.should_visit_place(context) || !place.is_indirect() {
            return;
        }

        // Since Deref projections must come first and only once, the pointer for an indirect place
        // is the Local that the Place is based on.
        let pointer = Place::from(place.local);
        let pointer_ty = self.local_decls[place.local].ty;

        // We only want to check places based on raw pointers
        if !pointer_ty.is_raw_ptr() {
            trace!("Indirect, but not based on an raw ptr, not checking {:?}", place);
            return;
        }

        let pointee_ty =
            pointer_ty.builtin_deref(true).expect("no builtin_deref for an raw pointer");
        // Ideally we'd support this in the future, but for now we are limited to sized types.
        if !pointee_ty.is_sized(self.tcx, self.typing_env) {
            trace!("Raw pointer, but pointee is not known to be sized: {:?}", pointer_ty);
            return;
        }

        // We don't need to look for slices, we already rejected unsized types above.
        let element_ty = match pointee_ty.kind() {
            ty::Array(ty, _) => *ty,
            _ => pointee_ty,
        };
        if self.excluded_pointees.contains(&element_ty) {
            trace!("Skipping pointer for type: {:?}", pointee_ty);
            return;
        }

        self.pointers.push((pointer, pointee_ty, context));

        self.super_place(place, context, location);
    }
}

fn split_block(
    basic_blocks: &mut IndexVec<BasicBlock, BasicBlockData<'_>>,
    location: Location,
) -> BasicBlock {
    let block_data = &mut basic_blocks[location.block];

    // Drain every statement after this one and move the current terminator to a new basic block.
    let new_block = BasicBlockData {
        statements: block_data.statements.split_off(location.statement_index),
        terminator: block_data.terminator.take(),
        is_cleanup: block_data.is_cleanup,
    };

    basic_blocks.push(new_block)
}
