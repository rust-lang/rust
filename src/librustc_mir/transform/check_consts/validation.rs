//! The `Visitor` responsible for actually checking a `mir::Body` for invalid operations.

use rustc::hir::{self, def_id::DefId};
use rustc::mir::visit::{PlaceContext, Visitor, MutatingUseContext, NonMutatingUseContext};
use rustc::mir::*;
use rustc::ty::cast::CastTy;
use rustc::ty::{self, TyCtxt};
use rustc_index::bit_set::BitSet;
use rustc_target::spec::abi::Abi;
use syntax::symbol::sym;
use syntax_pos::Span;

use std::cell::RefCell;
use std::fmt;
use std::ops::Deref;

use crate::dataflow as old_dataflow;
use super::{Item, Qualif, is_lang_panic_fn};
use super::resolver::{FlowSensitiveResolver, IndirectlyMutableResults, QualifResolver};
use super::qualifs::{HasMutInterior, NeedsDrop};
use super::ops::{self, NonConstOp};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CheckOpResult {
    Forbidden,
    Unleashed,
    Allowed,
}

/// What kind of item we are in.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    /// A `static` item.
    Static,
    /// A `static mut` item.
    StaticMut,
    /// A `const fn` item.
    ConstFn,
    /// A `const` item or an anonymous constant (e.g. in array lengths).
    Const,
}

impl Mode {
    /// Returns the validation mode for the item with the given `DefId`, or `None` if this item
    /// does not require validation (e.g. a non-const `fn`).
    pub fn for_item(tcx: TyCtxt<'tcx>, def_id: DefId) -> Option<Self> {
        use hir::BodyOwnerKind as HirKind;

        let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();

        let mode = match tcx.hir().body_owner_kind(hir_id) {
            HirKind::Closure => return None,

            HirKind::Fn if tcx.is_const_fn(def_id) => Mode::ConstFn,
            HirKind::Fn => return None,

            HirKind::Const => Mode::Const,

            HirKind::Static(hir::MutImmutable) => Mode::Static,
            HirKind::Static(hir::MutMutable) => Mode::StaticMut,
        };

        Some(mode)
    }

    pub fn is_static(self) -> bool {
        match self {
            Mode::Static | Mode::StaticMut => true,
            Mode::ConstFn | Mode::Const => false,
        }
    }

    /// Returns `true` if the value returned by this item must be `Sync`.
    ///
    /// This returns false for `StaticMut` since all accesses to one are `unsafe` anyway.
    pub fn requires_sync(self) -> bool {
        match self {
            Mode::Static => true,
            Mode::ConstFn | Mode::Const |  Mode::StaticMut => false,
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Mode::Const => write!(f, "constant"),
            Mode::Static | Mode::StaticMut => write!(f, "static"),
            Mode::ConstFn => write!(f, "constant function"),
        }
    }
}

pub struct Qualifs<'a, 'mir, 'tcx> {
    has_mut_interior: FlowSensitiveResolver<'a, 'mir, 'tcx, HasMutInterior>,
    needs_drop: FlowSensitiveResolver<'a, 'mir, 'tcx, NeedsDrop>,
}

pub struct Validator<'a, 'mir, 'tcx> {
    item: &'a Item<'mir, 'tcx>,
    qualifs: Qualifs<'a, 'mir, 'tcx>,

    /// The span of the current statement.
    span: Span,

    /// True if the local was assigned the result of an illegal borrow (`ops::MutBorrow`).
    ///
    /// This is used to hide errors from {re,}borrowing the newly-assigned local, instead pointing
    /// the user to the place where the illegal borrow occurred. This set is only populated once an
    /// error has been emitted, so it will never cause an erroneous `mir::Body` to pass validation.
    ///
    /// FIXME(ecstaticmorse): assert at the end of checking that if `tcx.has_errors() == false`,
    /// this set is empty. Note that if we start removing locals from
    /// `derived_from_illegal_borrow`, just checking at the end won't be enough.
    derived_from_illegal_borrow: BitSet<Local>,

    errors: Vec<(Span, String)>,

    /// Whether to actually emit errors or just store them in `errors`.
    pub(crate) suppress_errors: bool,
}

impl Deref for Validator<'_, 'mir, 'tcx> {
    type Target = Item<'mir, 'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

pub fn compute_indirectly_mutable_locals<'mir, 'tcx>(
    item: &Item<'mir, 'tcx>,
) -> RefCell<IndirectlyMutableResults<'mir, 'tcx>> {
    let dead_unwinds = BitSet::new_empty(item.body.basic_blocks().len());

    let indirectly_mutable_locals = old_dataflow::do_dataflow(
        item.tcx,
        item.body,
        item.def_id,
        &[],
        &dead_unwinds,
        old_dataflow::IndirectlyMutableLocals::new(item.tcx, item.body, item.param_env),
        |_, local| old_dataflow::DebugFormatted::new(&local),
    );

    let indirectly_mutable_locals = old_dataflow::DataflowResultsCursor::new(
        indirectly_mutable_locals,
        item.body,
    );

    RefCell::new(indirectly_mutable_locals)
}

impl Validator<'a, 'mir, 'tcx> {
    pub fn new(
        item: &'a Item<'mir, 'tcx>,
        indirectly_mutable_locals: &'a RefCell<IndirectlyMutableResults<'mir, 'tcx>>,
    ) -> Self {
        let dead_unwinds = BitSet::new_empty(item.body.basic_blocks().len());

        let needs_drop = FlowSensitiveResolver::new(
            NeedsDrop,
            item,
            indirectly_mutable_locals,
            &dead_unwinds,
        );

        let has_mut_interior = FlowSensitiveResolver::new(
            HasMutInterior,
            item,
            indirectly_mutable_locals,
            &dead_unwinds,
        );

        let qualifs = Qualifs {
            needs_drop,
            has_mut_interior,
        };

        Validator {
            span: item.body.span,
            item,
            qualifs,
            errors: vec![],
            derived_from_illegal_borrow: BitSet::new_empty(item.body.local_decls.len()),
            suppress_errors: false,
        }
    }

    /// Resets the `QualifResolver`s used by this `Validator` and returns them so they can be
    /// reused.
    pub fn into_qualifs(mut self) -> Qualifs<'a, 'mir, 'tcx> {
        self.qualifs.needs_drop.reset();
        self.qualifs.has_mut_interior.reset();
        self.qualifs
    }

    pub fn take_errors(&mut self) -> Vec<(Span, String)> {
        std::mem::replace(&mut self.errors, vec![])
    }

    /// Emits an error at the given `span` if an expression cannot be evaluated in the current
    /// context. Returns `Forbidden` if an error was emitted.
    pub fn check_op_spanned<O>(&mut self, op: O, span: Span) -> CheckOpResult
    where
        O: NonConstOp + fmt::Debug
    {
        trace!("check_op: op={:?}", op);

        if op.is_allowed_in_item(self) {
            return CheckOpResult::Allowed;
        }

        // If an operation is supported in miri (and is not already controlled by a feature gate) it
        // can be turned on with `-Zunleash-the-miri-inside-of-you`.
        let is_unleashable = O::IS_SUPPORTED_IN_MIRI
            && O::feature_gate(self.tcx).is_none();

        if is_unleashable && self.tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you {
            self.tcx.sess.span_warn(span, "skipping const checks");
            return CheckOpResult::Unleashed;
        }

        if !self.suppress_errors {
            op.emit_error(self, span);
        }

        self.errors.push((span, format!("{:?}", op)));
        CheckOpResult::Forbidden
    }

    /// Emits an error if an expression cannot be evaluated in the current context.
    pub fn check_op(&mut self, op: impl NonConstOp + fmt::Debug) -> CheckOpResult {
        let span = self.span;
        self.check_op_spanned(op, span)
    }
}

impl Visitor<'tcx> for Validator<'_, 'mir, 'tcx> {
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        trace!("visit_rvalue: rvalue={:?} location={:?}", rvalue, location);

        // Check nested operands and places.
        if let Rvalue::Ref(_, kind, ref place) = *rvalue {
            // Special-case reborrows to be more like a copy of a reference.
            let mut reborrow_place = None;
            if let box [proj_base @ .., elem] = &place.projection {
                if *elem == ProjectionElem::Deref {
                    let base_ty = Place::ty_from(&place.base, proj_base, self.body, self.tcx).ty;
                    if let ty::Ref(..) = base_ty.kind {
                        reborrow_place = Some(proj_base);
                    }
                }
            }

            if let Some(proj) = reborrow_place {
                let ctx = match kind {
                    BorrowKind::Shared => PlaceContext::NonMutatingUse(
                        NonMutatingUseContext::SharedBorrow,
                    ),
                    BorrowKind::Shallow => PlaceContext::NonMutatingUse(
                        NonMutatingUseContext::ShallowBorrow,
                    ),
                    BorrowKind::Unique => PlaceContext::NonMutatingUse(
                        NonMutatingUseContext::UniqueBorrow,
                    ),
                    BorrowKind::Mut { .. } => PlaceContext::MutatingUse(
                        MutatingUseContext::Borrow,
                    ),
                };
                self.visit_place_base(&place.base, ctx, location);
                self.visit_projection(&place.base, proj, ctx, location);
            } else {
                self.super_rvalue(rvalue, location);
            }
        } else {
            self.super_rvalue(rvalue, location);
        }

        match *rvalue {
            Rvalue::Use(_) |
            Rvalue::Repeat(..) |
            Rvalue::UnaryOp(UnOp::Neg, _) |
            Rvalue::UnaryOp(UnOp::Not, _) |
            Rvalue::NullaryOp(NullOp::SizeOf, _) |
            Rvalue::CheckedBinaryOp(..) |
            Rvalue::Cast(CastKind::Pointer(_), ..) |
            Rvalue::Discriminant(..) |
            Rvalue::Len(_) |
            Rvalue::Ref(..) |
            Rvalue::Aggregate(..) => {}

            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) => {
                let operand_ty = operand.ty(self.body, self.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");

                if let (CastTy::Ptr(_), CastTy::Int(_))
                     | (CastTy::FnPtr,  CastTy::Int(_)) = (cast_in, cast_out) {
                    self.check_op(ops::RawPtrToIntCast);
                }
            }

            Rvalue::BinaryOp(op, ref lhs, _) => {
                if let ty::RawPtr(_) | ty::FnPtr(..) = lhs.ty(self.body, self.tcx).kind {
                    assert!(op == BinOp::Eq || op == BinOp::Ne ||
                            op == BinOp::Le || op == BinOp::Lt ||
                            op == BinOp::Ge || op == BinOp::Gt ||
                            op == BinOp::Offset);


                    self.check_op(ops::RawPtrComparison);
                }
            }

            Rvalue::NullaryOp(NullOp::Box, _) => {
                self.check_op(ops::HeapAllocation);
            }
        }
    }

    fn visit_place_base(
        &mut self,
        place_base: &PlaceBase<'tcx>,
        context: PlaceContext,
        location: Location,
    ) {
        trace!(
            "visit_place_base: place_base={:?} context={:?} location={:?}",
            place_base,
            context,
            location,
        );
        self.super_place_base(place_base, context, location);

        match place_base {
            PlaceBase::Local(_) => {}
            PlaceBase::Static(box Static{ kind: StaticKind::Promoted(_, _), .. }) => {
                bug!("Promotion must be run after const validation");
            }

            PlaceBase::Static(box Static{ kind: StaticKind::Static, def_id, .. }) => {
                let is_thread_local = self.tcx.has_attr(*def_id, sym::thread_local);
                if is_thread_local {
                    self.check_op(ops::ThreadLocalAccess);
                } else if self.mode == Mode::Static && context.is_mutating_use() {
                    // this is not strictly necessary as miri will also bail out
                    // For interior mutability we can't really catch this statically as that
                    // goes through raw pointers and intermediate temporaries, so miri has
                    // to catch this anyway

                    self.tcx.sess.span_err(
                        self.span,
                        "cannot mutate statics in the initializer of another static",
                    );
                } else {
                    self.check_op(ops::StaticAccess);
                }
            }
        }
    }

    fn visit_assign(&mut self, dest: &Place<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        trace!("visit_assign: dest={:?} rvalue={:?} location={:?}", dest, rvalue, location);

        // Error on mutable borrows or shared borrows of values with interior mutability.
        //
        // This replicates the logic at the start of `assign` in the old const checker.  Note that
        // it depends on `HasMutInterior` being set for mutable borrows as well as values with
        // interior mutability.
        if let Rvalue::Ref(_, kind, ref borrowed_place) = *rvalue {
            let rvalue_has_mut_interior = HasMutInterior::in_rvalue(
                &self.item,
                self.qualifs.has_mut_interior.get(),
                rvalue,
            );

            if rvalue_has_mut_interior {
                let is_derived_from_illegal_borrow = match *borrowed_place {
                    // If an unprojected local was borrowed and its value was the result of an
                    // illegal borrow, suppress this error and mark the result of this borrow as
                    // illegal as well.
                    Place { base: PlaceBase::Local(borrowed_local), projection: box [] }
                        if self.derived_from_illegal_borrow.contains(borrowed_local) => true,

                    // Otherwise proceed normally: check the legality of a mutable borrow in this
                    // context.
                    _ => self.check_op(ops::MutBorrow(kind)) == CheckOpResult::Forbidden,
                };

                // When the target of the assignment is a local with no projections, mark it as
                // derived from an illegal borrow if necessary.
                //
                // FIXME: should we also clear `derived_from_illegal_borrow` when a local is
                // assigned a new value?
                if is_derived_from_illegal_borrow {
                    if let Place { base: PlaceBase::Local(dest), projection: box [] } = *dest {
                        self.derived_from_illegal_borrow.insert(dest);
                    }
                }
            }
        }

        self.super_assign(dest, rvalue, location);
    }

    fn visit_projection(
        &mut self,
        place_base: &PlaceBase<'tcx>,
        proj: &[PlaceElem<'tcx>],
        context: PlaceContext,
        location: Location,
    ) {
        trace!(
            "visit_place_projection: proj={:?} context={:?} location={:?}",
            proj,
            context,
            location,
        );
        self.super_projection(place_base, proj, context, location);

        let (elem, proj_base) = match proj.split_last() {
            Some(x) => x,
            None => return,
        };

        match elem {
            ProjectionElem::Deref => {
                if context.is_mutating_use() {
                    self.check_op(ops::MutDeref);
                }

                let base_ty = Place::ty_from(place_base, proj_base, self.body, self.tcx).ty;
                if let ty::RawPtr(_) = base_ty.kind {
                    self.check_op(ops::RawPtrDeref);
                }
            }

            ProjectionElem::ConstantIndex {..} |
            ProjectionElem::Subslice {..} |
            ProjectionElem::Field(..) |
            ProjectionElem::Index(_) => {
                let base_ty = Place::ty_from(place_base, proj_base, self.body, self.tcx).ty;
                match base_ty.ty_adt_def() {
                    Some(def) if def.is_union() => {
                        self.check_op(ops::UnionAccess);
                    }

                    _ => {}
                }
            }

            ProjectionElem::Downcast(..) => {
                self.check_op(ops::Downcast);
            }
        }
    }


    fn visit_source_info(&mut self, source_info: &SourceInfo) {
        trace!("visit_source_info: source_info={:?}", source_info);
        self.span = source_info.span;
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        trace!("visit_statement: statement={:?} location={:?}", statement, location);

        self.qualifs.needs_drop.visit_statement(statement, location);
        self.qualifs.has_mut_interior.visit_statement(statement, location);

        match statement.kind {
            StatementKind::Assign(..) => {
                self.super_statement(statement, location);
            }
            StatementKind::FakeRead(FakeReadCause::ForMatchedPlace, _) => {
                self.check_op(ops::IfOrMatch);
            }
            // FIXME(eddyb) should these really do nothing?
            StatementKind::FakeRead(..) |
            StatementKind::SetDiscriminant { .. } |
            StatementKind::StorageLive(_) |
            StatementKind::StorageDead(_) |
            StatementKind::InlineAsm {..} |
            StatementKind::Retag { .. } |
            StatementKind::AscribeUserType(..) |
            StatementKind::Nop => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        trace!("visit_terminator: terminator={:?} location={:?}", terminator, location);

        self.qualifs.needs_drop.visit_terminator(terminator, location);
        self.qualifs.has_mut_interior.visit_terminator(terminator, location);

        self.super_terminator(terminator, location);
    }

    fn visit_terminator_kind(&mut self, kind: &TerminatorKind<'tcx>, location: Location) {
        trace!("visit_terminator_kind: kind={:?} location={:?}", kind, location);
        self.super_terminator_kind(kind, location);

        match kind {
            TerminatorKind::Call { func, .. } => {
                let fn_ty = func.ty(self.body, self.tcx);

                let def_id = match fn_ty.kind {
                    ty::FnDef(def_id, _) => def_id,

                    ty::FnPtr(_) => {
                        self.check_op(ops::FnCallIndirect);
                        return;
                    }
                    _ => {
                        self.check_op(ops::FnCallOther);
                        return;
                    }
                };

                // At this point, we are calling a function whose `DefId` is known...

                if let Abi::RustIntrinsic | Abi::PlatformIntrinsic = self.tcx.fn_sig(def_id).abi() {
                    assert!(!self.tcx.is_const_fn(def_id));

                    if self.tcx.item_name(def_id) == sym::transmute {
                        self.check_op(ops::Transmute);
                        return;
                    }

                    // To preserve the current semantics, we return early, allowing all
                    // intrinsics (except `transmute`) to pass unchecked to miri.
                    //
                    // FIXME: We should keep a whitelist of allowed intrinsics (or at least a
                    // blacklist of unimplemented ones) and fail here instead.
                    return;
                }

                if self.tcx.is_const_fn(def_id) {
                    return;
                }

                if is_lang_panic_fn(self.tcx, def_id) {
                    self.check_op(ops::Panic);
                } else if let Some(feature) = self.tcx.is_unstable_const_fn(def_id) {
                    // Exempt unstable const fns inside of macros with
                    // `#[allow_internal_unstable]`.
                    if !self.span.allows_unstable(feature) {
                        self.check_op(ops::FnCallUnstable(def_id, feature));
                    }
                } else {
                    self.check_op(ops::FnCallNonConst(def_id));
                }

            }

            // Forbid all `Drop` terminators unless the place being dropped is a local with no
            // projections that cannot be `NeedsDrop`.
            | TerminatorKind::Drop { location: dropped_place, .. }
            | TerminatorKind::DropAndReplace { location: dropped_place, .. }
            => {
                let mut err_span = self.span;

                // Check to see if the type of this place can ever have a drop impl. If not, this
                // `Drop` terminator is frivolous.
                let ty_needs_drop = dropped_place
                    .ty(self.body, self.tcx)
                    .ty
                    .needs_drop(self.tcx, self.param_env);

                if !ty_needs_drop {
                    return;
                }

                let needs_drop = if let Place {
                    base: PlaceBase::Local(local),
                    projection: box [],
                } = *dropped_place {
                    // Use the span where the local was declared as the span of the drop error.
                    err_span = self.body.local_decls[local].source_info.span;
                    self.qualifs.needs_drop.contains(local)
                } else {
                    true
                };

                if needs_drop {
                    self.check_op_spanned(ops::LiveDrop, err_span);
                }
            }

            _ => {}
        }
    }
}
