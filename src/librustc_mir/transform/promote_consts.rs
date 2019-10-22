//! A pass that promotes borrows of constant rvalues.
//!
//! The rvalues considered constant are trees of temps,
//! each with exactly one initialization, and holding
//! a constant value with no interior mutability.
//! They are placed into a new MIR constant body in
//! `promoted` and the borrow rvalue is replaced with
//! a `Literal::Promoted` using the index into `promoted`
//! of that constant MIR.
//!
//! This pass assumes that every use is dominated by an
//! initialization and can otherwise silence errors, if
//! move analysis runs after promotion on broken MIR.

use rustc::hir::def_id::DefId;
use rustc::mir::*;
use rustc::mir::interpret::ConstValue;
use rustc::mir::visit::{PlaceContext, MutatingUseContext, MutVisitor, Visitor};
use rustc::mir::traversal::ReversePostorder;
use rustc::ty::{self, List, TyCtxt};
use rustc::ty::subst::InternalSubsts;
use rustc::ty::cast::CastTy;
use syntax::ast::LitKind;
use syntax::symbol::sym;
use syntax_pos::{Span, DUMMY_SP};

use rustc_index::vec::{IndexVec, Idx};
use rustc_target::spec::abi::Abi;

use std::{iter, mem, usize};

use crate::transform::check_consts::{qualifs, Item, ConstKind, is_lang_panic_fn};

/// State of a temporary during collection and promotion.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TempState {
    /// No references to this temp.
    Undefined,
    /// One direct assignment and any number of direct uses.
    /// A borrow of this temp is promotable if the assigned
    /// value is qualified as constant.
    Defined {
        location: Location,
        uses: usize
    },
    /// Any other combination of assignments/uses.
    Unpromotable,
    /// This temp was part of an rvalue which got extracted
    /// during promotion and needs cleanup.
    PromotedOut
}

impl TempState {
    pub fn is_promotable(&self) -> bool {
        debug!("is_promotable: self={:?}", self);
        if let TempState::Defined { .. } = *self {
            true
        } else {
            false
        }
    }
}

/// A "root candidate" for promotion, which will become the
/// returned value in a promoted MIR, unless it's a subset
/// of a larger candidate.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Candidate {
    /// Borrow of a constant temporary.
    Ref(Location),

    /// Promotion of the `x` in `[x; 32]`.
    Repeat(Location),

    /// Currently applied to function calls where the callee has the unstable
    /// `#[rustc_args_required_const]` attribute as well as the SIMD shuffle
    /// intrinsic. The intrinsic requires the arguments are indeed constant and
    /// the attribute currently provides the semantic requirement that arguments
    /// must be constant.
    Argument { bb: BasicBlock, index: usize },
}

fn args_required_const(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Vec<usize>> {
    let attrs = tcx.get_attrs(def_id);
    let attr = attrs.iter().find(|a| a.check_name(sym::rustc_args_required_const))?;
    let mut ret = vec![];
    for meta in attr.meta_item_list()? {
        match meta.literal()?.kind {
            LitKind::Int(a, _) => { ret.push(a as usize); }
            _ => return None,
        }
    }
    Some(ret)
}

struct Collector<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    temps: IndexVec<Local, TempState>,
    candidates: Vec<Candidate>,
    span: Span,
}

impl<'tcx> Visitor<'tcx> for Collector<'_, 'tcx> {
    fn visit_local(&mut self,
                   &index: &Local,
                   context: PlaceContext,
                   location: Location) {
        debug!("visit_local: index={:?} context={:?} location={:?}", index, context, location);
        // We're only interested in temporaries and the return place
        match self.body.local_kind(index) {
            | LocalKind::Temp
            | LocalKind::ReturnPointer
            => {},
            | LocalKind::Arg
            | LocalKind::Var
            => return,
        }

        // Ignore drops, if the temp gets promoted,
        // then it's constant and thus drop is noop.
        // Non-uses are also irrelevent.
        if context.is_drop() || !context.is_use() {
            debug!(
                "visit_local: context.is_drop={:?} context.is_use={:?}",
                context.is_drop(), context.is_use(),
            );
            return;
        }

        let temp = &mut self.temps[index];
        debug!("visit_local: temp={:?}", temp);
        if *temp == TempState::Undefined {
            match context {
                PlaceContext::MutatingUse(MutatingUseContext::Store) |
                PlaceContext::MutatingUse(MutatingUseContext::Call) => {
                    *temp = TempState::Defined {
                        location,
                        uses: 0
                    };
                    return;
                }
                _ => { /* mark as unpromotable below */ }
            }
        } else if let TempState::Defined { ref mut uses, .. } = *temp {
            // We always allow borrows, even mutable ones, as we need
            // to promote mutable borrows of some ZSTs e.g., `&mut []`.
            let allowed_use = context.is_borrow() || context.is_nonmutating_use();
            debug!("visit_local: allowed_use={:?}", allowed_use);
            if allowed_use {
                *uses += 1;
                return;
            }
            /* mark as unpromotable below */
        }
        *temp = TempState::Unpromotable;
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);

        match *rvalue {
            Rvalue::Ref(..) => {
                self.candidates.push(Candidate::Ref(location));
            }
            Rvalue::Repeat(..) if self.tcx.features().const_in_array_repeat_expressions => {
                // FIXME(#49147) only promote the element when it isn't `Copy`
                // (so that code that can copy it at runtime is unaffected).
                self.candidates.push(Candidate::Repeat(location));
            }
            _ => {}
        }
    }

    fn visit_terminator_kind(&mut self,
                             kind: &TerminatorKind<'tcx>,
                             location: Location) {
        self.super_terminator_kind(kind, location);

        if let TerminatorKind::Call { ref func, .. } = *kind {
            if let ty::FnDef(def_id, _) = func.ty(self.body, self.tcx).kind {
                let fn_sig = self.tcx.fn_sig(def_id);
                if let Abi::RustIntrinsic | Abi::PlatformIntrinsic = fn_sig.abi() {
                    let name = self.tcx.item_name(def_id);
                    // FIXME(eddyb) use `#[rustc_args_required_const(2)]` for shuffles.
                    if name.as_str().starts_with("simd_shuffle") {
                        self.candidates.push(Candidate::Argument {
                            bb: location.block,
                            index: 2,
                        });
                    }
                }

                if let Some(constant_args) = args_required_const(self.tcx, def_id) {
                    for index in constant_args {
                        self.candidates.push(Candidate::Argument { bb: location.block, index });
                    }
                }
            }
        }
    }

    fn visit_source_info(&mut self, source_info: &SourceInfo) {
        self.span = source_info.span;
    }
}

pub fn collect_temps_and_candidates(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    rpo: &mut ReversePostorder<'_, 'tcx>,
) -> (IndexVec<Local, TempState>, Vec<Candidate>) {
    let mut collector = Collector {
        tcx,
        body,
        temps: IndexVec::from_elem(TempState::Undefined, &body.local_decls),
        candidates: vec![],
        span: body.span,
    };
    for (bb, data) in rpo {
        collector.visit_basic_block_data(bb, data);
    }
    (collector.temps, collector.candidates)
}

/// Checks whether locals that appear in a promotion context (`Candidate`) are actually promotable.
///
/// This wraps an `Item`, and has access to all fields of that `Item` via `Deref` coercion.
struct Validator<'a, 'tcx> {
    item: Item<'a, 'tcx>,
    temps: &'a IndexVec<Local, TempState>,

    /// Explicit promotion happens e.g. for constant arguments declared via
    /// `rustc_args_required_const`.
    /// Implicit promotion has almost the same rules, except that disallows `const fn`
    /// except for those marked `#[rustc_promotable]`. This is to avoid changing
    /// a legitimate run-time operation into a failing compile-time operation
    /// e.g. due to addresses being compared inside the function.
    explicit: bool,
}

impl std::ops::Deref for Validator<'a, 'tcx> {
    type Target = Item<'a, 'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

struct Unpromotable;

impl<'tcx> Validator<'_, 'tcx> {
    fn validate_candidate(&self, candidate: Candidate) -> Result<(), Unpromotable> {
        match candidate {
            Candidate::Ref(loc) => {
                assert!(!self.explicit);

                let statement = &self.body[loc.block].statements[loc.statement_index];
                match &statement.kind {
                    StatementKind::Assign(box(_, Rvalue::Ref(_, kind, place))) => {
                        match kind {
                            BorrowKind::Shared | BorrowKind::Mut { .. } => {}

                            // FIXME(eddyb) these aren't promoted here but *could*
                            // be promoted as part of a larger value because
                            // `validate_rvalue`  doesn't check them, need to
                            // figure out what is the intended behavior.
                            BorrowKind::Shallow | BorrowKind::Unique => return Err(Unpromotable),
                        }

                        // We can only promote interior borrows of promotable temps (non-temps
                        // don't get promoted anyway).
                        let base = match place.base {
                            PlaceBase::Local(local) => local,
                            _ => return Err(Unpromotable),
                        };
                        self.validate_local(base)?;

                        if place.projection.contains(&ProjectionElem::Deref) {
                            return Err(Unpromotable);
                        }

                        let mut has_mut_interior =
                            self.qualif_local::<qualifs::HasMutInterior>(base);
                        // HACK(eddyb) this should compute the same thing as
                        // `<HasMutInterior as Qualif>::in_projection` from
                        // `check_consts::qualifs` but without recursion.
                        if has_mut_interior {
                            // This allows borrowing fields which don't have
                            // `HasMutInterior`, from a type that does, e.g.:
                            // `let _: &'static _ = &(Cell::new(1), 2).1;`
                            let mut place_projection = &place.projection[..];
                            // FIXME(eddyb) use a forward loop instead of a reverse one.
                            while let [proj_base @ .., elem] = place_projection {
                                // FIXME(eddyb) this is probably excessive, with
                                // the exception of `union` member accesses.
                                let ty =
                                    Place::ty_from(&place.base, proj_base, self.body, self.tcx)
                                        .projection_ty(self.tcx, elem)
                                        .ty;
                                if ty.is_freeze(self.tcx, self.param_env, DUMMY_SP) {
                                    has_mut_interior = false;
                                    break;
                                }

                                place_projection = proj_base;
                            }
                        }

                        // FIXME(eddyb) this duplicates part of `validate_rvalue`.
                        if has_mut_interior {
                            return Err(Unpromotable);
                        }
                        if self.qualif_local::<qualifs::NeedsDrop>(base) {
                            return Err(Unpromotable);
                        }

                        if let BorrowKind::Mut { .. } = kind {
                            let ty = place.ty(self.body, self.tcx).ty;

                            // In theory, any zero-sized value could be borrowed
                            // mutably without consequences. However, only &mut []
                            // is allowed right now, and only in functions.
                            if self.const_kind == Some(ConstKind::StaticMut) {
                                // Inside a `static mut`, &mut [...] is also allowed.
                                match ty.kind {
                                    ty::Array(..) | ty::Slice(_) => {}
                                    _ => return Err(Unpromotable),
                                }
                            } else if let ty::Array(_, len) = ty.kind {
                                // FIXME(eddyb) the `self.is_non_const_fn` condition
                                // seems unnecessary, given that this is merely a ZST.
                                match len.try_eval_usize(self.tcx, self.param_env) {
                                    Some(0) if self.const_kind.is_none() => {},
                                    _ => return Err(Unpromotable),
                                }
                            } else {
                                return Err(Unpromotable);
                            }
                        }

                        Ok(())
                    }
                    _ => bug!()
                }
            }
            Candidate::Repeat(loc) => {
                assert!(!self.explicit);

                let statement = &self.body[loc.block].statements[loc.statement_index];
                match &statement.kind {
                    StatementKind::Assign(box(_, Rvalue::Repeat(ref operand, _))) => {
                        if !self.tcx.features().const_in_array_repeat_expressions {
                            return Err(Unpromotable);
                        }

                        self.validate_operand(operand)
                    }
                    _ => bug!()
                }
            },
            Candidate::Argument { bb, index } => {
                assert!(self.explicit);

                let terminator = self.body[bb].terminator();
                match &terminator.kind {
                    TerminatorKind::Call { args, .. } => {
                        self.validate_operand(&args[index])
                    }
                    _ => bug!()
                }
            }
        }
    }

    // FIXME(eddyb) maybe cache this?
    fn qualif_local<Q: qualifs::Qualif>(&self, local: Local) -> bool {
        let per_local = &|l| self.qualif_local::<Q>(l);

        if let TempState::Defined { location: loc, .. } = self.temps[local] {
            let num_stmts = self.body[loc.block].statements.len();

            if loc.statement_index < num_stmts {
                let statement = &self.body[loc.block].statements[loc.statement_index];
                match &statement.kind {
                    StatementKind::Assign(box(_, rhs)) => {
                        Q::in_rvalue(&self.item, per_local, rhs)
                    }
                    _ => {
                        span_bug!(statement.source_info.span, "{:?} is not an assignment",
                                statement);
                    }
                }
            } else {
                let terminator = self.body[loc.block].terminator();
                match &terminator.kind {
                    TerminatorKind::Call { func, args, .. } => {
                        let return_ty = self.body.local_decls[local].ty;
                        Q::in_call(&self.item, per_local, func, args, return_ty)
                    }
                    kind => {
                        span_bug!(terminator.source_info.span, "{:?} not promotable", kind);
                    }
                }
            }
        } else {
            let span = self.body.local_decls[local].source_info.span;
            span_bug!(span, "{:?} not promotable, qualif_local shouldn't have been called", local);
        }
    }

    // FIXME(eddyb) maybe cache this?
    fn validate_local(&self, local: Local) -> Result<(), Unpromotable> {
        if let TempState::Defined { location: loc, .. } = self.temps[local] {
            let num_stmts = self.body[loc.block].statements.len();

            if loc.statement_index < num_stmts {
                let statement = &self.body[loc.block].statements[loc.statement_index];
                match &statement.kind {
                    StatementKind::Assign(box(_, rhs)) => self.validate_rvalue(rhs),
                    _ => {
                        span_bug!(statement.source_info.span, "{:?} is not an assignment",
                                statement);
                    }
                }
            } else {
                let terminator = self.body[loc.block].terminator();
                match &terminator.kind {
                    TerminatorKind::Call { func, args, .. } => self.validate_call(func, args),
                    kind => {
                        span_bug!(terminator.source_info.span, "{:?} not promotable", kind);
                    }
                }
            }
        } else {
            Err(Unpromotable)
        }
    }

    fn validate_place(&self, place: PlaceRef<'_, 'tcx>) -> Result<(), Unpromotable> {
        match place {
            PlaceRef {
                base: PlaceBase::Local(local),
                projection: [],
            } => self.validate_local(*local),
            PlaceRef {
                base: PlaceBase::Static(box Static {
                    kind: StaticKind::Promoted { .. },
                    ..
                }),
                projection: [],
            } => bug!("qualifying already promoted MIR"),
            PlaceRef {
                base: PlaceBase::Static(box Static {
                    kind: StaticKind::Static,
                    def_id,
                    ..
                }),
                projection: [],
            } => {
                // Only allow statics (not consts) to refer to other statics.
                // FIXME(eddyb) does this matter at all for promotion?
                let is_static = self.const_kind.map_or(false, |k| k.is_static());
                if !is_static {
                    return Err(Unpromotable);
                }

                let is_thread_local = self.tcx.has_attr(*def_id, sym::thread_local);
                if is_thread_local {
                    return Err(Unpromotable);
                }

                Ok(())
            }
            PlaceRef {
                base: _,
                projection: [proj_base @ .., elem],
            } => {
                match *elem {
                    ProjectionElem::Deref |
                    ProjectionElem::Downcast(..) => return Err(Unpromotable),

                    ProjectionElem::ConstantIndex {..} |
                    ProjectionElem::Subslice {..} => {}

                    ProjectionElem::Index(local) => {
                        self.validate_local(local)?;
                    }

                    ProjectionElem::Field(..) => {
                        if self.const_kind.is_none() {
                            let base_ty =
                                Place::ty_from(place.base, proj_base, self.body, self.tcx).ty;
                            if let Some(def) = base_ty.ty_adt_def() {
                                // No promotion of union field accesses.
                                if def.is_union() {
                                    return Err(Unpromotable);
                                }
                            }
                        }
                    }
                }

                self.validate_place(PlaceRef {
                    base: place.base,
                    projection: proj_base,
                })
            }
        }
    }

    fn validate_operand(&self, operand: &Operand<'tcx>) -> Result<(), Unpromotable> {
        match operand {
            Operand::Copy(place) |
            Operand::Move(place) => self.validate_place(place.as_ref()),

            Operand::Constant(constant) => {
                if let ConstValue::Unevaluated(def_id, _) = constant.literal.val {
                    if self.tcx.trait_of_item(def_id).is_some() {
                        // Don't peek inside trait associated constants.
                        // (see below what we do for other consts, for now)
                    } else {
                        // HACK(eddyb) ensure that errors propagate correctly.
                        // FIXME(eddyb) remove this once the old promotion logic
                        // is gone - we can always promote constants even if they
                        // fail to pass const-checking, as compilation would've
                        // errored independently and promotion can't change that.
                        let (bits, _) = self.tcx.at(constant.span).mir_const_qualif(def_id);
                        if bits == super::qualify_consts::QUALIF_ERROR_BIT {
                            self.tcx.sess.delay_span_bug(
                                constant.span,
                                "promote_consts: MIR had errors",
                            );
                            return Err(Unpromotable);
                        }
                    }
                }

                Ok(())
            }
        }
    }

    fn validate_rvalue(&self, rvalue: &Rvalue<'tcx>) -> Result<(), Unpromotable> {
        match *rvalue {
            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) if self.const_kind.is_none() => {
                let operand_ty = operand.ty(self.body, self.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                match (cast_in, cast_out) {
                    (CastTy::Ptr(_), CastTy::Int(_)) |
                    (CastTy::FnPtr, CastTy::Int(_)) => {
                        // in normal functions, mark such casts as not promotable
                        return Err(Unpromotable);
                    }
                    _ => {}
                }
            }

            Rvalue::BinaryOp(op, ref lhs, _) if self.const_kind.is_none() => {
                if let ty::RawPtr(_) | ty::FnPtr(..) = lhs.ty(self.body, self.tcx).kind {
                    assert!(op == BinOp::Eq || op == BinOp::Ne ||
                            op == BinOp::Le || op == BinOp::Lt ||
                            op == BinOp::Ge || op == BinOp::Gt ||
                            op == BinOp::Offset);

                    // raw pointer operations are not allowed inside promoteds
                    return Err(Unpromotable);
                }
            }

            Rvalue::NullaryOp(NullOp::Box, _) => return Err(Unpromotable),

            _ => {}
        }

        match rvalue {
            Rvalue::NullaryOp(..) => Ok(()),

            Rvalue::Discriminant(place) |
            Rvalue::Len(place) => self.validate_place(place.as_ref()),

            Rvalue::Use(operand) |
            Rvalue::Repeat(operand, _) |
            Rvalue::UnaryOp(_, operand) |
            Rvalue::Cast(_, operand, _) => self.validate_operand(operand),

            Rvalue::BinaryOp(_, lhs, rhs) |
            Rvalue::CheckedBinaryOp(_, lhs, rhs) => {
                self.validate_operand(lhs)?;
                self.validate_operand(rhs)
            }

            Rvalue::Ref(_, kind, place) => {
                if let BorrowKind::Mut { .. } = kind {
                    let ty = place.ty(self.body, self.tcx).ty;

                    // In theory, any zero-sized value could be borrowed
                    // mutably without consequences. However, only &mut []
                    // is allowed right now, and only in functions.
                    if self.const_kind == Some(ConstKind::StaticMut) {
                        // Inside a `static mut`, &mut [...] is also allowed.
                        match ty.kind {
                            ty::Array(..) | ty::Slice(_) => {}
                            _ => return Err(Unpromotable),
                        }
                    } else if let ty::Array(_, len) = ty.kind {
                        // FIXME(eddyb): We only return `Unpromotable` for `&mut []` inside a
                        // const context which seems unnecessary given that this is merely a ZST.
                        match len.try_eval_usize(self.tcx, self.param_env) {
                            Some(0) if self.const_kind.is_none() => {},
                            _ => return Err(Unpromotable),
                        }
                    } else {
                        return Err(Unpromotable);
                    }
                }

                // Special-case reborrows to be more like a copy of the reference.
                let mut place = place.as_ref();
                if let [proj_base @ .., ProjectionElem::Deref] = &place.projection {
                    let base_ty =
                        Place::ty_from(&place.base, proj_base, self.body, self.tcx).ty;
                    if let ty::Ref(..) = base_ty.kind {
                        place = PlaceRef {
                            base: &place.base,
                            projection: proj_base,
                        };
                    }
                }

                self.validate_place(place)?;

                // HACK(eddyb) this should compute the same thing as
                // `<HasMutInterior as Qualif>::in_projection` from
                // `check_consts::qualifs` but without recursion.
                let mut has_mut_interior = match place.base {
                    PlaceBase::Local(local) => {
                        self.qualif_local::<qualifs::HasMutInterior>(*local)
                    }
                    PlaceBase::Static(_) => false,
                };
                if has_mut_interior {
                    let mut place_projection = place.projection;
                    // FIXME(eddyb) use a forward loop instead of a reverse one.
                    while let [proj_base @ .., elem] = place_projection {
                        // FIXME(eddyb) this is probably excessive, with
                        // the exception of `union` member accesses.
                        let ty = Place::ty_from(place.base, proj_base, self.body, self.tcx)
                            .projection_ty(self.tcx, elem)
                            .ty;
                        if ty.is_freeze(self.tcx, self.param_env, DUMMY_SP) {
                            has_mut_interior = false;
                            break;
                        }

                        place_projection = proj_base;
                    }
                }
                if has_mut_interior {
                    return Err(Unpromotable);
                }

                Ok(())
            }

            Rvalue::Aggregate(_, ref operands) => {
                for o in operands {
                    self.validate_operand(o)?;
                }

                Ok(())
            }
        }
    }

    fn validate_call(
        &self,
        callee: &Operand<'tcx>,
        args: &[Operand<'tcx>],
    ) -> Result<(), Unpromotable> {
        let fn_ty = callee.ty(self.body, self.tcx);

        if !self.explicit && self.const_kind.is_none() {
            if let ty::FnDef(def_id, _) = fn_ty.kind {
                // Never promote runtime `const fn` calls of
                // functions without `#[rustc_promotable]`.
                if !self.tcx.is_promotable_const_fn(def_id) {
                    return Err(Unpromotable);
                }
            }
        }

        let is_const_fn = match fn_ty.kind {
            ty::FnDef(def_id, _) => {
                self.tcx.is_const_fn(def_id) ||
                self.tcx.is_unstable_const_fn(def_id).is_some() ||
                is_lang_panic_fn(self.tcx, self.def_id)
            }
            _ => false,
        };
        if !is_const_fn {
            return Err(Unpromotable);
        }

        self.validate_operand(callee)?;
        for arg in args {
            self.validate_operand(arg)?;
        }

        Ok(())
    }
}

// FIXME(eddyb) remove the differences for promotability in `static`, `const`, `const fn`.
pub fn validate_candidates(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    def_id: DefId,
    temps: &IndexVec<Local, TempState>,
    candidates: &[Candidate],
) -> Vec<Candidate> {
    let mut validator = Validator {
        item: Item::new(tcx, def_id, body),
        temps,
        explicit: false,
    };

    candidates.iter().copied().filter(|&candidate| {
        validator.explicit = match candidate {
            Candidate::Ref(_) |
            Candidate::Repeat(_) => false,
            Candidate::Argument { .. } => true,
        };

        // FIXME(eddyb) also emit the errors for shuffle indices
        // and `#[rustc_args_required_const]` arguments here.

        validator.validate_candidate(candidate).is_ok()
    }).collect()
}

struct Promoter<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    source: &'a mut Body<'tcx>,
    promoted: Body<'tcx>,
    temps: &'a mut IndexVec<Local, TempState>,

    /// If true, all nested temps are also kept in the
    /// source MIR, not moved to the promoted MIR.
    keep_original: bool,
}

impl<'a, 'tcx> Promoter<'a, 'tcx> {
    fn new_block(&mut self) -> BasicBlock {
        let span = self.promoted.span;
        self.promoted.basic_blocks_mut().push(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info: SourceInfo {
                    span,
                    scope: OUTERMOST_SOURCE_SCOPE
                },
                kind: TerminatorKind::Return
            }),
            is_cleanup: false
        })
    }

    fn assign(&mut self, dest: Local, rvalue: Rvalue<'tcx>, span: Span) {
        let last = self.promoted.basic_blocks().last().unwrap();
        let data = &mut self.promoted[last];
        data.statements.push(Statement {
            source_info: SourceInfo {
                span,
                scope: OUTERMOST_SOURCE_SCOPE
            },
            kind: StatementKind::Assign(box(Place::from(dest), rvalue))
        });
    }

    fn is_temp_kind(&self, local: Local) -> bool {
        self.source.local_kind(local) == LocalKind::Temp
    }

    /// Copies the initialization of this temp to the
    /// promoted MIR, recursing through temps.
    fn promote_temp(&mut self, temp: Local) -> Local {
        let old_keep_original = self.keep_original;
        let loc = match self.temps[temp] {
            TempState::Defined { location, uses } if uses > 0 => {
                if uses > 1 {
                    self.keep_original = true;
                }
                location
            }
            state =>  {
                span_bug!(self.promoted.span, "{:?} not promotable: {:?}",
                          temp, state);
            }
        };
        if !self.keep_original {
            self.temps[temp] = TempState::PromotedOut;
        }

        let num_stmts = self.source[loc.block].statements.len();
        let new_temp = self.promoted.local_decls.push(
            LocalDecl::new_temp(self.source.local_decls[temp].ty,
                                self.source.local_decls[temp].source_info.span));

        debug!("promote({:?} @ {:?}/{:?}, {:?})",
               temp, loc, num_stmts, self.keep_original);

        // First, take the Rvalue or Call out of the source MIR,
        // or duplicate it, depending on keep_original.
        if loc.statement_index < num_stmts {
            let (mut rvalue, source_info) = {
                let statement = &mut self.source[loc.block].statements[loc.statement_index];
                let rhs = match statement.kind {
                    StatementKind::Assign(box(_, ref mut rhs)) => rhs,
                    _ => {
                        span_bug!(statement.source_info.span, "{:?} is not an assignment",
                                  statement);
                    }
                };

                (if self.keep_original {
                    rhs.clone()
                } else {
                    let unit = Rvalue::Aggregate(box AggregateKind::Tuple, vec![]);
                    mem::replace(rhs, unit)
                }, statement.source_info)
            };

            self.visit_rvalue(&mut rvalue, loc);
            self.assign(new_temp, rvalue, source_info.span);
        } else {
            let terminator = if self.keep_original {
                self.source[loc.block].terminator().clone()
            } else {
                let terminator = self.source[loc.block].terminator_mut();
                let target = match terminator.kind {
                    TerminatorKind::Call { destination: Some((_, target)), .. } => target,
                    ref kind => {
                        span_bug!(terminator.source_info.span, "{:?} not promotable", kind);
                    }
                };
                Terminator {
                    source_info: terminator.source_info,
                    kind: mem::replace(&mut terminator.kind, TerminatorKind::Goto {
                        target,
                    })
                }
            };

            match terminator.kind {
                TerminatorKind::Call { mut func, mut args, from_hir_call, .. } => {
                    self.visit_operand(&mut func, loc);
                    for arg in &mut args {
                        self.visit_operand(arg, loc);
                    }

                    let last = self.promoted.basic_blocks().last().unwrap();
                    let new_target = self.new_block();

                    *self.promoted[last].terminator_mut() = Terminator {
                        kind: TerminatorKind::Call {
                            func,
                            args,
                            cleanup: None,
                            destination: Some(
                                (Place::from(new_temp), new_target)
                            ),
                            from_hir_call,
                        },
                        ..terminator
                    };
                }
                ref kind => {
                    span_bug!(terminator.source_info.span, "{:?} not promotable", kind);
                }
            };
        };

        self.keep_original = old_keep_original;
        new_temp
    }

    fn promote_candidate(
        mut self,
        def_id: DefId,
        candidate: Candidate,
        next_promoted_id: usize,
    ) -> Option<Body<'tcx>> {
        let mut operand = {
            let promoted = &mut self.promoted;
            let promoted_id = Promoted::new(next_promoted_id);
            let tcx = self.tcx;
            let mut promoted_place = |ty, span| {
                promoted.span = span;
                promoted.local_decls[RETURN_PLACE] = LocalDecl::new_return_place(ty, span);
                Place {
                    base: PlaceBase::Static(box Static {
                        kind:
                            StaticKind::Promoted(
                                promoted_id,
                                InternalSubsts::identity_for_item(tcx, def_id),
                            ),
                        ty,
                        def_id,
                    }),
                    projection: List::empty(),
                }
            };
            let (blocks, local_decls) = self.source.basic_blocks_and_local_decls_mut();
            match candidate {
                Candidate::Ref(loc) => {
                    let ref mut statement = blocks[loc.block].statements[loc.statement_index];
                    match statement.kind {
                        StatementKind::Assign(box(_, Rvalue::Ref(_, _, ref mut place))) => {
                            // Use the underlying local for this (necessarily interior) borrow.
                            let ty = place.base.ty(local_decls).ty;
                            let span = statement.source_info.span;

                            Operand::Move(Place {
                                base: mem::replace(
                                    &mut place.base,
                                    promoted_place(ty, span).base,
                                ),
                                projection: List::empty(),
                            })
                        }
                        _ => bug!()
                    }
                }
                Candidate::Repeat(loc) => {
                    let ref mut statement = blocks[loc.block].statements[loc.statement_index];
                    match statement.kind {
                        StatementKind::Assign(box(_, Rvalue::Repeat(ref mut operand, _))) => {
                            let ty = operand.ty(local_decls, self.tcx);
                            let span = statement.source_info.span;
                            mem::replace(
                                operand,
                                Operand::Copy(promoted_place(ty, span))
                            )
                        }
                        _ => bug!()
                    }
                },
                Candidate::Argument { bb, index } => {
                    let terminator = blocks[bb].terminator_mut();
                    match terminator.kind {
                        TerminatorKind::Call { ref mut args, .. } => {
                            let ty = args[index].ty(local_decls, self.tcx);
                            let span = terminator.source_info.span;
                            let operand = Operand::Copy(promoted_place(ty, span));
                            mem::replace(&mut args[index], operand)
                        }
                        // We expected a `TerminatorKind::Call` for which we'd like to promote an
                        // argument. `qualify_consts` saw a `TerminatorKind::Call` here, but
                        // we are seeing a `Goto`. That means that the `promote_temps` method
                        // already promoted this call away entirely. This case occurs when calling
                        // a function requiring a constant argument and as that constant value
                        // providing a value whose computation contains another call to a function
                        // requiring a constant argument.
                        TerminatorKind::Goto { .. } => return None,
                        _ => bug!()
                    }
                }
            }
        };

        assert_eq!(self.new_block(), START_BLOCK);
        self.visit_operand(&mut operand, Location {
            block: BasicBlock::new(0),
            statement_index: usize::MAX
        });

        let span = self.promoted.span;
        self.assign(RETURN_PLACE, Rvalue::Use(operand), span);
        Some(self.promoted)
    }
}

/// Replaces all temporaries with their promoted counterparts.
impl<'a, 'tcx> MutVisitor<'tcx> for Promoter<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self,
                   local: &mut Local,
                   _: PlaceContext,
                   _: Location) {
        if self.is_temp_kind(*local) {
            *local = self.promote_temp(*local);
        }
    }

    fn process_projection_elem(
        &mut self,
        elem: &PlaceElem<'tcx>,
    ) -> Option<PlaceElem<'tcx>> {
        match elem {
            PlaceElem::Index(local) if self.is_temp_kind(*local) => {
                Some(PlaceElem::Index(self.promote_temp(*local)))
            }
            _ => None,
        }
    }
}

pub fn promote_candidates<'tcx>(
    def_id: DefId,
    body: &mut Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    mut temps: IndexVec<Local, TempState>,
    candidates: Vec<Candidate>,
) -> IndexVec<Promoted, Body<'tcx>> {
    // Visit candidates in reverse, in case they're nested.
    debug!("promote_candidates({:?})", candidates);

    let mut promotions = IndexVec::new();

    for candidate in candidates.into_iter().rev() {
        match candidate {
            Candidate::Repeat(Location { block, statement_index }) |
            Candidate::Ref(Location { block, statement_index }) => {
                match &body[block].statements[statement_index].kind {
                    StatementKind::Assign(box(place, _)) => {
                        if let Some(local) = place.as_local() {
                            if temps[local] == TempState::PromotedOut {
                                // Already promoted.
                                continue;
                            }
                        }
                    }
                    _ => {}
                }
            }
            Candidate::Argument { .. } => {}
        }


        // Declare return place local so that `mir::Body::new` doesn't complain.
        let initial_locals = iter::once(
            LocalDecl::new_return_place(tcx.types.never, body.span)
        ).collect();

        let promoter = Promoter {
            promoted: Body::new(
                IndexVec::new(),
                // FIXME: maybe try to filter this to avoid blowing up
                // memory usage?
                body.source_scopes.clone(),
                body.source_scope_local_data.clone(),
                None,
                initial_locals,
                IndexVec::new(),
                0,
                vec![],
                body.span,
                vec![],
            ),
            tcx,
            source: body,
            temps: &mut temps,
            keep_original: false
        };

        //FIXME(oli-obk): having a `maybe_push()` method on `IndexVec` might be nice
        if let Some(promoted) = promoter.promote_candidate(def_id, candidate, promotions.len()) {
            promotions.push(promoted);
        }
    }

    // Eliminate assignments to, and drops of promoted temps.
    let promoted = |index: Local| temps[index] == TempState::PromotedOut;
    for block in body.basic_blocks_mut() {
        block.statements.retain(|statement| {
            match &statement.kind {
                StatementKind::Assign(box(place, _)) => {
                    if let Some(index) = place.as_local() {
                        !promoted(index)
                    } else {
                        true
                    }
                }
                StatementKind::StorageLive(index) |
                StatementKind::StorageDead(index) => {
                    !promoted(*index)
                }
                _ => true
            }
        });
        let terminator = block.terminator_mut();
        match &terminator.kind {
            TerminatorKind::Drop { location: place, target, .. } => {
                if let Some(index) = place.as_local() {
                    if promoted(index) {
                        terminator.kind = TerminatorKind::Goto {
                            target: *target,
                        };
                    }
                }
            }
            _ => {}
        }
    }

    promotions
}
