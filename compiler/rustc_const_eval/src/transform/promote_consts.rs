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

use rustc_hir as hir;
use rustc_middle::mir::traversal::ReversePostorder;
use rustc_middle::mir::visit::{MutVisitor, MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::cast::CastTy;
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::{self, List, TyCtxt, TypeFoldable};
use rustc_span::Span;

use rustc_index::vec::{Idx, IndexVec};

use std::cell::Cell;
use std::{cmp, iter, mem};

use crate::const_eval::{is_const_fn, is_unstable_const_fn};
use crate::transform::check_consts::{is_lang_panic_fn, qualifs, ConstCx};
use crate::transform::MirPass;

/// A `MirPass` for promotion.
///
/// Promotion is the extraction of promotable temps into separate MIR bodies so they can have
/// `'static` lifetime.
///
/// After this pass is run, `promoted_fragments` will hold the MIR body corresponding to each
/// newly created `Constant`.
#[derive(Default)]
pub struct PromoteTemps<'tcx> {
    pub promoted_fragments: Cell<IndexVec<Promoted, Body<'tcx>>>,
}

impl<'tcx> MirPass<'tcx> for PromoteTemps<'tcx> {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // There's not really any point in promoting errorful MIR.
        //
        // This does not include MIR that failed const-checking, which we still try to promote.
        if body.return_ty().references_error() {
            tcx.sess.delay_span_bug(body.span, "PromoteTemps: MIR had errors");
            return;
        }

        if body.source.promoted.is_some() {
            return;
        }

        let mut rpo = traversal::reverse_postorder(body);
        let ccx = ConstCx::new(tcx, body);
        let (temps, all_candidates) = collect_temps_and_candidates(&ccx, &mut rpo);

        let promotable_candidates = validate_candidates(&ccx, &temps, &all_candidates);

        let promoted = promote_candidates(body, tcx, temps, promotable_candidates);
        self.promoted_fragments.set(promoted);
    }
}

/// State of a temporary during collection and promotion.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TempState {
    /// No references to this temp.
    Undefined,
    /// One direct assignment and any number of direct uses.
    /// A borrow of this temp is promotable if the assigned
    /// value is qualified as constant.
    Defined { location: Location, uses: usize },
    /// Any other combination of assignments/uses.
    Unpromotable,
    /// This temp was part of an rvalue which got extracted
    /// during promotion and needs cleanup.
    PromotedOut,
}

impl TempState {
    pub fn is_promotable(&self) -> bool {
        debug!("is_promotable: self={:?}", self);
        matches!(self, TempState::Defined { .. })
    }
}

/// A "root candidate" for promotion, which will become the
/// returned value in a promoted MIR, unless it's a subset
/// of a larger candidate.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Candidate {
    /// Borrow of a constant temporary, candidate for lifetime extension.
    Ref(Location),
}

impl Candidate {
    fn source_info(&self, body: &Body<'_>) -> SourceInfo {
        match self {
            Candidate::Ref(location) => *body.source_info(*location),
        }
    }
}

struct Collector<'a, 'tcx> {
    ccx: &'a ConstCx<'a, 'tcx>,
    temps: IndexVec<Local, TempState>,
    candidates: Vec<Candidate>,
}

impl<'tcx> Visitor<'tcx> for Collector<'_, 'tcx> {
    fn visit_local(&mut self, &index: &Local, context: PlaceContext, location: Location) {
        debug!("visit_local: index={:?} context={:?} location={:?}", index, context, location);
        // We're only interested in temporaries and the return place
        match self.ccx.body.local_kind(index) {
            LocalKind::Temp | LocalKind::ReturnPointer => {}
            LocalKind::Arg | LocalKind::Var => return,
        }

        // Ignore drops, if the temp gets promoted,
        // then it's constant and thus drop is noop.
        // Non-uses are also irrelevant.
        if context.is_drop() || !context.is_use() {
            debug!(
                "visit_local: context.is_drop={:?} context.is_use={:?}",
                context.is_drop(),
                context.is_use(),
            );
            return;
        }

        let temp = &mut self.temps[index];
        debug!("visit_local: temp={:?}", temp);
        if *temp == TempState::Undefined {
            match context {
                PlaceContext::MutatingUse(MutatingUseContext::Store)
                | PlaceContext::MutatingUse(MutatingUseContext::Call) => {
                    *temp = TempState::Defined { location, uses: 0 };
                    return;
                }
                _ => { /* mark as unpromotable below */ }
            }
        } else if let TempState::Defined { ref mut uses, .. } = *temp {
            // We always allow borrows, even mutable ones, as we need
            // to promote mutable borrows of some ZSTs e.g., `&mut []`.
            let allowed_use = match context {
                PlaceContext::MutatingUse(MutatingUseContext::Borrow)
                | PlaceContext::NonMutatingUse(_) => true,
                PlaceContext::MutatingUse(_) | PlaceContext::NonUse(_) => false,
            };
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
            _ => {}
        }
    }
}

pub fn collect_temps_and_candidates(
    ccx: &ConstCx<'mir, 'tcx>,
    rpo: &mut ReversePostorder<'_, 'tcx>,
) -> (IndexVec<Local, TempState>, Vec<Candidate>) {
    let mut collector = Collector {
        temps: IndexVec::from_elem(TempState::Undefined, &ccx.body.local_decls),
        candidates: vec![],
        ccx,
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
    ccx: &'a ConstCx<'a, 'tcx>,
    temps: &'a IndexVec<Local, TempState>,
}

impl std::ops::Deref for Validator<'a, 'tcx> {
    type Target = ConstCx<'a, 'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.ccx
    }
}

struct Unpromotable;

impl<'tcx> Validator<'_, 'tcx> {
    fn validate_candidate(&self, candidate: Candidate) -> Result<(), Unpromotable> {
        match candidate {
            Candidate::Ref(loc) => {
                let statement = &self.body[loc.block].statements[loc.statement_index];
                match &statement.kind {
                    StatementKind::Assign(box (_, Rvalue::Ref(_, kind, place))) => {
                        // We can only promote interior borrows of promotable temps (non-temps
                        // don't get promoted anyway).
                        self.validate_local(place.local)?;

                        // The reference operation itself must be promotable.
                        // (Needs to come after `validate_local` to avoid ICEs.)
                        self.validate_ref(*kind, place)?;

                        // We do not check all the projections (they do not get promoted anyway),
                        // but we do stay away from promoting anything involving a dereference.
                        if place.projection.contains(&ProjectionElem::Deref) {
                            return Err(Unpromotable);
                        }

                        // We cannot promote things that need dropping, since the promoted value
                        // would not get dropped.
                        if self.qualif_local::<qualifs::NeedsNonConstDrop>(place.local) {
                            return Err(Unpromotable);
                        }

                        Ok(())
                    }
                    _ => bug!(),
                }
            }
        }
    }

    // FIXME(eddyb) maybe cache this?
    fn qualif_local<Q: qualifs::Qualif>(&self, local: Local) -> bool {
        if let TempState::Defined { location: loc, .. } = self.temps[local] {
            let num_stmts = self.body[loc.block].statements.len();

            if loc.statement_index < num_stmts {
                let statement = &self.body[loc.block].statements[loc.statement_index];
                match &statement.kind {
                    StatementKind::Assign(box (_, rhs)) => qualifs::in_rvalue::<Q, _>(
                        &self.ccx,
                        &mut |l| self.qualif_local::<Q>(l),
                        rhs,
                    ),
                    _ => {
                        span_bug!(
                            statement.source_info.span,
                            "{:?} is not an assignment",
                            statement
                        );
                    }
                }
            } else {
                let terminator = self.body[loc.block].terminator();
                match &terminator.kind {
                    TerminatorKind::Call { .. } => {
                        let return_ty = self.body.local_decls[local].ty;
                        Q::in_any_value_of_ty(&self.ccx, return_ty)
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
            let block = &self.body[loc.block];
            let num_stmts = block.statements.len();

            if loc.statement_index < num_stmts {
                let statement = &block.statements[loc.statement_index];
                match &statement.kind {
                    StatementKind::Assign(box (_, rhs)) => self.validate_rvalue(rhs),
                    _ => {
                        span_bug!(
                            statement.source_info.span,
                            "{:?} is not an assignment",
                            statement
                        );
                    }
                }
            } else {
                let terminator = block.terminator();
                match &terminator.kind {
                    TerminatorKind::Call { func, args, .. } => self.validate_call(func, args),
                    TerminatorKind::Yield { .. } => Err(Unpromotable),
                    kind => {
                        span_bug!(terminator.source_info.span, "{:?} not promotable", kind);
                    }
                }
            }
        } else {
            Err(Unpromotable)
        }
    }

    fn validate_place(&self, place: PlaceRef<'tcx>) -> Result<(), Unpromotable> {
        match place.last_projection() {
            None => self.validate_local(place.local),
            Some((place_base, elem)) => {
                // Validate topmost projection, then recurse.
                match elem {
                    ProjectionElem::Deref => {
                        let mut promotable = false;
                        // We need to make sure this is a `Deref` of a local with no further projections.
                        // Discussion can be found at
                        // https://github.com/rust-lang/rust/pull/74945#discussion_r463063247
                        if let Some(local) = place_base.as_local() {
                            // This is a special treatment for cases like *&STATIC where STATIC is a
                            // global static variable.
                            // This pattern is generated only when global static variables are directly
                            // accessed and is qualified for promotion safely.
                            if let TempState::Defined { location, .. } = self.temps[local] {
                                let def_stmt = self.body[location.block]
                                    .statements
                                    .get(location.statement_index);
                                if let Some(Statement {
                                    kind:
                                        StatementKind::Assign(box (
                                            _,
                                            Rvalue::Use(Operand::Constant(c)),
                                        )),
                                    ..
                                }) = def_stmt
                                {
                                    if let Some(did) = c.check_static_ptr(self.tcx) {
                                        // Evaluating a promoted may not read statics except if it got
                                        // promoted from a static (this is a CTFE check). So we
                                        // can only promote static accesses inside statics.
                                        if let Some(hir::ConstContext::Static(..)) = self.const_kind
                                        {
                                            if !self.tcx.is_thread_local_static(did) {
                                                promotable = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if !promotable {
                            return Err(Unpromotable);
                        }
                    }
                    ProjectionElem::Downcast(..) => {
                        return Err(Unpromotable);
                    }

                    ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. } => {}

                    ProjectionElem::Index(local) => {
                        let mut promotable = false;
                        // Only accept if we can predict the index and are indexing an array.
                        let val =
                            if let TempState::Defined { location: loc, .. } = self.temps[local] {
                                let block = &self.body[loc.block];
                                if loc.statement_index < block.statements.len() {
                                    let statement = &block.statements[loc.statement_index];
                                    match &statement.kind {
                                        StatementKind::Assign(box (
                                            _,
                                            Rvalue::Use(Operand::Constant(c)),
                                        )) => c.literal.try_eval_usize(self.tcx, self.param_env),
                                        _ => None,
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            };
                        if let Some(idx) = val {
                            // Determine the type of the thing we are indexing.
                            let ty = place_base.ty(self.body, self.tcx).ty;
                            match ty.kind() {
                                ty::Array(_, len) => {
                                    // It's an array; determine its length.
                                    if let Some(len) = len.try_eval_usize(self.tcx, self.param_env)
                                    {
                                        // If the index is in-bounds, go ahead.
                                        if idx < len {
                                            promotable = true;
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        if !promotable {
                            return Err(Unpromotable);
                        }

                        self.validate_local(local)?;
                    }

                    ProjectionElem::Field(..) => {
                        let base_ty = place_base.ty(self.body, self.tcx).ty;
                        if base_ty.is_union() {
                            // No promotion of union field accesses.
                            return Err(Unpromotable);
                        }
                    }
                }

                self.validate_place(place_base)
            }
        }
    }

    fn validate_operand(&self, operand: &Operand<'tcx>) -> Result<(), Unpromotable> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.validate_place(place.as_ref()),

            // The qualifs for a constant (e.g. `HasMutInterior`) are checked in
            // `validate_rvalue` upon access.
            Operand::Constant(c) => {
                if let Some(def_id) = c.check_static_ptr(self.tcx) {
                    // Only allow statics (not consts) to refer to other statics.
                    // FIXME(eddyb) does this matter at all for promotion?
                    // FIXME(RalfJung) it makes little sense to not promote this in `fn`/`const fn`,
                    // and in `const` this cannot occur anyway. The only concern is that we might
                    // promote even `let x = &STATIC` which would be useless, but this applies to
                    // promotion inside statics as well.
                    let is_static = matches!(self.const_kind, Some(hir::ConstContext::Static(_)));
                    if !is_static {
                        return Err(Unpromotable);
                    }

                    let is_thread_local = self.tcx.is_thread_local_static(def_id);
                    if is_thread_local {
                        return Err(Unpromotable);
                    }
                }

                Ok(())
            }
        }
    }

    fn validate_ref(&self, kind: BorrowKind, place: &Place<'tcx>) -> Result<(), Unpromotable> {
        match kind {
            // Reject these borrow types just to be safe.
            // FIXME(RalfJung): could we allow them? Should we? No point in it until we have a usecase.
            BorrowKind::Shallow | BorrowKind::Unique => return Err(Unpromotable),

            BorrowKind::Shared => {
                let has_mut_interior = self.qualif_local::<qualifs::HasMutInterior>(place.local);
                if has_mut_interior {
                    return Err(Unpromotable);
                }
            }

            BorrowKind::Mut { .. } => {
                let ty = place.ty(self.body, self.tcx).ty;

                // In theory, any zero-sized value could be borrowed
                // mutably without consequences. However, only &mut []
                // is allowed right now.
                if let ty::Array(_, len) = ty.kind() {
                    match len.try_eval_usize(self.tcx, self.param_env) {
                        Some(0) => {}
                        _ => return Err(Unpromotable),
                    }
                } else {
                    return Err(Unpromotable);
                }
            }
        }

        Ok(())
    }

    fn validate_rvalue(&self, rvalue: &Rvalue<'tcx>) -> Result<(), Unpromotable> {
        match rvalue {
            Rvalue::Use(operand) | Rvalue::Repeat(operand, _) => {
                self.validate_operand(operand)?;
            }

            Rvalue::Discriminant(place) | Rvalue::Len(place) => {
                self.validate_place(place.as_ref())?
            }

            Rvalue::ThreadLocalRef(_) => return Err(Unpromotable),

            Rvalue::Cast(kind, operand, cast_ty) => {
                if matches!(kind, CastKind::Misc) {
                    let operand_ty = operand.ty(self.body, self.tcx);
                    let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                    let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                    if let (CastTy::Ptr(_) | CastTy::FnPtr, CastTy::Int(_)) = (cast_in, cast_out) {
                        // ptr-to-int casts are not possible in consts and thus not promotable
                        return Err(Unpromotable);
                    }
                    // int-to-ptr casts are fine, they just use the integer value at pointer type.
                }

                self.validate_operand(operand)?;
            }

            Rvalue::NullaryOp(op, _) => match op {
                NullOp::Box => return Err(Unpromotable),
                NullOp::SizeOf => {}
                NullOp::AlignOf => {}
            },

            Rvalue::ShallowInitBox(_, _) => return Err(Unpromotable),

            Rvalue::UnaryOp(op, operand) => {
                match op {
                    // These operations can never fail.
                    UnOp::Neg | UnOp::Not => {}
                }

                self.validate_operand(operand)?;
            }

            Rvalue::BinaryOp(op, box (lhs, rhs)) | Rvalue::CheckedBinaryOp(op, box (lhs, rhs)) => {
                let op = *op;
                let lhs_ty = lhs.ty(self.body, self.tcx);

                if let ty::RawPtr(_) | ty::FnPtr(..) = lhs_ty.kind() {
                    // Raw and fn pointer operations are not allowed inside consts and thus not promotable.
                    assert!(matches!(
                        op,
                        BinOp::Eq
                            | BinOp::Ne
                            | BinOp::Le
                            | BinOp::Lt
                            | BinOp::Ge
                            | BinOp::Gt
                            | BinOp::Offset
                    ));
                    return Err(Unpromotable);
                }

                match op {
                    BinOp::Div | BinOp::Rem => {
                        if lhs_ty.is_integral() {
                            // Integer division: the RHS must be a non-zero const.
                            let const_val = match rhs {
                                Operand::Constant(c) => {
                                    c.literal.try_eval_bits(self.tcx, self.param_env, lhs_ty)
                                }
                                _ => None,
                            };
                            match const_val {
                                Some(x) if x != 0 => {}        // okay
                                _ => return Err(Unpromotable), // value not known or 0 -- not okay
                            }
                        }
                    }
                    // The remaining operations can never fail.
                    BinOp::Eq
                    | BinOp::Ne
                    | BinOp::Le
                    | BinOp::Lt
                    | BinOp::Ge
                    | BinOp::Gt
                    | BinOp::Offset
                    | BinOp::Add
                    | BinOp::Sub
                    | BinOp::Mul
                    | BinOp::BitXor
                    | BinOp::BitAnd
                    | BinOp::BitOr
                    | BinOp::Shl
                    | BinOp::Shr => {}
                }

                self.validate_operand(lhs)?;
                self.validate_operand(rhs)?;
            }

            Rvalue::AddressOf(_, place) => {
                // We accept `&raw *`, i.e., raw reborrows -- creating a raw pointer is
                // no problem, only using it is.
                if let Some((place_base, ProjectionElem::Deref)) = place.as_ref().last_projection()
                {
                    let base_ty = place_base.ty(self.body, self.tcx).ty;
                    if let ty::Ref(..) = base_ty.kind() {
                        return self.validate_place(place_base);
                    }
                }
                return Err(Unpromotable);
            }

            Rvalue::Ref(_, kind, place) => {
                // Special-case reborrows to be more like a copy of the reference.
                let mut place_simplified = place.as_ref();
                if let Some((place_base, ProjectionElem::Deref)) =
                    place_simplified.last_projection()
                {
                    let base_ty = place_base.ty(self.body, self.tcx).ty;
                    if let ty::Ref(..) = base_ty.kind() {
                        place_simplified = place_base;
                    }
                }

                self.validate_place(place_simplified)?;

                // Check that the reference is fine (using the original place!).
                // (Needs to come after `validate_place` to avoid ICEs.)
                self.validate_ref(*kind, place)?;
            }

            Rvalue::Aggregate(_, operands) => {
                for o in operands {
                    self.validate_operand(o)?;
                }
            }
        }

        Ok(())
    }

    fn validate_call(
        &self,
        callee: &Operand<'tcx>,
        args: &[Operand<'tcx>],
    ) -> Result<(), Unpromotable> {
        let fn_ty = callee.ty(self.body, self.tcx);

        // Inside const/static items, we promote all (eligible) function calls.
        // Everywhere else, we require `#[rustc_promotable]` on the callee.
        let promote_all_const_fn = matches!(
            self.const_kind,
            Some(hir::ConstContext::Static(_) | hir::ConstContext::Const)
        );
        if !promote_all_const_fn {
            if let ty::FnDef(def_id, _) = *fn_ty.kind() {
                // Never promote runtime `const fn` calls of
                // functions without `#[rustc_promotable]`.
                if !self.tcx.is_promotable_const_fn(def_id) {
                    return Err(Unpromotable);
                }
            }
        }

        let is_const_fn = match *fn_ty.kind() {
            ty::FnDef(def_id, _) => {
                is_const_fn(self.tcx, def_id)
                    || is_unstable_const_fn(self.tcx, def_id).is_some()
                    || is_lang_panic_fn(self.tcx, def_id)
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
    ccx: &ConstCx<'_, '_>,
    temps: &IndexVec<Local, TempState>,
    candidates: &[Candidate],
) -> Vec<Candidate> {
    let validator = Validator { ccx, temps };

    candidates
        .iter()
        .copied()
        .filter(|&candidate| validator.validate_candidate(candidate).is_ok())
        .collect()
}

struct Promoter<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    source: &'a mut Body<'tcx>,
    promoted: Body<'tcx>,
    temps: &'a mut IndexVec<Local, TempState>,
    extra_statements: &'a mut Vec<(Location, Statement<'tcx>)>,

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
                source_info: SourceInfo::outermost(span),
                kind: TerminatorKind::Return,
            }),
            is_cleanup: false,
        })
    }

    fn assign(&mut self, dest: Local, rvalue: Rvalue<'tcx>, span: Span) {
        let last = self.promoted.basic_blocks().last().unwrap();
        let data = &mut self.promoted[last];
        data.statements.push(Statement {
            source_info: SourceInfo::outermost(span),
            kind: StatementKind::Assign(Box::new((Place::from(dest), rvalue))),
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
            state => {
                span_bug!(self.promoted.span, "{:?} not promotable: {:?}", temp, state);
            }
        };
        if !self.keep_original {
            self.temps[temp] = TempState::PromotedOut;
        }

        let num_stmts = self.source[loc.block].statements.len();
        let new_temp = self.promoted.local_decls.push(LocalDecl::new(
            self.source.local_decls[temp].ty,
            self.source.local_decls[temp].source_info.span,
        ));

        debug!("promote({:?} @ {:?}/{:?}, {:?})", temp, loc, num_stmts, self.keep_original);

        // First, take the Rvalue or Call out of the source MIR,
        // or duplicate it, depending on keep_original.
        if loc.statement_index < num_stmts {
            let (mut rvalue, source_info) = {
                let statement = &mut self.source[loc.block].statements[loc.statement_index];
                let rhs = match statement.kind {
                    StatementKind::Assign(box (_, ref mut rhs)) => rhs,
                    _ => {
                        span_bug!(
                            statement.source_info.span,
                            "{:?} is not an assignment",
                            statement
                        );
                    }
                };

                (
                    if self.keep_original {
                        rhs.clone()
                    } else {
                        let unit = Rvalue::Use(Operand::Constant(Box::new(Constant {
                            span: statement.source_info.span,
                            user_ty: None,
                            literal: ty::Const::zero_sized(self.tcx, self.tcx.types.unit).into(),
                        })));
                        mem::replace(rhs, unit)
                    },
                    statement.source_info,
                )
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
                    kind: mem::replace(&mut terminator.kind, TerminatorKind::Goto { target }),
                }
            };

            match terminator.kind {
                TerminatorKind::Call { mut func, mut args, from_hir_call, fn_span, .. } => {
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
                            destination: Some((Place::from(new_temp), new_target)),
                            from_hir_call,
                            fn_span,
                        },
                        source_info: SourceInfo::outermost(terminator.source_info.span),
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
        candidate: Candidate,
        next_promoted_id: usize,
    ) -> Option<Body<'tcx>> {
        let def = self.source.source.with_opt_param();
        let mut rvalue = {
            let promoted = &mut self.promoted;
            let promoted_id = Promoted::new(next_promoted_id);
            let tcx = self.tcx;
            let mut promoted_operand = |ty, span| {
                promoted.span = span;
                promoted.local_decls[RETURN_PLACE] = LocalDecl::new(ty, span);

                Operand::Constant(Box::new(Constant {
                    span,
                    user_ty: None,
                    literal: tcx
                        .mk_const(ty::Const {
                            ty,
                            val: ty::ConstKind::Unevaluated(ty::Unevaluated {
                                def,
                                substs_: Some(InternalSubsts::for_item(
                                    tcx,
                                    def.did,
                                    |param, _| {
                                        if let ty::GenericParamDefKind::Lifetime = param.kind {
                                            tcx.lifetimes.re_erased.into()
                                        } else {
                                            tcx.mk_param_from_def(param)
                                        }
                                    },
                                )),
                                promoted: Some(promoted_id),
                            }),
                        })
                        .into(),
                }))
            };
            let (blocks, local_decls) = self.source.basic_blocks_and_local_decls_mut();
            match candidate {
                Candidate::Ref(loc) => {
                    let statement = &mut blocks[loc.block].statements[loc.statement_index];
                    match statement.kind {
                        StatementKind::Assign(box (
                            _,
                            Rvalue::Ref(ref mut region, borrow_kind, ref mut place),
                        )) => {
                            // Use the underlying local for this (necessarily interior) borrow.
                            let ty = local_decls.local_decls()[place.local].ty;
                            let span = statement.source_info.span;

                            let ref_ty = tcx.mk_ref(
                                tcx.lifetimes.re_erased,
                                ty::TypeAndMut { ty, mutbl: borrow_kind.to_mutbl_lossy() },
                            );

                            *region = tcx.lifetimes.re_erased;

                            let mut projection = vec![PlaceElem::Deref];
                            projection.extend(place.projection);
                            place.projection = tcx.intern_place_elems(&projection);

                            // Create a temp to hold the promoted reference.
                            // This is because `*r` requires `r` to be a local,
                            // otherwise we would use the `promoted` directly.
                            let mut promoted_ref = LocalDecl::new(ref_ty, span);
                            promoted_ref.source_info = statement.source_info;
                            let promoted_ref = local_decls.push(promoted_ref);
                            assert_eq!(self.temps.push(TempState::Unpromotable), promoted_ref);

                            let promoted_ref_statement = Statement {
                                source_info: statement.source_info,
                                kind: StatementKind::Assign(Box::new((
                                    Place::from(promoted_ref),
                                    Rvalue::Use(promoted_operand(ref_ty, span)),
                                ))),
                            };
                            self.extra_statements.push((loc, promoted_ref_statement));

                            Rvalue::Ref(
                                tcx.lifetimes.re_erased,
                                borrow_kind,
                                Place {
                                    local: mem::replace(&mut place.local, promoted_ref),
                                    projection: List::empty(),
                                },
                            )
                        }
                        _ => bug!(),
                    }
                }
            }
        };

        assert_eq!(self.new_block(), START_BLOCK);
        self.visit_rvalue(
            &mut rvalue,
            Location { block: BasicBlock::new(0), statement_index: usize::MAX },
        );

        let span = self.promoted.span;
        self.assign(RETURN_PLACE, rvalue, span);
        Some(self.promoted)
    }
}

/// Replaces all temporaries with their promoted counterparts.
impl<'a, 'tcx> MutVisitor<'tcx> for Promoter<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        if self.is_temp_kind(*local) {
            *local = self.promote_temp(*local);
        }
    }
}

pub fn promote_candidates<'tcx>(
    body: &mut Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    mut temps: IndexVec<Local, TempState>,
    candidates: Vec<Candidate>,
) -> IndexVec<Promoted, Body<'tcx>> {
    // Visit candidates in reverse, in case they're nested.
    debug!("promote_candidates({:?})", candidates);

    let mut promotions = IndexVec::new();

    let mut extra_statements = vec![];
    for candidate in candidates.into_iter().rev() {
        match candidate {
            Candidate::Ref(Location { block, statement_index }) => {
                if let StatementKind::Assign(box (place, _)) =
                    &body[block].statements[statement_index].kind
                {
                    if let Some(local) = place.as_local() {
                        if temps[local] == TempState::PromotedOut {
                            // Already promoted.
                            continue;
                        }
                    }
                }
            }
        }

        // Declare return place local so that `mir::Body::new` doesn't complain.
        let initial_locals = iter::once(LocalDecl::new(tcx.types.never, body.span)).collect();

        let mut scope = body.source_scopes[candidate.source_info(body).scope].clone();
        scope.parent_scope = None;

        let promoted = Body::new(
            tcx,
            body.source, // `promoted` gets filled in below
            IndexVec::new(),
            IndexVec::from_elem_n(scope, 1),
            initial_locals,
            IndexVec::new(),
            0,
            vec![],
            body.span,
            body.generator_kind(),
        );

        let promoter = Promoter {
            promoted,
            tcx,
            source: body,
            temps: &mut temps,
            extra_statements: &mut extra_statements,
            keep_original: false,
        };

        //FIXME(oli-obk): having a `maybe_push()` method on `IndexVec` might be nice
        if let Some(mut promoted) = promoter.promote_candidate(candidate, promotions.len()) {
            promoted.source.promoted = Some(promotions.next_index());
            promotions.push(promoted);
        }
    }

    // Insert each of `extra_statements` before its indicated location, which
    // has to be done in reverse location order, to not invalidate the rest.
    extra_statements.sort_by_key(|&(loc, _)| cmp::Reverse(loc));
    for (loc, statement) in extra_statements {
        body[loc.block].statements.insert(loc.statement_index, statement);
    }

    // Eliminate assignments to, and drops of promoted temps.
    let promoted = |index: Local| temps[index] == TempState::PromotedOut;
    for block in body.basic_blocks_mut() {
        block.statements.retain(|statement| match &statement.kind {
            StatementKind::Assign(box (place, _)) => {
                if let Some(index) = place.as_local() {
                    !promoted(index)
                } else {
                    true
                }
            }
            StatementKind::StorageLive(index) | StatementKind::StorageDead(index) => {
                !promoted(*index)
            }
            _ => true,
        });
        let terminator = block.terminator_mut();
        if let TerminatorKind::Drop { place, target, .. } = &terminator.kind {
            if let Some(index) = place.as_local() {
                if promoted(index) {
                    terminator.kind = TerminatorKind::Goto { target: *target };
                }
            }
        }
    }

    promotions
}

/// This function returns `true` if the function being called in the array
/// repeat expression is a `const` function.
pub fn is_const_fn_in_array_repeat_expression<'tcx>(
    ccx: &ConstCx<'_, 'tcx>,
    place: &Place<'tcx>,
    body: &Body<'tcx>,
) -> bool {
    match place.as_local() {
        // rule out cases such as: `let my_var = some_fn(); [my_var; N]`
        Some(local) if body.local_decls[local].is_user_variable() => return false,
        None => return false,
        _ => {}
    }

    for block in body.basic_blocks() {
        if let Some(Terminator { kind: TerminatorKind::Call { func, destination, .. }, .. }) =
            &block.terminator
        {
            if let Operand::Constant(box Constant { literal, .. }) = func {
                if let ty::FnDef(def_id, _) = *literal.ty().kind() {
                    if let Some((destination_place, _)) = destination {
                        if destination_place == place {
                            if is_const_fn(ccx.tcx, def_id) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }

    false
}
