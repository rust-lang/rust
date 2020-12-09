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

use rustc_ast::LitKind;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::traversal::ReversePostorder;
use rustc_middle::mir::visit::{MutVisitor, MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::cast::CastTy;
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::{self, List, TyCtxt, TypeFoldable};
use rustc_span::symbol::sym;
use rustc_span::Span;

use rustc_index::vec::{Idx, IndexVec};
use rustc_target::spec::abi::Abi;

use std::cell::Cell;
use std::{cmp, iter, mem};

use crate::const_eval::{is_const_fn, is_unstable_const_fn};
use crate::transform::check_consts::{is_lang_panic_fn, qualifs, ConstCx};
use crate::transform::MirPass;

/// A `MirPass` for promotion.
///
/// Promotion is the extraction of promotable temps into separate MIR bodies. This pass also emits
/// errors when promotion of `#[rustc_args_required_const]` arguments fails.
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
        matches!(self, TempState::Defined { .. } )
    }
}

/// A "root candidate" for promotion, which will become the
/// returned value in a promoted MIR, unless it's a subset
/// of a larger candidate.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Candidate {
    /// Borrow of a constant temporary, candidate for lifetime extension.
    Ref(Location),

    /// Promotion of the `x` in `[x; 32]`.
    Repeat(Location),

    /// Currently applied to function calls where the callee has the unstable
    /// `#[rustc_args_required_const]` attribute as well as the SIMD shuffle
    /// intrinsic. The intrinsic requires the arguments are indeed constant and
    /// the attribute currently provides the semantic requirement that arguments
    /// must be constant.
    Argument { bb: BasicBlock, index: usize },

    /// `const` operand in asm!.
    InlineAsm { bb: BasicBlock, index: usize },
}

impl Candidate {
    /// Returns `true` if we should use the "explicit" rules for promotability for this `Candidate`.
    fn forces_explicit_promotion(&self) -> bool {
        match self {
            Candidate::Ref(_) | Candidate::Repeat(_) => false,
            Candidate::Argument { .. } | Candidate::InlineAsm { .. } => true,
        }
    }

    fn source_info(&self, body: &Body<'_>) -> SourceInfo {
        match self {
            Candidate::Ref(location) | Candidate::Repeat(location) => *body.source_info(*location),
            Candidate::Argument { bb, .. } | Candidate::InlineAsm { bb, .. } => {
                *body.source_info(body.terminator_loc(*bb))
            }
        }
    }
}

fn args_required_const(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Vec<usize>> {
    let attrs = tcx.get_attrs(def_id);
    let attr = attrs.iter().find(|a| tcx.sess.check_name(a, sym::rustc_args_required_const))?;
    let mut ret = vec![];
    for meta in attr.meta_item_list()? {
        match meta.literal()?.kind {
            LitKind::Int(a, _) => {
                ret.push(a as usize);
            }
            _ => bug!("invalid arg index"),
        }
    }
    Some(ret)
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
            Rvalue::Repeat(..) if self.ccx.tcx.features().const_in_array_repeat_expressions => {
                // FIXME(#49147) only promote the element when it isn't `Copy`
                // (so that code that can copy it at runtime is unaffected).
                self.candidates.push(Candidate::Repeat(location));
            }
            _ => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);

        match terminator.kind {
            TerminatorKind::Call { ref func, .. } => {
                if let ty::FnDef(def_id, _) = *func.ty(self.ccx.body, self.ccx.tcx).kind() {
                    let fn_sig = self.ccx.tcx.fn_sig(def_id);
                    if let Abi::RustIntrinsic | Abi::PlatformIntrinsic = fn_sig.abi() {
                        let name = self.ccx.tcx.item_name(def_id);
                        // FIXME(eddyb) use `#[rustc_args_required_const(2)]` for shuffles.
                        if name.as_str().starts_with("simd_shuffle") {
                            self.candidates
                                .push(Candidate::Argument { bb: location.block, index: 2 });

                            return; // Don't double count `simd_shuffle` candidates
                        }
                    }

                    if let Some(constant_args) = args_required_const(self.ccx.tcx, def_id) {
                        for index in constant_args {
                            self.candidates.push(Candidate::Argument { bb: location.block, index });
                        }
                    }
                }
            }
            TerminatorKind::InlineAsm { ref operands, .. } => {
                for (index, op) in operands.iter().enumerate() {
                    if let InlineAsmOperand::Const { .. } = op {
                        self.candidates.push(Candidate::InlineAsm { bb: location.block, index })
                    }
                }
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

    /// Explicit promotion happens e.g. for constant arguments declared via
    /// `rustc_args_required_const`.
    /// Implicit promotion has almost the same rules, except that disallows `const fn`
    /// except for those marked `#[rustc_promotable]`. This is to avoid changing
    /// a legitimate run-time operation into a failing compile-time operation
    /// e.g. due to addresses being compared inside the function.
    explicit: bool,
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
                assert!(!self.explicit);

                let statement = &self.body[loc.block].statements[loc.statement_index];
                match &statement.kind {
                    StatementKind::Assign(box (_, Rvalue::Ref(_, kind, place))) => {
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
                        self.validate_local(place.local)?;

                        if place.projection.contains(&ProjectionElem::Deref) {
                            return Err(Unpromotable);
                        }
                        if self.qualif_local::<qualifs::NeedsDrop>(place.local) {
                            return Err(Unpromotable);
                        }

                        // FIXME(eddyb) this duplicates part of `validate_rvalue`.
                        let has_mut_interior =
                            self.qualif_local::<qualifs::HasMutInterior>(place.local);
                        if has_mut_interior {
                            return Err(Unpromotable);
                        }

                        if let BorrowKind::Mut { .. } = kind {
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

                        Ok(())
                    }
                    _ => bug!(),
                }
            }
            Candidate::Repeat(loc) => {
                assert!(!self.explicit);

                let statement = &self.body[loc.block].statements[loc.statement_index];
                match &statement.kind {
                    StatementKind::Assign(box (_, Rvalue::Repeat(ref operand, _))) => {
                        if !self.tcx.features().const_in_array_repeat_expressions {
                            return Err(Unpromotable);
                        }

                        self.validate_operand(operand)
                    }
                    _ => bug!(),
                }
            }
            Candidate::Argument { bb, index } => {
                assert!(self.explicit);

                let terminator = self.body[bb].terminator();
                match &terminator.kind {
                    TerminatorKind::Call { args, .. } => self.validate_operand(&args[index]),
                    _ => bug!(),
                }
            }
            Candidate::InlineAsm { bb, index } => {
                assert!(self.explicit);

                let terminator = self.body[bb].terminator();
                match &terminator.kind {
                    TerminatorKind::InlineAsm { operands, .. } => match &operands[index] {
                        InlineAsmOperand::Const { value } => self.validate_operand(value),
                        _ => bug!(),
                    },
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
            let num_stmts = self.body[loc.block].statements.len();

            if loc.statement_index < num_stmts {
                let statement = &self.body[loc.block].statements[loc.statement_index];
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
                let terminator = self.body[loc.block].terminator();
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
        match place {
            PlaceRef { local, projection: [] } => self.validate_local(local),
            PlaceRef { local, projection: [proj_base @ .., elem] } => {
                // Validate topmost projection, then recurse.
                match *elem {
                    ProjectionElem::Deref => {
                        let mut promotable = false;
                        // This is a special treatment for cases like *&STATIC where STATIC is a
                        // global static variable.
                        // This pattern is generated only when global static variables are directly
                        // accessed and is qualified for promotion safely.
                        if let TempState::Defined { location, .. } = self.temps[local] {
                            let def_stmt =
                                self.body[location.block].statements.get(location.statement_index);
                            if let Some(Statement {
                                kind:
                                    StatementKind::Assign(box (_, Rvalue::Use(Operand::Constant(c)))),
                                ..
                            }) = def_stmt
                            {
                                if let Some(did) = c.check_static_ptr(self.tcx) {
                                    // Evaluating a promoted may not read statics except if it got
                                    // promoted from a static (this is a CTFE check). So we
                                    // can only promote static accesses inside statics.
                                    if let Some(hir::ConstContext::Static(..)) = self.const_kind {
                                        // The `is_empty` predicate is introduced to exclude the case
                                        // where the projection operations are [ .field, * ].
                                        // The reason is because promotion will be illegal if field
                                        // accesses precede the dereferencing.
                                        // Discussion can be found at
                                        // https://github.com/rust-lang/rust/pull/74945#discussion_r463063247
                                        // There may be opportunity for generalization, but this needs to be
                                        // accounted for.
                                        if proj_base.is_empty()
                                            && !self.tcx.is_thread_local_static(did)
                                        {
                                            promotable = true;
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
                        self.validate_local(local)?;
                    }

                    ProjectionElem::Field(..) => {
                        let base_ty =
                            Place::ty_from(place.local, proj_base, self.body, self.tcx).ty;
                        if let Some(def) = base_ty.ty_adt_def() {
                            // No promotion of union field accesses.
                            if def.is_union() {
                                return Err(Unpromotable);
                            }
                        }
                    }
                }

                self.validate_place(PlaceRef { local: place.local, projection: proj_base })
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

    fn validate_rvalue(&self, rvalue: &Rvalue<'tcx>) -> Result<(), Unpromotable> {
        match *rvalue {
            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) => {
                let operand_ty = operand.ty(self.body, self.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                if let (CastTy::Ptr(_) | CastTy::FnPtr, CastTy::Int(_)) = (cast_in, cast_out) {
                    // ptr-to-int casts are not possible in consts and thus not promotable
                    return Err(Unpromotable);
                }
            }

            Rvalue::BinaryOp(op, ref lhs, _) => {
                if let ty::RawPtr(_) | ty::FnPtr(..) = lhs.ty(self.body, self.tcx).kind() {
                    assert!(
                        op == BinOp::Eq
                            || op == BinOp::Ne
                            || op == BinOp::Le
                            || op == BinOp::Lt
                            || op == BinOp::Ge
                            || op == BinOp::Gt
                            || op == BinOp::Offset
                    );

                    // raw pointer operations are not allowed inside consts and thus not promotable
                    return Err(Unpromotable);
                }
            }

            Rvalue::NullaryOp(NullOp::Box, _) => return Err(Unpromotable),

            // FIXME(RalfJung): the rest is *implicitly considered promotable*... that seems dangerous.
            _ => {}
        }

        match rvalue {
            Rvalue::ThreadLocalRef(_) => Err(Unpromotable),

            Rvalue::NullaryOp(..) => Ok(()),

            Rvalue::Discriminant(place) | Rvalue::Len(place) => self.validate_place(place.as_ref()),

            Rvalue::Use(operand)
            | Rvalue::Repeat(operand, _)
            | Rvalue::UnaryOp(_, operand)
            | Rvalue::Cast(_, operand, _) => self.validate_operand(operand),

            Rvalue::BinaryOp(_, lhs, rhs) | Rvalue::CheckedBinaryOp(_, lhs, rhs) => {
                self.validate_operand(lhs)?;
                self.validate_operand(rhs)
            }

            Rvalue::AddressOf(_, place) => {
                // We accept `&raw *`, i.e., raw reborrows -- creating a raw pointer is
                // no problem, only using it is.
                if let [proj_base @ .., ProjectionElem::Deref] = place.projection.as_ref() {
                    let base_ty = Place::ty_from(place.local, proj_base, self.body, self.tcx).ty;
                    if let ty::Ref(..) = base_ty.kind() {
                        return self.validate_place(PlaceRef {
                            local: place.local,
                            projection: proj_base,
                        });
                    }
                }
                Err(Unpromotable)
            }

            Rvalue::Ref(_, kind, place) => {
                if let BorrowKind::Mut { .. } = kind {
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

                // Special-case reborrows to be more like a copy of the reference.
                let mut place = place.as_ref();
                if let [proj_base @ .., ProjectionElem::Deref] = &place.projection {
                    let base_ty = Place::ty_from(place.local, proj_base, self.body, self.tcx).ty;
                    if let ty::Ref(..) = base_ty.kind() {
                        place = PlaceRef { local: place.local, projection: proj_base };
                    }
                }

                self.validate_place(place)?;

                let has_mut_interior = self.qualif_local::<qualifs::HasMutInterior>(place.local);
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

        // When doing explicit promotion and inside const/static items, we promote all (eligible) function calls.
        // Everywhere else, we require `#[rustc_promotable]` on the callee.
        let promote_all_const_fn = self.explicit
            || matches!(
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
    let mut validator = Validator { ccx, temps, explicit: false };

    candidates
        .iter()
        .copied()
        .filter(|&candidate| {
            validator.explicit = candidate.forces_explicit_promotion();

            // FIXME(eddyb) also emit the errors for shuffle indices
            // and `#[rustc_args_required_const]` arguments here.

            let is_promotable = validator.validate_candidate(candidate).is_ok();

            // If we use explicit validation, we carry the risk of turning a legitimate run-time
            // operation into a failing compile-time operation. Make sure that does not happen
            // by asserting that there is no possible run-time behavior here in case promotion
            // fails.
            if validator.explicit && !is_promotable {
                ccx.tcx.sess.delay_span_bug(
                    ccx.body.span,
                    "Explicit promotion requested, but failed to promote",
                );
            }

            match candidate {
                Candidate::Argument { bb, index } | Candidate::InlineAsm { bb, index }
                    if !is_promotable =>
                {
                    let span = ccx.body[bb].terminator().source_info.span;
                    let msg = format!("argument {} is required to be a constant", index + 1);
                    ccx.tcx.sess.span_err(span, &msg);
                }
                _ => (),
            }

            is_promotable
        })
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
            kind: StatementKind::Assign(box (Place::from(dest), rvalue)),
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
                        let unit = Rvalue::Use(Operand::Constant(box Constant {
                            span: statement.source_info.span,
                            user_ty: None,
                            literal: ty::Const::zero_sized(self.tcx, self.tcx.types.unit),
                        }));
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
                    literal: tcx.mk_const(ty::Const {
                        ty,
                        val: ty::ConstKind::Unevaluated(
                            def,
                            InternalSubsts::for_item(tcx, def.did, |param, _| {
                                if let ty::GenericParamDefKind::Lifetime = param.kind {
                                    tcx.lifetimes.re_erased.into()
                                } else {
                                    tcx.mk_param_from_def(param)
                                }
                            }),
                            Some(promoted_id),
                        ),
                    }),
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
                Candidate::Repeat(loc) => {
                    let statement = &mut blocks[loc.block].statements[loc.statement_index];
                    match statement.kind {
                        StatementKind::Assign(box (_, Rvalue::Repeat(ref mut operand, _))) => {
                            let ty = operand.ty(local_decls, self.tcx);
                            let span = statement.source_info.span;

                            Rvalue::Use(mem::replace(operand, promoted_operand(ty, span)))
                        }
                        _ => bug!(),
                    }
                }
                Candidate::Argument { bb, index } => {
                    let terminator = blocks[bb].terminator_mut();
                    match terminator.kind {
                        TerminatorKind::Call { ref mut args, .. } => {
                            let ty = args[index].ty(local_decls, self.tcx);
                            let span = terminator.source_info.span;

                            Rvalue::Use(mem::replace(&mut args[index], promoted_operand(ty, span)))
                        }
                        // We expected a `TerminatorKind::Call` for which we'd like to promote an
                        // argument. `qualify_consts` saw a `TerminatorKind::Call` here, but
                        // we are seeing a `Goto`. That means that the `promote_temps` method
                        // already promoted this call away entirely. This case occurs when calling
                        // a function requiring a constant argument and as that constant value
                        // providing a value whose computation contains another call to a function
                        // requiring a constant argument.
                        TerminatorKind::Goto { .. } => return None,
                        _ => bug!(),
                    }
                }
                Candidate::InlineAsm { bb, index } => {
                    let terminator = blocks[bb].terminator_mut();
                    match terminator.kind {
                        TerminatorKind::InlineAsm { ref mut operands, .. } => {
                            match &mut operands[index] {
                                InlineAsmOperand::Const { ref mut value } => {
                                    let ty = value.ty(local_decls, self.tcx);
                                    let span = terminator.source_info.span;

                                    Rvalue::Use(mem::replace(value, promoted_operand(ty, span)))
                                }
                                _ => bug!(),
                            }
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
            Candidate::Repeat(Location { block, statement_index })
            | Candidate::Ref(Location { block, statement_index }) => {
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
            Candidate::Argument { .. } | Candidate::InlineAsm { .. } => {}
        }

        // Declare return place local so that `mir::Body::new` doesn't complain.
        let initial_locals = iter::once(LocalDecl::new(tcx.types.never, body.span)).collect();

        let mut scope = body.source_scopes[candidate.source_info(body).scope].clone();
        scope.parent_scope = None;

        let promoted = Body::new(
            body.source, // `promoted` gets filled in below
            IndexVec::new(),
            IndexVec::from_elem_n(scope, 1),
            initial_locals,
            IndexVec::new(),
            0,
            vec![],
            body.span,
            body.generator_kind,
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

/// This function returns `true` if the `const_in_array_repeat_expressions` feature attribute should
/// be suggested. This function is probably quite expensive, it shouldn't be run in the happy path.
/// Feature attribute should be suggested if `operand` can be promoted and the feature is not
/// enabled.
crate fn should_suggest_const_in_array_repeat_expressions_attribute<'tcx>(
    ccx: &ConstCx<'_, 'tcx>,
    operand: &Operand<'tcx>,
) -> bool {
    let mut rpo = traversal::reverse_postorder(&ccx.body);
    let (temps, _) = collect_temps_and_candidates(&ccx, &mut rpo);
    let validator = Validator { ccx, temps: &temps, explicit: false };

    let should_promote = validator.validate_operand(operand).is_ok();
    let feature_flag = validator.ccx.tcx.features().const_in_array_repeat_expressions;
    debug!(
        "should_suggest_const_in_array_repeat_expressions_flag: def_id={:?} \
            should_promote={:?} feature_flag={:?}",
        validator.ccx.def_id(),
        should_promote,
        feature_flag
    );
    should_promote && !feature_flag
}
