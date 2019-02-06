//! A pass that qualifies constness of temporaries in constants,
//! static initializers and functions and also drives promotion.
//!
//! The Qualif flags below can be used to also provide better
//! diagnostics as to why a constant rvalue wasn't promoted.

use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lrc;
use rustc_target::spec::abi::Abi;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::traits::{self, TraitEngine};
use rustc::ty::{self, TyCtxt, Ty, TypeFoldable};
use rustc::ty::cast::CastTy;
use rustc::ty::query::Providers;
use rustc::mir::*;
use rustc::mir::traversal::ReversePostorder;
use rustc::mir::visit::{PlaceContext, Visitor, MutatingUseContext, NonMutatingUseContext};
use rustc::middle::lang_items;
use rustc::session::config::nightly_options;
use syntax::ast::LitKind;
use syntax::feature_gate::{emit_feature_err, GateIssue};
use syntax_pos::{Span, DUMMY_SP};

use std::fmt;
use std::usize;

use crate::transform::{MirPass, MirSource};
use super::promote_consts::{self, Candidate, TempState};

bitflags::bitflags! {
    // Borrows of temporaries can be promoted only if
    // they have none of these qualifications, with
    // the exception of `STATIC_REF` (in statics only).
    struct Qualif: u8 {
        // Constant containing interior mutability (UnsafeCell).
        const MUTABLE_INTERIOR  = 1 << 0;

        // Constant containing an ADT that implements Drop.
        const NEEDS_DROP        = 1 << 1;

        // Not constant at all - non-`const fn` calls, asm!,
        // pointer comparisons, ptr-to-int casts, etc.
        const NOT_CONST         = 1 << 2;

        // Refers to temporaries which cannot be promoted as
        // promote_consts decided they weren't simple enough.
        const NOT_PROMOTABLE    = 1 << 3;

        // Const items can only have MUTABLE_INTERIOR
        // and NOT_PROMOTABLE without producing an error.
        const CONST_ERROR       = !Qualif::MUTABLE_INTERIOR.bits &
                                  !Qualif::NOT_PROMOTABLE.bits;
    }
}

impl<'a, 'tcx> Qualif {
    /// Compute the qualifications for the given type.
    fn any_value_of_ty(
        ty: Ty<'tcx>,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        let mut qualif = Self::empty();
        if !ty.is_freeze(tcx, param_env, DUMMY_SP) {
            qualif = qualif | Qualif::MUTABLE_INTERIOR;
        }
        if ty.needs_drop(tcx, param_env) {
            qualif = qualif | Qualif::NEEDS_DROP;
        }
        qualif
    }

    /// Remove flags which are impossible for the given type.
    fn restrict(
        &mut self,
        ty: Ty<'tcx>,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) {
        let ty_qualif = Self::any_value_of_ty(ty, tcx, param_env);
        if !ty_qualif.contains(Qualif::MUTABLE_INTERIOR) {
            *self = *self - Qualif::MUTABLE_INTERIOR;
        }
        if !ty_qualif.contains(Qualif::NEEDS_DROP) {
            *self = *self - Qualif::NEEDS_DROP;
        }
    }
}

/// What kind of item we are in.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Mode {
    Const,
    Static,
    StaticMut,
    ConstFn,
    Fn
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Mode::Const => write!(f, "constant"),
            Mode::Static | Mode::StaticMut => write!(f, "static"),
            Mode::ConstFn => write!(f, "constant function"),
            Mode::Fn => write!(f, "function")
        }
    }
}

struct Qualifier<'a, 'tcx> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    mode: Mode,
    mir: &'a Mir<'tcx>,

    local_qualif: &'a IndexVec<Local, Qualif>,
}

impl<'a, 'tcx> Qualifier<'a, 'tcx> {
    fn qualify_any_value_of_ty(&self, ty: Ty<'tcx>) -> Qualif {
        Qualif::any_value_of_ty(ty, self.tcx, self.param_env)
    }

    fn qualify_local(&self, local: Local) -> Qualif {
        self.local_qualif[local]
    }

    fn qualify_projection_elem(&self, proj: &PlaceElem<'tcx>) -> Qualif {
        match *proj {
            ProjectionElem::Deref |
            ProjectionElem::Subslice { .. } |
            ProjectionElem::Field(..) |
            ProjectionElem::ConstantIndex { .. } |
            ProjectionElem::Downcast(..) => Qualif::empty(),

            ProjectionElem::Index(local) => self.qualify_local(local),
        }
    }

    fn qualify_place(&self, place: &Place<'tcx>) -> Qualif {
        match *place {
            Place::Local(local) => self.qualify_local(local),
            Place::Promoted(_) => bug!("qualifying already promoted MIR"),
            Place::Static(ref global) => {
                if self.tcx
                       .get_attrs(global.def_id)
                       .iter()
                       .any(|attr| attr.check_name("thread_local")) {
                    return Qualif::NOT_CONST;
                }

                // Only allow statics (not consts) to refer to other statics.
                if self.mode == Mode::Static || self.mode == Mode::StaticMut {
                    Qualif::empty()
                } else {
                    Qualif::NOT_CONST
                }
            }
            Place::Projection(ref proj) => {
                let mut qualif =
                    self.qualify_place(&proj.base) |
                    self.qualify_projection_elem(&proj.elem);
                match proj.elem {
                    ProjectionElem::Deref |
                    ProjectionElem::Downcast(..) => qualif | Qualif::NOT_CONST,

                    ProjectionElem::ConstantIndex {..} |
                    ProjectionElem::Subslice {..} |
                    ProjectionElem::Field(..) |
                    ProjectionElem::Index(_) => {
                        let base_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                        if let Some(def) = base_ty.ty_adt_def() {
                            if def.is_union() {
                                match self.mode {
                                    Mode::Fn => qualif = qualif | Qualif::NOT_CONST,

                                    Mode::ConstFn |
                                    Mode::Static |
                                    Mode::StaticMut |
                                    Mode::Const => {}
                                }
                            }
                        }

                        let ty = place.ty(self.mir, self.tcx).to_ty(self.tcx);
                        qualif.restrict(ty, self.tcx, self.param_env);
                        qualif
                    }
                }
            }
        }
    }

    fn qualify_operand(&self, operand: &Operand<'tcx>) -> Qualif {
        match *operand {
            Operand::Copy(ref place) |
            Operand::Move(ref place) => self.qualify_place(place),

            Operand::Constant(ref constant) => {
                if let ty::LazyConst::Unevaluated(def_id, _) = constant.literal {
                    // Don't peek inside trait associated constants.
                    if self.tcx.trait_of_item(*def_id).is_some() {
                        self.qualify_any_value_of_ty(constant.ty)
                    } else {
                        let (bits, _) = self.tcx.at(constant.span).mir_const_qualif(*def_id);

                        let mut qualif = Qualif::from_bits(bits).expect("invalid mir_const_qualif");

                        // Just in case the type is more specific than
                        // the definition, e.g., impl associated const
                        // with type parameters, take it into account.
                        qualif.restrict(constant.ty, self.tcx, self.param_env);
                        qualif
                    }
                } else {
                    Qualif::empty()
                }
            }
        }
    }

    fn qualify_rvalue(&self, rvalue: &Rvalue<'tcx>) -> Qualif {
        match *rvalue {
            Rvalue::NullaryOp(NullOp::SizeOf, _) => Qualif::empty(),

            Rvalue::Use(ref operand) |
            Rvalue::Repeat(ref operand, _) |
            Rvalue::UnaryOp(UnOp::Neg, ref operand) |
            Rvalue::UnaryOp(UnOp::Not, ref operand) |
            Rvalue::Cast(CastKind::ReifyFnPointer, ref operand, _) |
            Rvalue::Cast(CastKind::UnsafeFnPointer, ref operand, _) |
            Rvalue::Cast(CastKind::ClosureFnPointer, ref operand, _) |
            Rvalue::Cast(CastKind::Unsize, ref operand, _) => {
                self.qualify_operand(operand)
            }

            Rvalue::CheckedBinaryOp(_, ref lhs, ref rhs) => {
                self.qualify_operand(lhs) | self.qualify_operand(rhs)
            }

            Rvalue::Discriminant(ref place) |
            Rvalue::Len(ref place) => self.qualify_place(place),

            Rvalue::Ref(_, kind, ref place) => {
                let mut reborrow_qualif = None;
                if let Place::Projection(ref proj) = *place {
                    if let ProjectionElem::Deref = proj.elem {
                        let base_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                        if let ty::Ref(..) = base_ty.sty {
                            reborrow_qualif = Some(self.qualify_place(&proj.base));
                        }
                    }
                }

                let mut qualif = reborrow_qualif.unwrap_or_else(|| {
                    self.qualify_place(place)
                });

                let ty = place.ty(self.mir, self.tcx).to_ty(self.tcx);

                if let BorrowKind::Mut { .. } = kind {
                    // In theory, any zero-sized value could be borrowed
                    // mutably without consequences. However, only &mut []
                    // is allowed right now, and only in functions.
                    let allowed = if self.mode == Mode::StaticMut {
                        // Inside a `static mut`, &mut [...] is also allowed.
                        match ty.sty {
                            ty::Array(..) | ty::Slice(_) => true,
                            _ => false
                        }
                    } else if let ty::Array(_, len) = ty.sty {
                        // FIXME(eddyb) the `self.mode == Mode::Fn` condition
                        // seems unnecessary, given that this is merely a ZST.
                        len.unwrap_usize(self.tcx) == 0 && self.mode == Mode::Fn
                    } else {
                        false
                    };

                    if !allowed {
                        qualif = qualif | Qualif::MUTABLE_INTERIOR;
                    }
                }

                qualif
            }

            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) => {
                let mut qualif = self.qualify_operand(operand);

                let operand_ty = operand.ty(self.mir, self.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                match (cast_in, cast_out) {
                    (CastTy::Ptr(_), CastTy::Int(_)) |
                    (CastTy::FnPtr, CastTy::Int(_)) => {
                        if let Mode::Fn = self.mode {
                            // in normal functions, mark such casts as not promotable
                            qualif = qualif | Qualif::NOT_CONST;
                        }
                    }
                    _ => {}
                }

                qualif
            }

            Rvalue::BinaryOp(op, ref lhs, ref rhs) => {
                let mut qualif = self.qualify_operand(lhs) | self.qualify_operand(rhs);

                if let ty::RawPtr(_) | ty::FnPtr(..) = lhs.ty(self.mir, self.tcx).sty {
                    assert!(op == BinOp::Eq || op == BinOp::Ne ||
                            op == BinOp::Le || op == BinOp::Lt ||
                            op == BinOp::Ge || op == BinOp::Gt ||
                            op == BinOp::Offset);

                    if let Mode::Fn = self.mode {
                        // raw pointer operations are not allowed inside promoteds
                        qualif = qualif | Qualif::NOT_CONST;
                    }
                }

                qualif
            }

            Rvalue::NullaryOp(NullOp::Box, _) => Qualif::NOT_CONST,

            Rvalue::Aggregate(ref kind, ref operands) => {
                let mut qualif = operands.iter().map(|o| self.qualify_operand(o))
                    .fold(Qualif::empty(), |a, b| a | b);

                if let AggregateKind::Adt(def, ..) = **kind {
                    if Some(def.did) == self.tcx.lang_items().unsafe_cell_type() {
                        let ty = rvalue.ty(self.mir, self.tcx);
                        qualif = qualif | self.qualify_any_value_of_ty(ty);
                        assert!(qualif.contains(Qualif::MUTABLE_INTERIOR));
                    }

                    if def.has_dtor(self.tcx) {
                        qualif = qualif | Qualif::NEEDS_DROP;
                    }
                }

                qualif
            }
        }
    }

    fn is_const_panic_fn(&self, def_id: DefId) -> bool {
        Some(def_id) == self.tcx.lang_items().panic_fn() ||
        Some(def_id) == self.tcx.lang_items().begin_panic_fn()
    }

    fn qualify_call(
        &self,
        callee: &Operand<'tcx>,
        args: &[Operand<'tcx>],
        return_ty: Ty<'tcx>,
    ) -> Qualif {
        let fn_ty = callee.ty(self.mir, self.tcx);
        let mut is_promotable_const_fn = false;
        let is_const_fn = match fn_ty.sty {
            ty::FnDef(def_id, _) => {
                match self.tcx.fn_sig(def_id).abi() {
                    Abi::RustIntrinsic |
                    Abi::PlatformIntrinsic => {
                        assert!(!self.tcx.is_const_fn(def_id));
                        match &self.tcx.item_name(def_id).as_str()[..] {
                            | "size_of"
                            | "min_align_of"
                            | "needs_drop"
                            | "type_id"
                            | "bswap"
                            | "bitreverse"
                            | "ctpop"
                            | "cttz"
                            | "cttz_nonzero"
                            | "ctlz"
                            | "ctlz_nonzero"
                            | "overflowing_add"
                            | "overflowing_sub"
                            | "overflowing_mul"
                            | "unchecked_shl"
                            | "unchecked_shr"
                            | "rotate_left"
                            | "rotate_right"
                            | "add_with_overflow"
                            | "sub_with_overflow"
                            | "mul_with_overflow"
                            | "saturating_add"
                            | "saturating_sub"
                            | "transmute"
                            => true,

                            _ => false,
                        }
                    }
                    _ => {
                        // Never promote runtime `const fn` calls of
                        // functions without `#[rustc_promotable]`.
                        if self.tcx.is_promotable_const_fn(def_id) {
                            is_promotable_const_fn = true;
                        }

                        if self.mode == Mode::Fn {
                            self.tcx.is_const_fn(def_id)
                        } else {
                            self.tcx.is_const_fn(def_id) ||
                            self.is_const_panic_fn(def_id) ||
                            self.tcx.is_unstable_const_fn(def_id).is_some()
                        }
                    }
                }
            }
            _ => false,
        };

        // Bail out on oon-`const fn` calls or if the callee had errors.
        if !is_const_fn || self.qualify_operand(callee).intersects(Qualif::CONST_ERROR) {
            return Qualif::NOT_CONST;
        }

        // Bail out if any arguments had errors.
        for arg in args {
            if self.qualify_operand(arg).intersects(Qualif::CONST_ERROR) {
                return Qualif::NOT_CONST;
            }
        }

        // Be conservative about the returned value of a const fn.
        let qualif = self.qualify_any_value_of_ty(return_ty);
        if !is_promotable_const_fn && self.mode == Mode::Fn {
            qualif | Qualif::NOT_PROMOTABLE
        } else {
            qualif
        }
    }
}

struct Checker<'a, 'tcx> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    mode: Mode,
    span: Span,
    def_id: DefId,
    mir: &'a Mir<'tcx>,
    rpo: ReversePostorder<'a, 'tcx>,

    local_qualif: IndexVec<Local, Qualif>,
    temp_promotion_state: IndexVec<Local, TempState>,
    promotion_candidates: Vec<Candidate>,
}

macro_rules! unleash_miri {
    ($this:expr) => {{
        if $this.tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you {
            $this.tcx.sess.span_warn($this.span, "skipping const checks");
            return;
        }
    }}
}

impl<'a, 'tcx> Checker<'a, 'tcx> {
    fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
           def_id: DefId,
           mir: &'a Mir<'tcx>,
           mode: Mode)
           -> Self {
        assert!(def_id.is_local());
        let mut rpo = traversal::reverse_postorder(mir);
        let temps = promote_consts::collect_temps(mir, &mut rpo);
        rpo.reset();

        let param_env = tcx.param_env(def_id);

        let local_qualif = mir.local_decls.iter_enumerated().map(|(local, decl)| {
            match mir.local_kind(local) {
                LocalKind::Arg => {
                    Qualif::any_value_of_ty(decl.ty, tcx, param_env) |
                        Qualif::NOT_PROMOTABLE
                }

                LocalKind::Var if mode == Mode::Fn => Qualif::NOT_CONST,

                LocalKind::Temp if !temps[local].is_promotable() => {
                    Qualif::NOT_PROMOTABLE
                }

                _ => Qualif::empty(),
            }
        }).collect();

        Checker {
            mode,
            span: mir.span,
            def_id,
            mir,
            rpo,
            tcx,
            param_env,
            local_qualif,
            temp_promotion_state: temps,
            promotion_candidates: vec![]
        }
    }

    fn qualifier(&'a self) -> Qualifier<'a, 'tcx> {
        Qualifier {
            tcx: self.tcx,
            param_env: self.param_env,
            mode: self.mode,
            mir: self.mir,
            local_qualif: &self.local_qualif,
        }
    }

    // FIXME(eddyb) we could split the errors into meaningful
    // categories, but enabling full miri would make that
    // slightly pointless (even with feature-gating).
    fn not_const(&mut self) {
        unleash_miri!(self);
        if self.mode != Mode::Fn {
            let mut err = struct_span_err!(
                self.tcx.sess,
                self.span,
                E0019,
                "{} contains unimplemented expression type",
                self.mode
            );
            if self.tcx.sess.teach(&err.get_code().unwrap()) {
                err.note("A function call isn't allowed in the const's initialization expression \
                          because the expression's value must be known at compile-time.");
                err.note("Remember: you can't use a function call inside a const's initialization \
                          expression! However, you can use it anywhere else.");
            }
            err.emit();
        }
    }

    /// Assigns an rvalue/call qualification to the given destination.
    fn assign(&mut self, dest: &Place<'tcx>, qualif: Qualif, location: Location) {
        trace!("assign: {:?} <- {:?}", dest, qualif);

        // Only handle promotable temps in non-const functions.
        if self.mode == Mode::Fn {
            if let Place::Local(index) = *dest {
                if self.mir.local_kind(index) == LocalKind::Temp
                && self.temp_promotion_state[index].is_promotable() {
                    debug!("store to promotable temp {:?} ({:?})", index, qualif);
                    let slot = &mut self.local_qualif[index];
                    if !slot.is_empty() {
                        span_bug!(self.span, "multiple assignments to {:?}", dest);
                    }
                    *slot = qualif;
                }
            }
            return;
        }

        let mut dest = dest;
        let index = loop {
            match dest {
                // We treat all locals equal in constants
                Place::Local(index) => break *index,
                // projections are transparent for assignments
                // we qualify the entire destination at once, even if just a field would have
                // stricter qualification
                Place::Projection(proj) => {
                    // Catch more errors in the destination. `visit_place` also checks various
                    // projection rules like union field access and raw pointer deref
                    self.visit_place(
                        dest,
                        PlaceContext::MutatingUse(MutatingUseContext::Store),
                        location
                    );
                    dest = &proj.base;
                },
                Place::Promoted(..) => bug!("promoteds don't exist yet during promotion"),
                Place::Static(..) => {
                    // Catch more errors in the destination. `visit_place` also checks that we
                    // do not try to access statics from constants or try to mutate statics
                    self.visit_place(
                        dest,
                        PlaceContext::MutatingUse(MutatingUseContext::Store),
                        location
                    );
                    return;
                }
            }
        };
        debug!("store to var {:?}", index);
        // this is overly restrictive, because even full assignments do not clear the qualif
        // While we could special case full assignments, this would be inconsistent with
        // aggregates where we overwrite all fields via assignments, which would not get
        // that feature.
        let slot = &mut self.local_qualif[index];
        *slot = *slot | qualif;

        // Ensure we keep the `NOT_PROMOTABLE` flag is preserved.
        // NOTE(eddyb) this is actually unnecessary right now, as
        // we never replace the local's qualif (but we might in
        // the future) - also, if `NOT_PROMOTABLE` only matters
        // for `Mode::Fn`, then this is also pointless.
        if self.mir.local_kind(index) == LocalKind::Temp {
            if !self.temp_promotion_state[index].is_promotable() {
                *slot = *slot | Qualif::NOT_PROMOTABLE;
            }
        }
    }

    /// Check a whole const, static initializer or const fn.
    fn check_const(&mut self) -> (Qualif, Lrc<BitSet<Local>>) {
        debug!("const-checking {} {:?}", self.mode, self.def_id);

        let mir = self.mir;

        let mut seen_blocks = BitSet::new_empty(mir.basic_blocks().len());
        let mut bb = START_BLOCK;
        loop {
            seen_blocks.insert(bb.index());

            self.visit_basic_block_data(bb, &mir[bb]);

            let target = match mir[bb].terminator().kind {
                TerminatorKind::Goto { target } |
                TerminatorKind::Drop { target, .. } |
                TerminatorKind::Assert { target, .. } |
                TerminatorKind::Call { destination: Some((_, target)), .. } => {
                    Some(target)
                }

                // Non-terminating calls cannot produce any value.
                TerminatorKind::Call { destination: None, .. } => {
                    break;
                }

                TerminatorKind::SwitchInt {..} |
                TerminatorKind::DropAndReplace { .. } |
                TerminatorKind::Resume |
                TerminatorKind::Abort |
                TerminatorKind::GeneratorDrop |
                TerminatorKind::Yield { .. } |
                TerminatorKind::Unreachable |
                TerminatorKind::FalseEdges { .. } |
                TerminatorKind::FalseUnwind { .. } => None,

                TerminatorKind::Return => {
                    break;
                }
            };

            match target {
                // No loops allowed.
                Some(target) if !seen_blocks.contains(target.index()) => {
                    bb = target;
                }
                _ => {
                    self.not_const();
                    break;
                }
            }
        }

        let mut qualif = self.local_qualif[RETURN_PLACE];

        // Account for errors in consts by using the
        // conservative type qualification instead.
        if qualif.intersects(Qualif::CONST_ERROR) {
            qualif = self.qualifier().qualify_any_value_of_ty(mir.return_ty());
        }


        // Collect all the temps we need to promote.
        let mut promoted_temps = BitSet::new_empty(self.temp_promotion_state.len());

        debug!("qualify_const: promotion_candidates={:?}", self.promotion_candidates);
        for candidate in &self.promotion_candidates {
            match *candidate {
                Candidate::Ref(Location { block: bb, statement_index: stmt_idx }) => {
                    match self.mir[bb].statements[stmt_idx].kind {
                        StatementKind::Assign(_, box Rvalue::Ref(_, _, Place::Local(index))) => {
                            promoted_temps.insert(index);
                        }
                        _ => {}
                    }
                }
                Candidate::Argument { .. } => {}
            }
        }

        (qualif, Lrc::new(promoted_temps))
    }
}

/// Checks MIR for const-correctness, using `Qualifier`
/// for value qualifications, and accumulates writes of
/// rvalue/call results to locals, in `local_qualif`.
/// For functions (constant or not), it also records
/// candidates for promotion in `promotion_candidates`.
impl<'a, 'tcx> Visitor<'tcx> for Checker<'a, 'tcx> {
    fn visit_place(&mut self,
                    place: &Place<'tcx>,
                    context: PlaceContext<'tcx>,
                    location: Location) {
        debug!("visit_place: place={:?} context={:?} location={:?}", place, context, location);
        self.super_place(place, context, location);
        match *place {
            Place::Local(_) |
            Place::Promoted(_) => {}
            Place::Static(ref global) => {
                if self.tcx
                       .get_attrs(global.def_id)
                       .iter()
                       .any(|attr| attr.check_name("thread_local")) {
                    if self.mode != Mode::Fn {
                        span_err!(self.tcx.sess, self.span, E0625,
                                  "thread-local statics cannot be \
                                   accessed at compile-time");
                    }
                    return;
                }

                // Only allow statics (not consts) to refer to other statics.
                if self.mode == Mode::Static || self.mode == Mode::StaticMut {
                    if self.mode == Mode::Static && context.is_mutating_use() {
                        // this is not strictly necessary as miri will also bail out
                        // For interior mutability we can't really catch this statically as that
                        // goes through raw pointers and intermediate temporaries, so miri has
                        // to catch this anyway
                        self.tcx.sess.span_err(
                            self.span,
                            "cannot mutate statics in the initializer of another static",
                        );
                    }
                    return;
                }
                unleash_miri!(self);

                if self.mode != Mode::Fn {
                    let mut err = struct_span_err!(self.tcx.sess, self.span, E0013,
                                                   "{}s cannot refer to statics, use \
                                                    a constant instead", self.mode);
                    if self.tcx.sess.teach(&err.get_code().unwrap()) {
                        err.note(
                            "Static and const variables can refer to other const variables. But a \
                             const variable cannot refer to a static variable."
                        );
                        err.help(
                            "To fix this, the value can be extracted as a const and then used."
                        );
                    }
                    err.emit()
                }
            }
            Place::Projection(ref proj) => {
                match proj.elem {
                    ProjectionElem::Deref => {
                        if context.is_mutating_use() {
                            // `not_const` errors out in const contexts
                            self.not_const()
                        }
                        let base_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                        match self.mode {
                            Mode::Fn => {},
                            _ => {
                                if let ty::RawPtr(_) = base_ty.sty {
                                    if !self.tcx.features().const_raw_ptr_deref {
                                        emit_feature_err(
                                            &self.tcx.sess.parse_sess, "const_raw_ptr_deref",
                                            self.span, GateIssue::Language,
                                            &format!(
                                                "dereferencing raw pointers in {}s is unstable",
                                                self.mode,
                                            ),
                                        );
                                    }
                                }
                            }
                        }
                    }

                    ProjectionElem::ConstantIndex {..} |
                    ProjectionElem::Subslice {..} |
                    ProjectionElem::Field(..) |
                    ProjectionElem::Index(_) => {
                        let base_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                        if let Some(def) = base_ty.ty_adt_def() {
                            if def.is_union() {
                                match self.mode {
                                    Mode::ConstFn => {
                                        if !self.tcx.features().const_fn_union {
                                            emit_feature_err(
                                                &self.tcx.sess.parse_sess, "const_fn_union",
                                                self.span, GateIssue::Language,
                                                "unions in const fn are unstable",
                                            );
                                        }
                                    },

                                    | Mode::Fn
                                    | Mode::Static
                                    | Mode::StaticMut
                                    | Mode::Const
                                    => {},
                                }
                            }
                        }
                    }

                    ProjectionElem::Downcast(..) => {
                        self.not_const()
                    }
                }
            }
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        debug!("visit_operand: operand={:?} location={:?}", operand, location);
        self.super_operand(operand, location);

        match *operand {
            Operand::Move(ref place) => {
                // Mark the consumed locals to indicate later drops are noops.
                if let Place::Local(local) = *place {
                    let slot = &mut self.local_qualif[local];
                    *slot = *slot - Qualif::NEEDS_DROP;
                }
            }
            Operand::Copy(_) |
            Operand::Constant(_) => {}
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        debug!("visit_rvalue: rvalue={:?} location={:?}", rvalue, location);

        // Check nested operands and places.
        if let Rvalue::Ref(region, kind, ref place) = *rvalue {
            // Special-case reborrows.
            let mut is_reborrow = false;
            if let Place::Projection(ref proj) = *place {
                if let ProjectionElem::Deref = proj.elem {
                    let base_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                    if let ty::Ref(..) = base_ty.sty {
                        is_reborrow = true;
                    }
                }
            }

            if is_reborrow {
                let ctx = match kind {
                    BorrowKind::Shared => PlaceContext::NonMutatingUse(
                        NonMutatingUseContext::SharedBorrow(region),
                    ),
                    BorrowKind::Shallow => PlaceContext::NonMutatingUse(
                        NonMutatingUseContext::ShallowBorrow(region),
                    ),
                    BorrowKind::Unique => PlaceContext::NonMutatingUse(
                        NonMutatingUseContext::UniqueBorrow(region),
                    ),
                    BorrowKind::Mut { .. } => PlaceContext::MutatingUse(
                        MutatingUseContext::Borrow(region),
                    ),
                };
                self.super_place(place, ctx, location);
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
            Rvalue::Cast(CastKind::ReifyFnPointer, ..) |
            Rvalue::Cast(CastKind::UnsafeFnPointer, ..) |
            Rvalue::Cast(CastKind::ClosureFnPointer, ..) |
            Rvalue::Cast(CastKind::Unsize, ..) |
            Rvalue::Discriminant(..) |
            Rvalue::Len(_) |
            Rvalue::Ref(..) |
            Rvalue::Aggregate(..) => {}

            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) => {
                let operand_ty = operand.ty(self.mir, self.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                match (cast_in, cast_out) {
                    (CastTy::Ptr(_), CastTy::Int(_)) |
                    (CastTy::FnPtr, CastTy::Int(_)) if self.mode != Mode::Fn => {
                        unleash_miri!(self);
                        if !self.tcx.features().const_raw_ptr_to_usize_cast {
                            // in const fn and constants require the feature gate
                            // FIXME: make it unsafe inside const fn and constants
                            emit_feature_err(
                                &self.tcx.sess.parse_sess, "const_raw_ptr_to_usize_cast",
                                self.span, GateIssue::Language,
                                &format!(
                                    "casting pointers to integers in {}s is unstable",
                                    self.mode,
                                ),
                            );
                        }
                    }
                    _ => {}
                }
            }

            Rvalue::BinaryOp(op, ref lhs, _) => {
                if let ty::RawPtr(_) | ty::FnPtr(..) = lhs.ty(self.mir, self.tcx).sty {
                    assert!(op == BinOp::Eq || op == BinOp::Ne ||
                            op == BinOp::Le || op == BinOp::Lt ||
                            op == BinOp::Ge || op == BinOp::Gt ||
                            op == BinOp::Offset);

                    unleash_miri!(self);
                    if self.mode != Mode::Fn && !self.tcx.features().const_compare_raw_pointers {
                        // require the feature gate inside constants and const fn
                        // FIXME: make it unsafe to use these operations
                        emit_feature_err(
                            &self.tcx.sess.parse_sess,
                            "const_compare_raw_pointers",
                            self.span,
                            GateIssue::Language,
                            &format!("comparing raw pointers inside {}", self.mode),
                        );
                    }
                }
            }

            Rvalue::NullaryOp(NullOp::Box, _) => {
                unleash_miri!(self);
                if self.mode != Mode::Fn {
                    let mut err = struct_span_err!(self.tcx.sess, self.span, E0010,
                                                   "allocations are not allowed in {}s", self.mode);
                    err.span_label(self.span, format!("allocation not allowed in {}s", self.mode));
                    if self.tcx.sess.teach(&err.get_code().unwrap()) {
                        err.note(
                            "The value of statics and constants must be known at compile time, \
                             and they live for the entire lifetime of a program. Creating a boxed \
                             value allocates memory on the heap at runtime, and therefore cannot \
                             be done at compile time."
                        );
                    }
                    err.emit();
                }
            }
        }
    }

    fn visit_terminator_kind(&mut self,
                             bb: BasicBlock,
                             kind: &TerminatorKind<'tcx>,
                             location: Location) {
        debug!("visit_terminator_kind: bb={:?} kind={:?} location={:?}", bb, kind, location);
        if let TerminatorKind::Call { ref func, ref args, ref destination, .. } = *kind {
            if let Some((ref dest, _)) = *destination {
                let ty = dest.ty(self.mir, self.tcx).to_ty(self.tcx);
                let qualif = self.qualifier().qualify_call(func, args, ty);
                self.assign(dest, qualif, location);
            }

            let fn_ty = func.ty(self.mir, self.tcx);
            let mut callee_def_id = None;
            let mut is_shuffle = false;
            match fn_ty.sty {
                ty::FnDef(def_id, _) => {
                    callee_def_id = Some(def_id);
                    match self.tcx.fn_sig(def_id).abi() {
                        Abi::RustIntrinsic |
                        Abi::PlatformIntrinsic => {
                            assert!(!self.tcx.is_const_fn(def_id));
                            match &self.tcx.item_name(def_id).as_str()[..] {
                                // special intrinsic that can be called diretly without an intrinsic
                                // feature gate needs a language feature gate
                                "transmute" => {
                                    // never promote transmute calls
                                    if self.mode != Mode::Fn {
                                        // const eval transmute calls only with the feature gate
                                        if !self.tcx.features().const_transmute {
                                            emit_feature_err(
                                                &self.tcx.sess.parse_sess, "const_transmute",
                                                self.span, GateIssue::Language,
                                                &format!("The use of std::mem::transmute() \
                                                is gated in {}s", self.mode));
                                        }
                                    }
                                }

                                name if name.starts_with("simd_shuffle") => {
                                    is_shuffle = true;
                                }

                                // no need to check feature gates, intrinsics are only callable
                                // from the libstd or with forever unstable feature gates
                                _ => {}
                            }
                        }
                        _ => {
                            // In normal functions no calls are feature-gated.
                            if self.mode != Mode::Fn {
                                let unleash_miri = self
                                    .tcx
                                    .sess
                                    .opts
                                    .debugging_opts
                                    .unleash_the_miri_inside_of_you;
                                if self.tcx.is_const_fn(def_id) || unleash_miri {
                                    // stable const fns or unstable const fns
                                    // with their feature gate active
                                    // FIXME(eddyb) move stability checks from `is_const_fn` here.
                                } else if self.qualifier().is_const_panic_fn(def_id) {
                                    // Check the const_panic feature gate.
                                    // FIXME: cannot allow this inside `allow_internal_unstable`
                                    // because that would make `panic!` insta stable in constants,
                                    // since the macro is marked with the attribute.
                                    if !self.tcx.features().const_panic {
                                        // Don't allow panics in constants without the feature gate.
                                        emit_feature_err(
                                            &self.tcx.sess.parse_sess,
                                            "const_panic",
                                            self.span,
                                            GateIssue::Language,
                                            &format!("panicking in {}s is unstable", self.mode),
                                        );
                                    }
                                } else if let Some(feature)
                                              = self.tcx.is_unstable_const_fn(def_id) {
                                    // Check `#[unstable]` const fns or `#[rustc_const_unstable]`
                                    // functions without the feature gate active in this crate in
                                    // order to report a better error message than the one below.
                                    if !self.span.allows_unstable(&feature.as_str()) {
                                        let mut err = self.tcx.sess.struct_span_err(self.span,
                                            &format!("`{}` is not yet stable as a const fn",
                                                    self.tcx.item_path_str(def_id)));
                                        if nightly_options::is_nightly_build() {
                                            help!(&mut err,
                                                  "add `#![feature({})]` to the \
                                                   crate attributes to enable",
                                                  feature);
                                        }
                                        err.emit();
                                    }
                                } else {
                                    let mut err = struct_span_err!(
                                        self.tcx.sess,
                                        self.span,
                                        E0015,
                                        "calls in {}s are limited to constant functions, \
                                         tuple structs and tuple variants",
                                        self.mode,
                                    );
                                    err.emit();
                                }
                            }
                        }
                    }
                }
                ty::FnPtr(_) => {
                    if self.mode != Mode::Fn {
                        let mut err = self.tcx.sess.struct_span_err(
                            self.span,
                            &format!("function pointers are not allowed in const fn"));
                        err.emit();
                    }
                }
                _ => {
                    self.not_const();
                }
            }

            if self.mode == Mode::Fn {
                let constant_args = callee_def_id.and_then(|id| {
                    args_required_const(self.tcx, id)
                }).unwrap_or_default();
                for (i, arg) in args.iter().enumerate() {
                    if !(is_shuffle && i == 2 || constant_args.contains(&i)) {
                        continue;
                    }

                    let candidate = Candidate::Argument { bb, index: i };
                    // Since the argument is required to be constant,
                    // we care about constness, not promotability.
                    // If we checked for promotability, we'd miss out on
                    // the results of function calls (which are never promoted
                    // in runtime code).
                    // This is not a problem, because the argument explicitly
                    // requests constness, in contrast to regular promotion
                    // which happens even without the user requesting it.
                    // We can error out with a hard error if the argument is not
                    // constant here.
                    let arg_qualif = self.qualifier().qualify_operand(arg);
                    if (arg_qualif - Qualif::NOT_PROMOTABLE).is_empty() {
                        debug!("visit_terminator_kind: candidate={:?}", candidate);
                        self.promotion_candidates.push(candidate);
                    } else {
                        if is_shuffle {
                            span_err!(self.tcx.sess, self.span, E0526,
                                      "shuffle indices are not constant");
                        } else {
                            self.tcx.sess.span_err(self.span,
                                &format!("argument {} is required to be a constant",
                                         i + 1));
                        }
                    }
                }
            }

            // Check callee and argument operands.
            self.visit_operand(func, location);
            for arg in args {
                self.visit_operand(arg, location);
            }
        } else if let TerminatorKind::Drop { location: ref place, .. } = *kind {
            self.super_terminator_kind(bb, kind, location);

            // Deny *any* live drops anywhere other than functions.
            if self.mode != Mode::Fn {
                unleash_miri!(self);
                // HACK(eddyb): emulate a bit of dataflow analysis,
                // conservatively, that drop elaboration will do.
                let needs_drop = if let Place::Local(local) = *place {
                    if self.local_qualif[local].contains(Qualif::NEEDS_DROP) {
                        Some(self.mir.local_decls[local].source_info.span)
                    } else {
                        None
                    }
                } else {
                    Some(self.span)
                };

                if let Some(span) = needs_drop {
                    // Double-check the type being dropped, to minimize false positives.
                    let ty = place.ty(self.mir, self.tcx).to_ty(self.tcx);
                    if ty.needs_drop(self.tcx, self.param_env) {
                        struct_span_err!(self.tcx.sess, span, E0493,
                                         "destructors cannot be evaluated at compile-time")
                            .span_label(span, format!("{}s cannot evaluate destructors",
                                                      self.mode))
                            .emit();
                    }
                }
            }
        } else {
            // Qualify any operands inside other terminators.
            self.super_terminator_kind(bb, kind, location);
        }
    }

    fn visit_assign(&mut self,
                    _: BasicBlock,
                    dest: &Place<'tcx>,
                    rvalue: &Rvalue<'tcx>,
                    location: Location) {
        debug!("visit_assign: dest={:?} rvalue={:?} location={:?}", dest, rvalue, location);
        let mut qualif = self.qualifier().qualify_rvalue(rvalue);

        if let Rvalue::Ref(_, kind, ref place) = *rvalue {
            // Getting `MUTABLE_INTERIOR` from `qualify_rvalue` means
            // the borrowed place is disallowed from being borrowed,
            // due to either a mutable borrow (with some exceptions),
            // or an shared borrow of a value with interior mutability.
            // Then `MUTABLE_INTERIOR` is replaced with `NOT_CONST`,
            // to avoid duplicate errors (e.g. from reborrowing).
            if qualif.contains(Qualif::MUTABLE_INTERIOR) {
                qualif = (qualif - Qualif::MUTABLE_INTERIOR) | Qualif::NOT_CONST;

                if self.mode != Mode::Fn {
                    if let BorrowKind::Mut { .. } = kind {
                        let mut err = struct_span_err!(self.tcx.sess,  self.span, E0017,
                                                       "references in {}s may only refer \
                                                        to immutable values", self.mode);
                        err.span_label(self.span, format!("{}s require immutable values",
                                                            self.mode));
                        if self.tcx.sess.teach(&err.get_code().unwrap()) {
                            err.note("References in statics and constants may only refer to \
                                      immutable values.\n\n\
                                      Statics are shared everywhere, and if they refer to \
                                      mutable data one might violate memory safety since \
                                      holding multiple mutable references to shared data is \
                                      not allowed.\n\n\
                                      If you really want global mutable state, try using \
                                      static mut or a global UnsafeCell.");
                        }
                        err.emit();
                    } else {
                        span_err!(self.tcx.sess, self.span, E0492,
                                  "cannot borrow a constant which may contain \
                                   interior mutability, create a static instead");
                    }
                }
            } else {
                // We might have a candidate for promotion.
                let candidate = Candidate::Ref(location);
                // We can only promote interior borrows of promotable temps.
                let mut place = place;
                while let Place::Projection(ref proj) = *place {
                    if proj.elem == ProjectionElem::Deref {
                        break;
                    }
                    place = &proj.base;
                }
                debug!("qualify_consts: promotion candidate: place={:?}", place);
                if let Place::Local(local) = *place {
                    if self.mir.local_kind(local) == LocalKind::Temp {
                        debug!("qualify_consts: promotion candidate: local={:?}", local);
                        let qualif = self.local_qualif[local];
                        // The borrowed place doesn't have `MUTABLE_INTERIOR`
                        // (from `qualify_rvalue`), so we can safely ignore
                        // `MUTABLE_INTERIOR` from the local's qualifications.
                        // This allows borrowing fields which don't have
                        // `MUTABLE_INTERIOR`, from a type that does, e.g.:
                        // `let _: &'static _ = &(Cell::new(1), 2).1;`
                        debug!("qualify_consts: promotion candidate: qualif={:?}", qualif);
                        if (qualif - Qualif::MUTABLE_INTERIOR).is_empty() {
                            debug!("qualify_consts: promotion candidate: {:?}", candidate);
                            self.promotion_candidates.push(candidate);
                        }
                    }
                }
            }
        }

        self.assign(dest, qualif, location);

        self.visit_rvalue(rvalue, location);
    }

    fn visit_source_info(&mut self, source_info: &SourceInfo) {
        debug!("visit_source_info: source_info={:?}", source_info);
        self.span = source_info.span;
    }

    fn visit_statement(&mut self, bb: BasicBlock, statement: &Statement<'tcx>, location: Location) {
        debug!("visit_statement: bb={:?} statement={:?} location={:?}", bb, statement, location);
        match statement.kind {
            StatementKind::Assign(..) => {
                self.super_statement(bb, statement, location);
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

    fn visit_terminator(&mut self,
                        bb: BasicBlock,
                        terminator: &Terminator<'tcx>,
                        location: Location) {
        debug!("visit_terminator: bb={:?} terminator={:?} location={:?}", bb, terminator, location);
        self.super_terminator(bb, terminator, location);
    }
}

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        mir_const_qualif,
        ..*providers
    };
}

fn mir_const_qualif<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              def_id: DefId)
                              -> (u8, Lrc<BitSet<Local>>) {
    // N.B., this `borrow()` is guaranteed to be valid (i.e., the value
    // cannot yet be stolen), because `mir_validated()`, which steals
    // from `mir_const(), forces this query to execute before
    // performing the steal.
    let mir = &tcx.mir_const(def_id).borrow();

    if mir.return_ty().references_error() {
        tcx.sess.delay_span_bug(mir.span, "mir_const_qualif: Mir had errors");
        return (Qualif::NOT_CONST.bits(), Lrc::new(BitSet::new_empty(0)));
    }

    let mut checker = Checker::new(tcx, def_id, mir, Mode::Const);
    let (qualif, promoted_temps) = checker.check_const();
    (qualif.bits(), promoted_temps)
}

pub struct QualifyAndPromoteConstants;

impl MirPass for QualifyAndPromoteConstants {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          src: MirSource<'tcx>,
                          mir: &mut Mir<'tcx>) {
        // There's not really any point in promoting errorful MIR.
        if mir.return_ty().references_error() {
            tcx.sess.delay_span_bug(mir.span, "QualifyAndPromoteConstants: Mir had errors");
            return;
        }

        if src.promoted.is_some() {
            return;
        }

        let def_id = src.def_id();
        let id = tcx.hir().as_local_hir_id(def_id).unwrap();
        let mut const_promoted_temps = None;
        let mode = match tcx.hir().body_owner_kind_by_hir_id(id) {
            hir::BodyOwnerKind::Closure => Mode::Fn,
            hir::BodyOwnerKind::Fn => {
                if tcx.is_const_fn(def_id) {
                    Mode::ConstFn
                } else {
                    Mode::Fn
                }
            }
            hir::BodyOwnerKind::Const => {
                const_promoted_temps = Some(tcx.mir_const_qualif(def_id).1);
                Mode::Const
            }
            hir::BodyOwnerKind::Static(hir::MutImmutable) => Mode::Static,
            hir::BodyOwnerKind::Static(hir::MutMutable) => Mode::StaticMut,
        };

        debug!("run_pass: mode={:?}", mode);
        if mode == Mode::Fn || mode == Mode::ConstFn {
            // This is ugly because Checker holds onto mir,
            // which can't be mutated until its scope ends.
            let (temps, candidates) = {
                let mut checker = Checker::new(tcx, def_id, mir, mode);
                if mode == Mode::ConstFn {
                    if tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you {
                        checker.check_const();
                    } else if tcx.is_min_const_fn(def_id) {
                        // enforce `min_const_fn` for stable const fns
                        use super::qualify_min_const_fn::is_min_const_fn;
                        if let Err((span, err)) = is_min_const_fn(tcx, def_id, mir) {
                            tcx.sess.span_err(span, &err);
                        } else {
                            // this should not produce any errors, but better safe than sorry
                            // FIXME(#53819)
                            checker.check_const();
                        }
                    } else {
                        // Enforce a constant-like CFG for `const fn`.
                        checker.check_const();
                    }
                } else {
                    while let Some((bb, data)) = checker.rpo.next() {
                        checker.visit_basic_block_data(bb, data);
                    }
                }

                (checker.temp_promotion_state, checker.promotion_candidates)
            };

            // Do the actual promotion, now that we know what's viable.
            promote_consts::promote_candidates(mir, tcx, temps, candidates);
        } else {
            if !mir.control_flow_destroyed.is_empty() {
                let mut locals = mir.vars_iter();
                if let Some(local) = locals.next() {
                    let span = mir.local_decls[local].source_info.span;
                    let mut error = tcx.sess.struct_span_err(
                        span,
                        &format!(
                            "new features like let bindings are not permitted in {}s \
                            which also use short circuiting operators",
                            mode,
                        ),
                    );
                    for (span, kind) in mir.control_flow_destroyed.iter() {
                        error.span_note(
                            *span,
                            &format!("use of {} here does not actually short circuit due to \
                            the const evaluator presently not being able to do control flow. \
                            See https://github.com/rust-lang/rust/issues/49146 for more \
                            information.", kind),
                        );
                    }
                    for local in locals {
                        let span = mir.local_decls[local].source_info.span;
                        error.span_note(
                            span,
                            "more locals defined here",
                        );
                    }
                    error.emit();
                }
            }
            let promoted_temps = if mode == Mode::Const {
                // Already computed by `mir_const_qualif`.
                const_promoted_temps.unwrap()
            } else {
                Checker::new(tcx, def_id, mir, mode).check_const().1
            };

            // In `const` and `static` everything without `StorageDead`
            // is `'static`, we don't have to create promoted MIR fragments,
            // just remove `Drop` and `StorageDead` on "promoted" locals.
            debug!("run_pass: promoted_temps={:?}", promoted_temps);
            for block in mir.basic_blocks_mut() {
                block.statements.retain(|statement| {
                    match statement.kind {
                        StatementKind::StorageDead(index) => {
                            !promoted_temps.contains(index)
                        }
                        _ => true
                    }
                });
                let terminator = block.terminator_mut();
                match terminator.kind {
                    TerminatorKind::Drop { location: Place::Local(index), target, .. } => {
                        if promoted_temps.contains(index) {
                            terminator.kind = TerminatorKind::Goto {
                                target,
                            };
                        }
                    }
                    _ => {}
                }
            }
        }

        // Statics must be Sync.
        if mode == Mode::Static {
            // `#[thread_local]` statics don't have to be `Sync`.
            for attr in &tcx.get_attrs(def_id)[..] {
                if attr.check_name("thread_local") {
                    return;
                }
            }
            let ty = mir.return_ty();
            tcx.infer_ctxt().enter(|infcx| {
                let param_env = ty::ParamEnv::empty();
                let cause = traits::ObligationCause::new(mir.span, id, traits::SharedStatic);
                let mut fulfillment_cx = traits::FulfillmentContext::new();
                fulfillment_cx.register_bound(&infcx,
                                              param_env,
                                              ty,
                                              tcx.require_lang_item(lang_items::SyncTraitLangItem),
                                              cause);
                if let Err(err) = fulfillment_cx.select_all_or_error(&infcx) {
                    infcx.report_fulfillment_errors(&err, None, false);
                }
            });
        }
    }
}

fn args_required_const(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> Option<FxHashSet<usize>> {
    let attrs = tcx.get_attrs(def_id);
    let attr = attrs.iter().find(|a| a.check_name("rustc_args_required_const"))?;
    let mut ret = FxHashSet::default();
    for meta in attr.meta_item_list()? {
        match meta.literal()?.node {
            LitKind::Int(a, _) => { ret.insert(a as usize); }
            _ => return None,
        }
    }
    Some(ret)
}
