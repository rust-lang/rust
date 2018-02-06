// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that qualifies constness of temporaries in constants,
//! static initializers and functions and also drives promotion.
//!
//! The Qualif flags below can be used to also provide better
//! diagnostics as to why a constant rvalue wasn't promoted.

use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_set::IdxSetBuf;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::middle::const_val::ConstVal;
use rustc::traits::{self, Reveal};
use rustc::ty::{self, TyCtxt, Ty, TypeFoldable};
use rustc::ty::cast::CastTy;
use rustc::ty::maps::Providers;
use rustc::mir::*;
use rustc::mir::traversal::ReversePostorder;
use rustc::mir::visit::{PlaceContext, Visitor};
use rustc::middle::lang_items;
use syntax::abi::Abi;
use syntax::attr;
use syntax::feature_gate::UnstableFeatures;
use syntax_pos::{Span, DUMMY_SP};

use std::fmt;
use std::rc::Rc;
use std::usize;

use transform::{MirPass, MirSource};
use super::promote_consts::{self, Candidate, TempState};

bitflags! {
    // Borrows of temporaries can be promoted only if
    // they have none of these qualifications, with
    // the exception of `STATIC_REF` (in statics only).
    struct Qualif: u8 {
        // Constant containing interior mutability (UnsafeCell).
        const MUTABLE_INTERIOR  = 1 << 0;

        // Constant containing an ADT that implements Drop.
        const NEEDS_DROP        = 1 << 1;

        // Function argument.
        const FN_ARGUMENT       = 1 << 2;

        // Static place or move from a static.
        const STATIC            = 1 << 3;

        // Reference to a static.
        const STATIC_REF        = 1 << 4;

        // Not constant at all - non-`const fn` calls, asm!,
        // pointer comparisons, ptr-to-int casts, etc.
        const NOT_CONST         = 1 << 5;

        // Refers to temporaries which cannot be promoted as
        // promote_consts decided they weren't simple enough.
        const NOT_PROMOTABLE    = 1 << 6;

        // Const items can only have MUTABLE_INTERIOR
        // and NOT_PROMOTABLE without producing an error.
        const CONST_ERROR       = !Qualif::MUTABLE_INTERIOR.bits &
                                  !Qualif::NOT_PROMOTABLE.bits;
    }
}

impl<'a, 'tcx> Qualif {
    /// Remove flags which are impossible for the given type.
    fn restrict(&mut self, ty: Ty<'tcx>,
                tcx: TyCtxt<'a, 'tcx, 'tcx>,
                param_env: ty::ParamEnv<'tcx>) {
        if ty.is_freeze(tcx, param_env, DUMMY_SP) {
            *self = *self - Qualif::MUTABLE_INTERIOR;
        }
        if !ty.needs_drop(tcx, param_env) {
            *self = *self - Qualif::NEEDS_DROP;
        }
    }
}

/// What kind of item we are in.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Mode {
    Const,
    Static,
    StaticMut,
    ConstFn,
    Fn
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Mode::Const => write!(f, "constant"),
            Mode::Static | Mode::StaticMut => write!(f, "static"),
            Mode::ConstFn => write!(f, "constant function"),
            Mode::Fn => write!(f, "function")
        }
    }
}

struct Qualifier<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    mode: Mode,
    span: Span,
    def_id: DefId,
    mir: &'a Mir<'tcx>,
    rpo: ReversePostorder<'a, 'tcx>,
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    temp_qualif: IndexVec<Local, Option<Qualif>>,
    return_qualif: Option<Qualif>,
    qualif: Qualif,
    const_fn_arg_vars: BitVector,
    temp_promotion_state: IndexVec<Local, TempState>,
    promotion_candidates: Vec<Candidate>
}

impl<'a, 'tcx> Qualifier<'a, 'tcx, 'tcx> {
    fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
           def_id: DefId,
           mir: &'a Mir<'tcx>,
           mode: Mode)
           -> Qualifier<'a, 'tcx, 'tcx> {
        let mut rpo = traversal::reverse_postorder(mir);
        let temps = promote_consts::collect_temps(mir, &mut rpo);
        rpo.reset();

        let param_env = tcx.param_env(def_id);

        let mut temp_qualif = IndexVec::from_elem(None, &mir.local_decls);
        for arg in mir.args_iter() {
            let mut qualif = Qualif::NEEDS_DROP;
            qualif.restrict(mir.local_decls[arg].ty, tcx, param_env);
            temp_qualif[arg] = Some(qualif);
        }

        Qualifier {
            mode,
            span: mir.span,
            def_id,
            mir,
            rpo,
            tcx,
            param_env,
            temp_qualif,
            return_qualif: None,
            qualif: Qualif::empty(),
            const_fn_arg_vars: BitVector::new(mir.local_decls.len()),
            temp_promotion_state: temps,
            promotion_candidates: vec![]
        }
    }

    // FIXME(eddyb) we could split the errors into meaningful
    // categories, but enabling full miri would make that
    // slightly pointless (even with feature-gating).
    fn not_const(&mut self) {
        self.add(Qualif::NOT_CONST);
        if self.mode != Mode::Fn {
            span_err!(self.tcx.sess, self.span, E0019,
                      "{} contains unimplemented expression type", self.mode);
        }
    }

    /// Error about extra statements in a constant.
    fn statement_like(&mut self) {
        self.add(Qualif::NOT_CONST);
        if self.mode != Mode::Fn {
            span_err!(self.tcx.sess, self.span, E0016,
                      "blocks in {}s are limited to items and tail expressions",
                      self.mode);
        }
    }

    /// Add the given qualification to self.qualif.
    fn add(&mut self, qualif: Qualif) {
        self.qualif = self.qualif | qualif;
    }

    /// Add the given type's qualification to self.qualif.
    fn add_type(&mut self, ty: Ty<'tcx>) {
        self.add(Qualif::MUTABLE_INTERIOR | Qualif::NEEDS_DROP);
        self.qualif.restrict(ty, self.tcx, self.param_env);
    }

    /// Within the provided closure, self.qualif will start
    /// out empty, and its value after the closure returns will
    /// be combined with the value before the call to nest.
    fn nest<F: FnOnce(&mut Self)>(&mut self, f: F) {
        let original = self.qualif;
        self.qualif = Qualif::empty();
        f(self);
        self.add(original);
    }

    /// Check if a Local with the current qualifications is promotable.
    fn can_promote(&mut self) -> bool {
        // References to statics are allowed, but only in other statics.
        if self.mode == Mode::Static || self.mode == Mode::StaticMut {
            (self.qualif - Qualif::STATIC_REF).is_empty()
        } else {
            self.qualif.is_empty()
        }
    }

    /// Check if a Place with the current qualifications could
    /// be consumed, by either an operand or a Deref projection.
    fn try_consume(&mut self) -> bool {
        if self.qualif.intersects(Qualif::STATIC) && self.mode != Mode::Fn {
            let msg = if self.mode == Mode::Static ||
                         self.mode == Mode::StaticMut {
                "cannot refer to other statics by value, use the \
                 address-of operator or a constant instead"
            } else {
                "cannot refer to statics by value, use a constant instead"
            };
            struct_span_err!(self.tcx.sess, self.span, E0394, "{}", msg)
                .span_label(self.span, "referring to another static by value")
                .note("use the address-of operator or a constant instead")
                .emit();

            // Replace STATIC with NOT_CONST to avoid further errors.
            self.qualif = self.qualif - Qualif::STATIC;
            self.add(Qualif::NOT_CONST);

            false
        } else {
            true
        }
    }

    /// Assign the current qualification to the given destination.
    fn assign(&mut self, dest: &Place<'tcx>, location: Location) {
        let qualif = self.qualif;
        let span = self.span;
        let store = |slot: &mut Option<Qualif>| {
            if slot.is_some() {
                span_bug!(span, "multiple assignments to {:?}", dest);
            }
            *slot = Some(qualif);
        };

        // Only handle promotable temps in non-const functions.
        if self.mode == Mode::Fn {
            if let Place::Local(index) = *dest {
                if self.mir.local_kind(index) == LocalKind::Temp
                && self.temp_promotion_state[index].is_promotable() {
                    debug!("store to promotable temp {:?}", index);
                    store(&mut self.temp_qualif[index]);
                }
            }
            return;
        }

        match *dest {
            Place::Local(index) if self.mir.local_kind(index) == LocalKind::Temp => {
                debug!("store to temp {:?}", index);
                store(&mut self.temp_qualif[index])
            }
            Place::Local(index) if self.mir.local_kind(index) == LocalKind::ReturnPointer => {
                debug!("store to return place {:?}", index);
                store(&mut self.return_qualif)
            }

            Place::Projection(box Projection {
                base: Place::Local(index),
                elem: ProjectionElem::Deref
            }) if self.mir.local_kind(index) == LocalKind::Temp
               && self.mir.local_decls[index].ty.is_box()
               && self.temp_qualif[index].map_or(false, |qualif| {
                    qualif.intersects(Qualif::NOT_CONST)
               }) => {
                // Part of `box expr`, we should've errored
                // already for the Box allocation Rvalue.
            }

            // This must be an explicit assignment.
            _ => {
                // Catch more errors in the destination.
                self.visit_place(dest, PlaceContext::Store, location);
                self.statement_like();
            }
        }
    }

    /// Qualify a whole const, static initializer or const fn.
    fn qualify_const(&mut self) -> (Qualif, Rc<IdxSetBuf<Local>>) {
        debug!("qualifying {} {:?}", self.mode, self.def_id);

        let mir = self.mir;

        let mut seen_blocks = BitVector::new(mir.basic_blocks().len());
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
                TerminatorKind::FalseEdges { .. } => None,

                TerminatorKind::Return => {
                    // Check for unused values. This usually means
                    // there are extra statements in the AST.
                    for temp in mir.temps_iter() {
                        if self.temp_qualif[temp].is_none() {
                            continue;
                        }

                        let state = self.temp_promotion_state[temp];
                        if let TempState::Defined { location, uses: 0 } = state {
                            let data = &mir[location.block];
                            let stmt_idx = location.statement_index;

                            // Get the span for the initialization.
                            let source_info = if stmt_idx < data.statements.len() {
                                data.statements[stmt_idx].source_info
                            } else {
                                data.terminator().source_info
                            };
                            self.span = source_info.span;

                            // Treat this as a statement in the AST.
                            self.statement_like();
                        }
                    }

                    // Make sure there are no extra unassigned variables.
                    self.qualif = Qualif::NOT_CONST;
                    for index in mir.vars_iter() {
                        if !self.const_fn_arg_vars.contains(index.index()) {
                            debug!("unassigned variable {:?}", index);
                            self.assign(&Place::Local(index), Location {
                                block: bb,
                                statement_index: usize::MAX,
                            });
                        }
                    }

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

        self.qualif = self.return_qualif.unwrap_or(Qualif::NOT_CONST);

        // Account for errors in consts by using the
        // conservative type qualification instead.
        if self.qualif.intersects(Qualif::CONST_ERROR) {
            self.qualif = Qualif::empty();
            let return_ty = mir.return_ty();
            self.add_type(return_ty);
        }


        // Collect all the temps we need to promote.
        let mut promoted_temps = IdxSetBuf::new_empty(self.temp_promotion_state.len());

        for candidate in &self.promotion_candidates {
            match *candidate {
                Candidate::Ref(Location { block: bb, statement_index: stmt_idx }) => {
                    match self.mir[bb].statements[stmt_idx].kind {
                        StatementKind::Assign(_, Rvalue::Ref(_, _, Place::Local(index))) => {
                            promoted_temps.add(&index);
                        }
                        _ => {}
                    }
                }
                Candidate::ShuffleIndices(_) => {}
            }
        }

        (self.qualif, Rc::new(promoted_temps))
    }
}

/// Accumulates an Rvalue or Call's effects in self.qualif.
/// For functions (constant or not), it also records
/// candidates for promotion in promotion_candidates.
impl<'a, 'tcx> Visitor<'tcx> for Qualifier<'a, 'tcx, 'tcx> {
    fn visit_local(&mut self,
                   &local: &Local,
                   _: PlaceContext<'tcx>,
                   _: Location) {
        let kind = self.mir.local_kind(local);
        match kind {
            LocalKind::ReturnPointer => {
                self.not_const();
            }
            LocalKind::Var => {
                self.add(Qualif::NOT_CONST);
            }
            LocalKind::Arg |
            LocalKind::Temp => {
                if let LocalKind::Arg = kind {
                    self.add(Qualif::FN_ARGUMENT);
                }

                if !self.temp_promotion_state[local].is_promotable() {
                    self.add(Qualif::NOT_PROMOTABLE);
                }

                if let Some(qualif) = self.temp_qualif[local] {
                    self.add(qualif);
                } else {
                    self.not_const();
                }
            }
        }
    }

    fn visit_place(&mut self,
                    place: &Place<'tcx>,
                    context: PlaceContext<'tcx>,
                    location: Location) {
        match *place {
            Place::Local(ref local) => self.visit_local(local, context, location),
            Place::Static(ref global) => {
                self.add(Qualif::STATIC);

                if self.mode != Mode::Fn {
                    for attr in &self.tcx.get_attrs(global.def_id)[..] {
                        if attr.check_name("thread_local") {
                            span_err!(self.tcx.sess, self.span, E0625,
                                      "thread-local statics cannot be \
                                       accessed at compile-time");
                            self.add(Qualif::NOT_CONST);
                            return;
                        }
                    }
                }

                if self.mode == Mode::Const || self.mode == Mode::ConstFn {
                    span_err!(self.tcx.sess, self.span, E0013,
                              "{}s cannot refer to statics, use \
                               a constant instead", self.mode);
                }
            }
            Place::Projection(ref proj) => {
                self.nest(|this| {
                    this.super_place(place, context, location);
                    match proj.elem {
                        ProjectionElem::Deref => {
                            if !this.try_consume() {
                                return;
                            }

                            if this.qualif.intersects(Qualif::STATIC_REF) {
                                this.qualif = this.qualif - Qualif::STATIC_REF;
                                this.add(Qualif::STATIC);
                            }

                            let base_ty = proj.base.ty(this.mir, this.tcx).to_ty(this.tcx);
                            if let ty::TyRawPtr(_) = base_ty.sty {
                                this.add(Qualif::NOT_CONST);
                                if this.mode != Mode::Fn {
                                    struct_span_err!(this.tcx.sess,
                                        this.span, E0396,
                                        "raw pointers cannot be dereferenced in {}s",
                                        this.mode)
                                    .span_label(this.span,
                                        "dereference of raw pointer in constant")
                                    .emit();
                                }
                            }
                        }

                        ProjectionElem::Field(..) |
                        ProjectionElem::Index(_) => {
                            if this.mode != Mode::Fn &&
                               this.qualif.intersects(Qualif::STATIC) {
                                span_err!(this.tcx.sess, this.span, E0494,
                                          "cannot refer to the interior of another \
                                           static, use a constant instead");
                            }
                            let ty = place.ty(this.mir, this.tcx).to_ty(this.tcx);
                            this.qualif.restrict(ty, this.tcx, this.param_env);
                        }

                        ProjectionElem::ConstantIndex {..} |
                        ProjectionElem::Subslice {..} |
                        ProjectionElem::Downcast(..) => {
                            this.not_const()
                        }
                    }
                });
            }
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        match *operand {
            Operand::Copy(_) |
            Operand::Move(_) => {
                self.nest(|this| {
                    this.super_operand(operand, location);
                    this.try_consume();
                });

                // Mark the consumed locals to indicate later drops are noops.
                if let Operand::Move(Place::Local(local)) = *operand {
                    self.temp_qualif[local] = self.temp_qualif[local].map(|q|
                        q - Qualif::NEEDS_DROP
                    );
                }
            }
            Operand::Constant(ref constant) => {
                if let Literal::Value {
                    value: &ty::Const { val: ConstVal::Unevaluated(def_id, _), ty }
                } = constant.literal {
                    // Don't peek inside trait associated constants.
                    if self.tcx.trait_of_item(def_id).is_some() {
                        self.add_type(ty);
                    } else {
                        let (bits, _) = self.tcx.at(constant.span).mir_const_qualif(def_id);

                        let qualif = Qualif::from_bits(bits).expect("invalid mir_const_qualif");
                        self.add(qualif);

                        // Just in case the type is more specific than
                        // the definition, e.g. impl associated const
                        // with type parameters, take it into account.
                        self.qualif.restrict(ty, self.tcx, self.param_env);
                    }
                }
            }
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        // Recurse through operands and places.
        self.super_rvalue(rvalue, location);

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
            Rvalue::Discriminant(..) => {}

            Rvalue::Len(_) => {
                // Static places in consts would have errored already,
                // don't treat length checks as reads from statics.
                self.qualif = self.qualif - Qualif::STATIC;
            }

            Rvalue::Ref(_, kind, ref place) => {
                // Static places in consts would have errored already,
                // only keep track of references to them here.
                if self.qualif.intersects(Qualif::STATIC) {
                    self.qualif = self.qualif - Qualif::STATIC;
                    self.add(Qualif::STATIC_REF);
                }

                let ty = place.ty(self.mir, self.tcx).to_ty(self.tcx);
                if kind == BorrowKind::Mut {
                    // In theory, any zero-sized value could be borrowed
                    // mutably without consequences. However, only &mut []
                    // is allowed right now, and only in functions.
                    let allow = if self.mode == Mode::StaticMut {
                        // Inside a `static mut`, &mut [...] is also allowed.
                        match ty.sty {
                            ty::TyArray(..) | ty::TySlice(_) => true,
                            _ => false
                        }
                    } else if let ty::TyArray(_, len) = ty.sty {
                        len.val.to_const_int().unwrap().to_u64().unwrap() == 0 &&
                            self.mode == Mode::Fn
                    } else {
                        false
                    };

                    if !allow {
                        self.add(Qualif::NOT_CONST);
                        if self.mode != Mode::Fn {
                            struct_span_err!(self.tcx.sess,  self.span, E0017,
                                             "references in {}s may only refer \
                                              to immutable values", self.mode)
                                .span_label(self.span, format!("{}s require immutable values",
                                                                self.mode))
                                .emit();
                        }
                    }
                } else {
                    // Constants cannot be borrowed if they contain interior mutability as
                    // it means that our "silent insertion of statics" could change
                    // initializer values (very bad).
                    if self.qualif.intersects(Qualif::MUTABLE_INTERIOR) {
                        // Replace MUTABLE_INTERIOR with NOT_CONST to avoid
                        // duplicate errors (from reborrowing, for example).
                        self.qualif = self.qualif - Qualif::MUTABLE_INTERIOR;
                        self.add(Qualif::NOT_CONST);
                        if self.mode != Mode::Fn {
                            span_err!(self.tcx.sess, self.span, E0492,
                                      "cannot borrow a constant which may contain \
                                       interior mutability, create a static instead");
                        }
                    }
                }

                // We might have a candidate for promotion.
                let candidate = Candidate::Ref(location);
                if self.can_promote() {
                    // We can only promote direct borrows of temps.
                    if let Place::Local(local) = *place {
                        if self.mir.local_kind(local) == LocalKind::Temp {
                            self.promotion_candidates.push(candidate);
                        }
                    }
                }
            }

            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) => {
                let operand_ty = operand.ty(self.mir, self.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                match (cast_in, cast_out) {
                    (CastTy::Ptr(_), CastTy::Int(_)) |
                    (CastTy::FnPtr, CastTy::Int(_)) => {
                        self.add(Qualif::NOT_CONST);
                        if self.mode != Mode::Fn {
                            span_err!(self.tcx.sess, self.span, E0018,
                                      "raw pointers cannot be cast to integers in {}s",
                                      self.mode);
                        }
                    }
                    _ => {}
                }
            }

            Rvalue::BinaryOp(op, ref lhs, _) => {
                if let ty::TyRawPtr(_) = lhs.ty(self.mir, self.tcx).sty {
                    assert!(op == BinOp::Eq || op == BinOp::Ne ||
                            op == BinOp::Le || op == BinOp::Lt ||
                            op == BinOp::Ge || op == BinOp::Gt ||
                            op == BinOp::Offset);

                    self.add(Qualif::NOT_CONST);
                    if self.mode != Mode::Fn {
                        struct_span_err!(
                            self.tcx.sess, self.span, E0395,
                            "raw pointers cannot be compared in {}s",
                            self.mode)
                        .span_label(
                            self.span,
                            "comparing raw pointers in static")
                        .emit();
                    }
                }
            }

            Rvalue::NullaryOp(NullOp::Box, _) => {
                self.add(Qualif::NOT_CONST);
                if self.mode != Mode::Fn {
                    struct_span_err!(self.tcx.sess, self.span, E0010,
                                     "allocations are not allowed in {}s", self.mode)
                        .span_label(self.span, format!("allocation not allowed in {}s", self.mode))
                        .emit();
                }
            }

            Rvalue::Aggregate(ref kind, _) => {
                if let AggregateKind::Adt(def, ..) = **kind {
                    if def.has_dtor(self.tcx) {
                        self.add(Qualif::NEEDS_DROP);
                    }

                    if Some(def.did) == self.tcx.lang_items().unsafe_cell_type() {
                        let ty = rvalue.ty(self.mir, self.tcx);
                        self.add_type(ty);
                        assert!(self.qualif.intersects(Qualif::MUTABLE_INTERIOR));
                    }
                }
            }
        }
    }

    fn visit_terminator_kind(&mut self,
                             bb: BasicBlock,
                             kind: &TerminatorKind<'tcx>,
                             location: Location) {
        if let TerminatorKind::Call { ref func, ref args, ref destination, .. } = *kind {
            self.visit_operand(func, location);

            let fn_ty = func.ty(self.mir, self.tcx);
            let (mut is_shuffle, mut is_const_fn) = (false, None);
            if let ty::TyFnDef(def_id, _) = fn_ty.sty {
                match self.tcx.fn_sig(def_id).abi() {
                    Abi::RustIntrinsic |
                    Abi::PlatformIntrinsic => {
                        assert!(!self.tcx.is_const_fn(def_id));
                        match &self.tcx.item_name(def_id)[..] {
                            "size_of" | "min_align_of" => is_const_fn = Some(def_id),

                            name if name.starts_with("simd_shuffle") => {
                                is_shuffle = true;
                            }

                            _ => {}
                        }
                    }
                    _ => {
                        if self.tcx.is_const_fn(def_id) {
                            is_const_fn = Some(def_id);
                        }
                    }
                }
            }

            for (i, arg) in args.iter().enumerate() {
                self.nest(|this| {
                    this.visit_operand(arg, location);
                    if is_shuffle && i == 2 && this.mode == Mode::Fn {
                        let candidate = Candidate::ShuffleIndices(bb);
                        if this.can_promote() {
                            this.promotion_candidates.push(candidate);
                        } else {
                            span_err!(this.tcx.sess, this.span, E0526,
                                      "shuffle indices are not constant");
                        }
                    }
                });
            }

            // Const fn calls.
            if let Some(def_id) = is_const_fn {
                // find corresponding rustc_const_unstable feature
                if let Some(&attr::Stability {
                    rustc_const_unstable: Some(attr::RustcConstUnstable {
                        feature: ref feature_name
                    }),
                .. }) = self.tcx.lookup_stability(def_id) {

                    // We are in a const or static initializer,
                    if self.mode != Mode::Fn &&

                        // feature-gate is not enabled,
                        !self.tcx.sess.features.borrow()
                            .declared_lib_features
                            .iter()
                            .any(|&(ref sym, _)| sym == feature_name) &&

                        // this doesn't come from a crate with the feature-gate enabled,
                        self.def_id.is_local() &&

                        // this doesn't come from a macro that has #[allow_internal_unstable]
                        !self.span.allows_unstable()
                    {
                        let mut err = self.tcx.sess.struct_span_err(self.span,
                            &format!("`{}` is not yet stable as a const fn",
                                     self.tcx.item_path_str(def_id)));
                        help!(&mut err,
                              "in Nightly builds, add `#![feature({})]` \
                               to the crate attributes to enable",
                              feature_name);
                        err.emit();
                    }
                }
            } else {
                self.qualif = Qualif::NOT_CONST;
                if self.mode != Mode::Fn {
                    // FIXME(#24111) Remove this check when const fn stabilizes
                    let (msg, note) = if let UnstableFeatures::Disallow =
                            self.tcx.sess.opts.unstable_features {
                        (format!("calls in {}s are limited to \
                                  struct and enum constructors",
                                 self.mode),
                         Some("a limited form of compile-time function \
                               evaluation is available on a nightly \
                               compiler via `const fn`"))
                    } else {
                        (format!("calls in {}s are limited \
                                  to constant functions, \
                                  struct and enum constructors",
                                 self.mode),
                         None)
                    };
                    let mut err = struct_span_err!(self.tcx.sess, self.span, E0015, "{}", msg);
                    if let Some(note) = note {
                        err.span_note(self.span, note);
                    }
                    err.emit();
                }
            }

            if let Some((ref dest, _)) = *destination {
                // Avoid propagating irrelevant callee/argument qualifications.
                if self.qualif.intersects(Qualif::CONST_ERROR) {
                    self.qualif = Qualif::NOT_CONST;
                } else {
                    // Be conservative about the returned value of a const fn.
                    let tcx = self.tcx;
                    let ty = dest.ty(self.mir, tcx).to_ty(tcx);
                    self.qualif = Qualif::empty();
                    self.add_type(ty);
                }
                self.assign(dest, location);
            }
        } else if let TerminatorKind::Drop { location: ref place, .. } = *kind {
            self.super_terminator_kind(bb, kind, location);

            // Deny *any* live drops anywhere other than functions.
            if self.mode != Mode::Fn {
                // HACK(eddyb) Emulate a bit of dataflow analysis,
                // conservatively, that drop elaboration will do.
                let needs_drop = if let Place::Local(local) = *place {
                    if self.temp_qualif[local].map_or(true, |q| q.intersects(Qualif::NEEDS_DROP)) {
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
        self.visit_rvalue(rvalue, location);

        // Check the allowed const fn argument forms.
        if let (Mode::ConstFn, &Place::Local(index)) = (self.mode, dest) {
            if self.mir.local_kind(index) == LocalKind::Var &&
               self.const_fn_arg_vars.insert(index.index()) {

                // Direct use of an argument is permitted.
                match *rvalue {
                    Rvalue::Use(Operand::Copy(Place::Local(local))) |
                    Rvalue::Use(Operand::Move(Place::Local(local))) => {
                        if self.mir.local_kind(local) == LocalKind::Arg {
                            return;
                        }
                    }
                    _ => {}
                }

                // Avoid a generic error for other uses of arguments.
                if self.qualif.intersects(Qualif::FN_ARGUMENT) {
                    let decl = &self.mir.local_decls[index];
                    span_err!(self.tcx.sess, decl.source_info.span, E0022,
                              "arguments of constant functions can only \
                               be immutable by-value bindings");
                    return;
                }
            }
        }

        self.assign(dest, location);
    }

    fn visit_source_info(&mut self, source_info: &SourceInfo) {
        self.span = source_info.span;
    }

    fn visit_statement(&mut self, bb: BasicBlock, statement: &Statement<'tcx>, location: Location) {
        self.nest(|this| {
            this.visit_source_info(&statement.source_info);
            match statement.kind {
                StatementKind::Assign(ref place, ref rvalue) => {
                    this.visit_assign(bb, place, rvalue, location);
                }
                StatementKind::SetDiscriminant { .. } |
                StatementKind::StorageLive(_) |
                StatementKind::StorageDead(_) |
                StatementKind::InlineAsm {..} |
                StatementKind::EndRegion(_) |
                StatementKind::Validate(..) |
                StatementKind::Nop => {}
            }
        });
    }

    fn visit_terminator(&mut self,
                        bb: BasicBlock,
                        terminator: &Terminator<'tcx>,
                        location: Location) {
        self.nest(|this| this.super_terminator(bb, terminator, location));
    }
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        mir_const_qualif,
        ..*providers
    };
}

fn mir_const_qualif<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              def_id: DefId)
                              -> (u8, Rc<IdxSetBuf<Local>>) {
    // NB: This `borrow()` is guaranteed to be valid (i.e., the value
    // cannot yet be stolen), because `mir_validated()`, which steals
    // from `mir_const(), forces this query to execute before
    // performing the steal.
    let mir = &tcx.mir_const(def_id).borrow();

    if mir.return_ty().references_error() {
        tcx.sess.delay_span_bug(mir.span, "mir_const_qualif: Mir had errors");
        return (Qualif::NOT_CONST.bits(), Rc::new(IdxSetBuf::new_empty(0)));
    }

    let mut qualifier = Qualifier::new(tcx, def_id, mir, Mode::Const);
    let (qualif, promoted_temps) = qualifier.qualify_const();
    (qualif.bits(), promoted_temps)
}

pub struct QualifyAndPromoteConstants;

impl MirPass for QualifyAndPromoteConstants {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          src: MirSource,
                          mir: &mut Mir<'tcx>) {
        // There's not really any point in promoting errorful MIR.
        if mir.return_ty().references_error() {
            tcx.sess.delay_span_bug(mir.span, "QualifyAndPromoteConstants: Mir had errors");
            return;
        }

        if src.promoted.is_some() {
            return;
        }

        let def_id = src.def_id;
        let id = tcx.hir.as_local_node_id(def_id).unwrap();
        let mut const_promoted_temps = None;
        let mode = match tcx.hir.body_owner_kind(id) {
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

        if mode == Mode::Fn || mode == Mode::ConstFn {
            // This is ugly because Qualifier holds onto mir,
            // which can't be mutated until its scope ends.
            let (temps, candidates) = {
                let mut qualifier = Qualifier::new(tcx, def_id, mir, mode);
                if mode == Mode::ConstFn {
                    // Enforce a constant-like CFG for `const fn`.
                    qualifier.qualify_const();
                } else {
                    while let Some((bb, data)) = qualifier.rpo.next() {
                        qualifier.visit_basic_block_data(bb, data);
                    }
                }

                (qualifier.temp_promotion_state, qualifier.promotion_candidates)
            };

            // Do the actual promotion, now that we know what's viable.
            promote_consts::promote_candidates(mir, tcx, temps, candidates);
        } else {
            let promoted_temps = if mode == Mode::Const {
                // Already computed by `mir_const_qualif`.
                const_promoted_temps.unwrap()
            } else {
                Qualifier::new(tcx, def_id, mir, mode).qualify_const().1
            };

            // In `const` and `static` everything without `StorageDead`
            // is `'static`, we don't have to create promoted MIR fragments,
            // just remove `Drop` and `StorageDead` on "promoted" locals.
            for block in mir.basic_blocks_mut() {
                block.statements.retain(|statement| {
                    match statement.kind {
                        StatementKind::StorageDead(index) => {
                            !promoted_temps.contains(&index)
                        }
                        _ => true
                    }
                });
                let terminator = block.terminator_mut();
                match terminator.kind {
                    TerminatorKind::Drop { location: Place::Local(index), target, .. } => {
                        if promoted_temps.contains(&index) {
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
                let param_env = ty::ParamEnv::empty(Reveal::UserFacing);
                let cause = traits::ObligationCause::new(mir.span, id, traits::SharedStatic);
                let mut fulfillment_cx = traits::FulfillmentContext::new();
                fulfillment_cx.register_bound(&infcx,
                                              param_env,
                                              ty,
                                              tcx.require_lang_item(lang_items::SyncTraitLangItem),
                                              cause);
                if let Err(err) = fulfillment_cx.select_all_or_error(&infcx) {
                    infcx.report_fulfillment_errors(&err, None);
                }
            });
        }
    }
}
