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
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::FnKind;
use rustc::hir::map::blocks::FnLikeNode;
use rustc::traits::{self, ProjectionMode};
use rustc::ty::{self, TyCtxt, Ty};
use rustc::ty::cast::CastTy;
use rustc::mir::repr::*;
use rustc::mir::mir_map::MirMap;
use rustc::mir::traversal::{self, ReversePostorder};
use rustc::mir::transform::{Pass, MirMapPass, MirPassHook, MirSource};
use rustc::mir::visit::{LvalueContext, Visitor};
use rustc::util::nodemap::DefIdMap;
use syntax::abi::Abi;
use syntax::feature_gate::UnstableFeatures;
use syntax_pos::Span;

use std::collections::hash_map::Entry;
use std::fmt;

use build::Location;

use super::promote_consts::{self, Candidate, TempState};

bitflags! {
    flags Qualif: u8 {
        // Const item's qualification while recursing.
        // Recursive consts are an error.
        const RECURSIVE         = 1 << 0,

        // Constant containing interior mutability (UnsafeCell).
        const MUTABLE_INTERIOR  = 1 << 1,

        // Constant containing an ADT that implements Drop.
        const NEEDS_DROP        = 1 << 2,

        // Function argument.
        const FN_ARGUMENT       = 1 << 3,

        // Static lvalue or move from a static.
        const STATIC            = 1 << 4,

        // Reference to a static.
        const STATIC_REF        = 1 << 5,

        // Not constant at all - non-`const fn` calls, asm!,
        // pointer comparisons, ptr-to-int casts, etc.
        const NOT_CONST         = 1 << 6,

        // Refers to temporaries which cannot be promoted as
        // promote_consts decided they weren't simple enough.
        const NOT_PROMOTABLE    = 1 << 7,

        // Borrows of temporaries can be promoted only
        // if they have none of the above qualifications.
        const NEVER_PROMOTE     = !0,

        // Const items can only have MUTABLE_INTERIOR
        // and NOT_PROMOTABLE without producing an error.
        const CONST_ERROR       = !Qualif::MUTABLE_INTERIOR.bits &
                                  !Qualif::NOT_PROMOTABLE.bits
    }
}

impl<'a, 'tcx> Qualif {
    /// Remove flags which are impossible for the given type.
    fn restrict(&mut self, ty: Ty<'tcx>,
                tcx: TyCtxt<'a, 'tcx, 'tcx>,
                param_env: &ty::ParameterEnvironment<'tcx>) {
        if !ty.type_contents(tcx).interior_unsafe() {
            *self = *self - Qualif::MUTABLE_INTERIOR;
        }
        if !tcx.type_needs_drop_given_env(ty, param_env) {
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

fn is_const_fn(tcx: TyCtxt, def_id: DefId) -> bool {
    if let Some(node_id) = tcx.map.as_local_node_id(def_id) {
        let fn_like = FnLikeNode::from_node(tcx.map.get(node_id));
        match fn_like.map(|f| f.kind()) {
            Some(FnKind::ItemFn(_, _, _, c, _, _, _)) => {
                c == hir::Constness::Const
            }
            Some(FnKind::Method(_, m, _, _)) => {
                m.constness == hir::Constness::Const
            }
            _ => false
        }
    } else {
        tcx.sess.cstore.is_const_fn(def_id)
    }
}

struct Qualifier<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    mode: Mode,
    span: Span,
    def_id: DefId,
    mir: &'a Mir<'tcx>,
    rpo: ReversePostorder<'a, 'tcx>,
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    param_env: ty::ParameterEnvironment<'tcx>,
    qualif_map: &'a mut DefIdMap<Qualif>,
    mir_map: Option<&'a MirMap<'tcx>>,
    temp_qualif: IndexVec<Temp, Option<Qualif>>,
    return_qualif: Option<Qualif>,
    qualif: Qualif,
    const_fn_arg_vars: BitVector,
    location: Location,
    temp_promotion_state: IndexVec<Temp, TempState>,
    promotion_candidates: Vec<Candidate>
}

impl<'a, 'tcx> Qualifier<'a, 'tcx, 'tcx> {
    fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
           param_env: ty::ParameterEnvironment<'tcx>,
           qualif_map: &'a mut DefIdMap<Qualif>,
           mir_map: Option<&'a MirMap<'tcx>>,
           def_id: DefId,
           mir: &'a Mir<'tcx>,
           mode: Mode)
           -> Qualifier<'a, 'tcx, 'tcx> {
        let mut rpo = traversal::reverse_postorder(mir);
        let temps = promote_consts::collect_temps(mir, &mut rpo);
        rpo.reset();
        Qualifier {
            mode: mode,
            span: mir.span,
            def_id: def_id,
            mir: mir,
            rpo: rpo,
            tcx: tcx,
            param_env: param_env,
            qualif_map: qualif_map,
            mir_map: mir_map,
            temp_qualif: IndexVec::from_elem(None, &mir.temp_decls),
            return_qualif: None,
            qualif: Qualif::empty(),
            const_fn_arg_vars: BitVector::new(mir.var_decls.len()),
            location: Location {
                block: START_BLOCK,
                statement_index: 0
            },
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
        self.qualif.restrict(ty, self.tcx, &self.param_env);
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

    /// Check for NEEDS_DROP (from an ADT or const fn call) and
    /// error, unless we're in a function, or the feature-gate
    /// for globals with destructors is enabled.
    fn deny_drop(&self) {
        if self.mode == Mode::Fn || !self.qualif.intersects(Qualif::NEEDS_DROP) {
            return;
        }

        // Static and const fn's allow destructors, but they're feature-gated.
        let msg = if self.mode != Mode::Const {
            // Feature-gate for globals with destructors is enabled.
            if self.tcx.sess.features.borrow().drop_types_in_const {
                return;
            }

            // This comes from a macro that has #[allow_internal_unstable].
            if self.tcx.sess.codemap().span_allows_unstable(self.span) {
                return;
            }

            format!("destructors in {}s are an unstable feature",
                    self.mode)
        } else {
            format!("{}s are not allowed to have destructors",
                    self.mode)
        };

        let mut err =
            struct_span_err!(self.tcx.sess, self.span, E0493, "{}", msg);
        if self.mode != Mode::Const {
            help!(&mut err,
                  "in Nightly builds, add `#![feature(drop_types_in_const)]` \
                   to the crate attributes to enable");
        }
        err.emit();
    }

    /// Check if an Lvalue with the current qualifications could
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
            span_err!(self.tcx.sess, self.span, E0394, "{}", msg);

            // Replace STATIC with NOT_CONST to avoid further errors.
            self.qualif = self.qualif - Qualif::STATIC;
            self.add(Qualif::NOT_CONST);

            false
        } else {
            true
        }
    }

    /// Assign the current qualification to the given destination.
    fn assign(&mut self, dest: &Lvalue<'tcx>) {
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
            if let Lvalue::Temp(index) = *dest {
                if self.temp_promotion_state[index].is_promotable() {
                    store(&mut self.temp_qualif[index]);
                }
            }
            return;
        }

        match *dest {
            Lvalue::Temp(index) => store(&mut self.temp_qualif[index]),
            Lvalue::ReturnPointer => store(&mut self.return_qualif),

            Lvalue::Projection(box Projection {
                base: Lvalue::Temp(index),
                elem: ProjectionElem::Deref
            }) if self.mir.temp_decls[index].ty.is_unique()
               && self.temp_qualif[index].map_or(false, |qualif| {
                    qualif.intersects(Qualif::NOT_CONST)
               }) => {
                // Part of `box expr`, we should've errored
                // already for the Box allocation Rvalue.
            }

            // This must be an explicit assignment.
            _ => {
                // Catch more errors in the destination.
                self.visit_lvalue(dest, LvalueContext::Store);
                self.statement_like();
            }
        }
    }

    /// Qualify a whole const, static initializer or const fn.
    fn qualify_const(&mut self) -> Qualif {
        let mir = self.mir;

        let mut seen_blocks = BitVector::new(mir.basic_blocks().len());
        let mut bb = START_BLOCK;
        loop {
            seen_blocks.insert(bb.index());

            self.visit_basic_block_data(bb, &mir[bb]);

            let target = match mir[bb].terminator().kind {
                TerminatorKind::Goto { target } |
                // Drops are considered noops.
                TerminatorKind::Drop { target, .. } |
                TerminatorKind::Assert { target, .. } |
                TerminatorKind::Call { destination: Some((_, target)), .. } => {
                    Some(target)
                }

                // Non-terminating calls cannot produce any value.
                TerminatorKind::Call { destination: None, .. } => {
                    return Qualif::empty();
                }

                TerminatorKind::If {..} |
                TerminatorKind::Switch {..} |
                TerminatorKind::SwitchInt {..} |
                TerminatorKind::DropAndReplace { .. } |
                TerminatorKind::Resume |
                TerminatorKind::Unreachable => None,

                TerminatorKind::Return => {
                    // Check for unused values. This usually means
                    // there are extra statements in the AST.
                    for temp in mir.temp_decls.indices() {
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
                    for index in 0..mir.var_decls.len() {
                        if !self.const_fn_arg_vars.contains(index) {
                            self.assign(&Lvalue::Var(Var::new(index)));
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

        let return_ty = mir.return_ty.unwrap();
        self.qualif = self.return_qualif.unwrap_or(Qualif::NOT_CONST);

        match self.mode {
            Mode::StaticMut => {
                // Check for destructors in static mut.
                self.add_type(return_ty);
                self.deny_drop();
            }
            _ => {
                // Account for errors in consts by using the
                // conservative type qualification instead.
                if self.qualif.intersects(Qualif::CONST_ERROR) {
                    self.qualif = Qualif::empty();
                    self.add_type(return_ty);
                }
            }
        }
        self.qualif
    }
}

/// Accumulates an Rvalue or Call's effects in self.qualif.
/// For functions (constant or not), it also records
/// candidates for promotion in promotion_candidates.
impl<'a, 'tcx> Visitor<'tcx> for Qualifier<'a, 'tcx, 'tcx> {
    fn visit_lvalue(&mut self, lvalue: &Lvalue<'tcx>, context: LvalueContext) {
        match *lvalue {
            Lvalue::Arg(_) => {
                self.add(Qualif::FN_ARGUMENT);
            }
            Lvalue::Var(_) => {
                self.add(Qualif::NOT_CONST);
            }
            Lvalue::Temp(index) => {
                if !self.temp_promotion_state[index].is_promotable() {
                    self.add(Qualif::NOT_PROMOTABLE);
                }

                if let Some(qualif) = self.temp_qualif[index] {
                    self.add(qualif);
                } else {
                    self.not_const();
                }
            }
            Lvalue::Static(_) => {
                self.add(Qualif::STATIC);
                if self.mode == Mode::Const || self.mode == Mode::ConstFn {
                    span_err!(self.tcx.sess, self.span, E0013,
                              "{}s cannot refer to statics, use \
                               a constant instead", self.mode);
                }
            }
            Lvalue::ReturnPointer => {
                self.not_const();
            }
            Lvalue::Projection(ref proj) => {
                self.nest(|this| {
                    this.super_lvalue(lvalue, context);
                    match proj.elem {
                        ProjectionElem::Deref => {
                            if !this.try_consume() {
                                return;
                            }

                            if this.qualif.intersects(Qualif::STATIC_REF) {
                                this.qualif = this.qualif - Qualif::STATIC_REF;
                                this.add(Qualif::STATIC);
                            }

                            let base_ty = this.mir.lvalue_ty(this.tcx, &proj.base)
                                              .to_ty(this.tcx);
                            if let ty::TyRawPtr(_) = base_ty.sty {
                                this.add(Qualif::NOT_CONST);
                                if this.mode != Mode::Fn {
                                    span_err!(this.tcx.sess, this.span, E0396,
                                              "raw pointers cannot be dereferenced in {}s",
                                              this.mode);
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
                            let ty = this.mir.lvalue_ty(this.tcx, lvalue)
                                         .to_ty(this.tcx);
                            this.qualif.restrict(ty, this.tcx, &this.param_env);
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

    fn visit_operand(&mut self, operand: &Operand<'tcx>) {
        match *operand {
            Operand::Consume(_) => {
                self.nest(|this| {
                    this.super_operand(operand);
                    this.try_consume();
                });
            }
            Operand::Constant(ref constant) => {
                // Only functions and methods can have these types.
                if let ty::TyFnDef(..) = constant.ty.sty {
                    return;
                }

                if let Literal::Item { def_id, substs } = constant.literal {
                    // Don't peek inside generic (associated) constants.
                    if !substs.types.is_empty() {
                        self.add_type(constant.ty);
                    } else {
                        let qualif = qualify_const_item_cached(self.tcx,
                                                               self.qualif_map,
                                                               self.mir_map,
                                                               def_id);
                        self.add(qualif);
                    }

                    // FIXME(eddyb) check recursive constants here,
                    // instead of rustc_passes::static_recursion.
                    if self.qualif.intersects(Qualif::RECURSIVE) {
                        span_bug!(constant.span,
                                  "recursive constant wasn't caught earlier");
                    }

                    // Let `const fn` transitively have destructors,
                    // but they do get stopped in `const` or `static`.
                    if self.mode != Mode::ConstFn {
                        self.deny_drop();
                    }
                }
            }
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>) {
        // Recurse through operands and lvalues.
        self.super_rvalue(rvalue);

        match *rvalue {
            Rvalue::Use(_) |
            Rvalue::Repeat(..) |
            Rvalue::UnaryOp(..) |
            Rvalue::CheckedBinaryOp(..) |
            Rvalue::Cast(CastKind::ReifyFnPointer, _, _) |
            Rvalue::Cast(CastKind::UnsafeFnPointer, _, _) |
            Rvalue::Cast(CastKind::Unsize, _, _) => {}

            Rvalue::Len(_) => {
                // Static lvalues in consts would have errored already,
                // don't treat length checks as reads from statics.
                self.qualif = self.qualif - Qualif::STATIC;
            }

            Rvalue::Ref(_, kind, ref lvalue) => {
                // Static lvalues in consts would have errored already,
                // only keep track of references to them here.
                if self.qualif.intersects(Qualif::STATIC) {
                    self.qualif = self.qualif - Qualif::STATIC;
                    self.add(Qualif::STATIC_REF);
                }

                let ty = self.mir.lvalue_ty(self.tcx, lvalue).to_ty(self.tcx);
                if kind == BorrowKind::Mut {
                    // In theory, any zero-sized value could be borrowed
                    // mutably without consequences. However, only &mut []
                    // is allowed right now, and only in functions.
                    let allow = if self.mode == Mode::StaticMut {
                        // Inside a `static mut`, &mut [...] is also allowed.
                        match ty.sty {
                            ty::TyArray(..) | ty::TySlice(_) => {
                                // Mutating can expose drops, be conservative.
                                self.add_type(ty);
                                self.deny_drop();
                                true
                            }
                            _ => false
                        }
                    } else if let ty::TyArray(_, 0) = ty.sty {
                        self.mode == Mode::Fn
                    } else {
                        false
                    };

                    if !allow {
                        self.add(Qualif::NOT_CONST);
                        if self.mode != Mode::Fn {
                            span_err!(self.tcx.sess, self.span, E0017,
                                      "references in {}s may only refer \
                                       to immutable values", self.mode);
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
                                      "cannot borrow a constant which contains \
                                       interior mutability, create a static instead");
                        }
                    }
                }

                // We might have a candidate for promotion.
                let candidate = Candidate::Ref(self.location);
                if self.mode == Mode::Fn || self.mode == Mode::ConstFn {
                    if !self.qualif.intersects(Qualif::NEVER_PROMOTE) {
                        // We can only promote direct borrows of temps.
                        if let Lvalue::Temp(_) = *lvalue {
                            self.promotion_candidates.push(candidate);
                        }
                    }
                }
            }

            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) => {
                let operand_ty = self.mir.operand_ty(self.tcx, operand);
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
                if let ty::TyRawPtr(_) = self.mir.operand_ty(self.tcx, lhs).sty {
                    assert!(op == BinOp::Eq || op == BinOp::Ne ||
                            op == BinOp::Le || op == BinOp::Lt ||
                            op == BinOp::Ge || op == BinOp::Gt);

                    self.add(Qualif::NOT_CONST);
                    if self.mode != Mode::Fn {
                        span_err!(self.tcx.sess, self.span, E0395,
                                  "raw pointers cannot be compared in {}s",
                                  self.mode);
                    }
                }
            }

            Rvalue::Box(_) => {
                self.add(Qualif::NOT_CONST);
                if self.mode != Mode::Fn {
                    span_err!(self.tcx.sess, self.span, E0010,
                              "allocations are not allowed in {}s", self.mode);
                }
            }

            Rvalue::Aggregate(ref kind, _) => {
                if let AggregateKind::Adt(def, _, _) = *kind {
                    if def.has_dtor() {
                        self.add(Qualif::NEEDS_DROP);
                        self.deny_drop();
                    }

                    if Some(def.did) == self.tcx.lang_items.unsafe_cell_type() {
                        let ty = self.mir.rvalue_ty(self.tcx, rvalue).unwrap();
                        self.add_type(ty);
                        assert!(self.qualif.intersects(Qualif::MUTABLE_INTERIOR));
                        // Even if the value inside may not need dropping,
                        // mutating it would change that.
                        if !self.qualif.intersects(Qualif::NOT_CONST) {
                            self.deny_drop();
                        }
                    }
                }
            }

            Rvalue::InlineAsm {..} => {
                self.not_const();
            }
        }
    }

    fn visit_terminator_kind(&mut self, bb: BasicBlock, kind: &TerminatorKind<'tcx>) {
        if let TerminatorKind::Call { ref func, ref args, ref destination, .. } = *kind {
            self.visit_operand(func);

            let fn_ty = self.mir.operand_ty(self.tcx, func);
            let (is_shuffle, is_const_fn) = match fn_ty.sty {
                ty::TyFnDef(def_id, _, f) => {
                    (f.abi == Abi::PlatformIntrinsic &&
                     self.tcx.item_name(def_id).as_str().starts_with("simd_shuffle"),
                     is_const_fn(self.tcx, def_id))
                }
                _ => (false, false)
            };

            for (i, arg) in args.iter().enumerate() {
                self.nest(|this| {
                    this.visit_operand(arg);
                    if is_shuffle && i == 2 && this.mode == Mode::Fn {
                        let candidate = Candidate::ShuffleIndices(bb);
                        if !this.qualif.intersects(Qualif::NEVER_PROMOTE) {
                            this.promotion_candidates.push(candidate);
                        } else {
                            span_err!(this.tcx.sess, this.span, E0526,
                                      "shuffle indices are not constant");
                        }
                    }
                });
            }

            // Const fn calls.
            if is_const_fn {
                // We are in a const or static initializer,
                if self.mode != Mode::Fn &&

                    // feature-gate is not enabled,
                    !self.tcx.sess.features.borrow().const_fn &&

                    // this doesn't come from a crate with the feature-gate enabled,
                    self.def_id.is_local() &&

                    // this doesn't come from a macro that has #[allow_internal_unstable]
                    !self.tcx.sess.codemap().span_allows_unstable(self.span)
                {
                    let mut err = self.tcx.sess.struct_span_err(self.span,
                        "const fns are an unstable feature");
                    help!(&mut err,
                          "in Nightly builds, add `#![feature(const_fn)]` \
                           to the crate attributes to enable");
                    err.emit();
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
                    let ty = self.mir.lvalue_ty(tcx, dest).to_ty(tcx);
                    self.qualif = Qualif::empty();
                    self.add_type(ty);

                    // Let `const fn` transitively have destructors,
                    // but they do get stopped in `const` or `static`.
                    if self.mode != Mode::ConstFn {
                        self.deny_drop();
                    }
                }
                self.assign(dest);
            }
        } else {
            // Qualify any operands inside other terminators.
            self.super_terminator_kind(bb, kind);
        }
    }

    fn visit_assign(&mut self, _: BasicBlock, dest: &Lvalue<'tcx>, rvalue: &Rvalue<'tcx>) {
        self.visit_rvalue(rvalue);

        // Check the allowed const fn argument forms.
        if let (Mode::ConstFn, &Lvalue::Var(index)) = (self.mode, dest) {
            if self.const_fn_arg_vars.insert(index.index()) {
                // Direct use of an argument is permitted.
                if let Rvalue::Use(Operand::Consume(Lvalue::Arg(_))) = *rvalue {
                    return;
                }

                // Avoid a generic error for other uses of arguments.
                if self.qualif.intersects(Qualif::FN_ARGUMENT) {
                    let decl = &self.mir.var_decls[index];
                    span_err!(self.tcx.sess, decl.source_info.span, E0022,
                              "arguments of constant functions can only \
                               be immutable by-value bindings");
                    return;
                }
            }
        }

        self.assign(dest);
    }

    fn visit_source_info(&mut self, source_info: &SourceInfo) {
        self.span = source_info.span;
    }

    fn visit_statement(&mut self, bb: BasicBlock, statement: &Statement<'tcx>) {
        assert_eq!(self.location.block, bb);
        self.nest(|this| this.super_statement(bb, statement));
        self.location.statement_index += 1;
    }

    fn visit_terminator(&mut self, bb: BasicBlock, terminator: &Terminator<'tcx>) {
        assert_eq!(self.location.block, bb);
        self.nest(|this| this.super_terminator(bb, terminator));
    }

    fn visit_basic_block_data(&mut self, bb: BasicBlock, data: &BasicBlockData<'tcx>) {
        self.location.statement_index = 0;
        self.location.block = bb;
        self.super_basic_block_data(bb, data);
    }
}

fn qualify_const_item_cached<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       qualif_map: &mut DefIdMap<Qualif>,
                                       mir_map: Option<&MirMap<'tcx>>,
                                       def_id: DefId)
                                       -> Qualif {
    match qualif_map.entry(def_id) {
        Entry::Occupied(entry) => return *entry.get(),
        Entry::Vacant(entry) => {
            // Guard against `const` recursion.
            entry.insert(Qualif::RECURSIVE);
        }
    }

    let extern_mir;
    let param_env_and_mir = if def_id.is_local() {
        let node_id = tcx.map.as_local_node_id(def_id).unwrap();
        mir_map.and_then(|map| map.map.get(&node_id)).map(|mir| {
            (ty::ParameterEnvironment::for_item(tcx, node_id), mir)
        })
    } else if let Some(mir) = tcx.sess.cstore.maybe_get_item_mir(tcx, def_id) {
        // These should only be monomorphic constants.
        extern_mir = mir;
        Some((tcx.empty_parameter_environment(), &extern_mir))
    } else {
        None
    };

    let (param_env, mir) = param_env_and_mir.unwrap_or_else(|| {
        bug!("missing constant MIR for {}", tcx.item_path_str(def_id))
    });

    let mut qualifier = Qualifier::new(tcx, param_env, qualif_map, mir_map,
                                       def_id, mir, Mode::Const);
    let qualif = qualifier.qualify_const();
    qualifier.qualif_map.insert(def_id, qualif);
    qualif
}

pub struct QualifyAndPromoteConstants;

impl Pass for QualifyAndPromoteConstants {}

impl<'tcx> MirMapPass<'tcx> for QualifyAndPromoteConstants {
    fn run_pass<'a>(&mut self,
                    tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    map: &mut MirMap<'tcx>,
                    hooks: &mut [Box<for<'s> MirPassHook<'s>>]) {
        let mut qualif_map = DefIdMap();

        // First, visit `const` items, potentially recursing, to get
        // accurate MUTABLE_INTERIOR and NEEDS_DROP qualifications.
        for &id in map.map.keys() {
            let def_id = tcx.map.local_def_id(id);
            let _task = tcx.dep_graph.in_task(self.dep_node(def_id));
            let src = MirSource::from_node(tcx, id);
            if let MirSource::Const(_) = src {
                qualify_const_item_cached(tcx, &mut qualif_map, Some(map), def_id);
            }
        }

        // Then, handle everything else, without recursing,
        // as the MIR map is not shared, since promotion
        // in functions (including `const fn`) mutates it.
        for (&id, mir) in &mut map.map {
            let def_id = tcx.map.local_def_id(id);
            let _task = tcx.dep_graph.in_task(self.dep_node(def_id));
            let src = MirSource::from_node(tcx, id);
            let mode = match src {
                MirSource::Fn(_) => {
                    if is_const_fn(tcx, def_id) {
                        Mode::ConstFn
                    } else {
                        Mode::Fn
                    }
                }
                MirSource::Const(_) => continue,
                MirSource::Static(_, hir::MutImmutable) => Mode::Static,
                MirSource::Static(_, hir::MutMutable) => Mode::StaticMut,
                MirSource::Promoted(..) => bug!()
            };
            let param_env = ty::ParameterEnvironment::for_item(tcx, id);

            for hook in &mut *hooks {
                hook.on_mir_pass(tcx, src, mir, self, false);
            }

            if mode == Mode::Fn || mode == Mode::ConstFn {
                // This is ugly because Qualifier holds onto mir,
                // which can't be mutated until its scope ends.
                let (temps, candidates) = {
                    let mut qualifier = Qualifier::new(tcx, param_env, &mut qualif_map,
                                                       None, def_id, mir, mode);
                    if mode == Mode::ConstFn {
                        // Enforce a constant-like CFG for `const fn`.
                        qualifier.qualify_const();
                    } else {
                        while let Some((bb, data)) = qualifier.rpo.next() {
                            qualifier.visit_basic_block_data(bb, data);
                        }
                    }

                    (qualifier.temp_promotion_state,
                     qualifier.promotion_candidates)
                };

                // Do the actual promotion, now that we know what's viable.
                promote_consts::promote_candidates(mir, tcx, temps, candidates);
            } else {
                let mut qualifier = Qualifier::new(tcx, param_env, &mut qualif_map,
                                                   None, def_id, mir, mode);
                qualifier.qualify_const();
            }

            for hook in &mut *hooks {
                hook.on_mir_pass(tcx, src, mir, self, true);
            }

            // Statics must be Sync.
            if mode == Mode::Static {
                let ty = mir.return_ty.unwrap();
                tcx.infer_ctxt(None, None, ProjectionMode::AnyFinal).enter(|infcx| {
                    let cause = traits::ObligationCause::new(mir.span, id, traits::SharedStatic);
                    let mut fulfillment_cx = traits::FulfillmentContext::new();
                    fulfillment_cx.register_builtin_bound(&infcx, ty, ty::BoundSync, cause);
                    if let Err(err) = fulfillment_cx.select_all_or_error(&infcx) {
                        infcx.report_fulfillment_errors(&err);
                    }

                    if let Err(errors) = fulfillment_cx.select_rfc1592_obligations(&infcx) {
                        infcx.report_fulfillment_errors_as_warnings(&errors, id);
                    }
                });
            }
        }
    }
}
