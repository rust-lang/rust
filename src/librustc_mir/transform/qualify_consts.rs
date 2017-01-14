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
use rustc::hir::map as hir_map;
use rustc::hir::def_id::DefId;
use rustc::hir::map::blocks::FnLikeNode;
use rustc::traits::{self, Reveal};
use rustc::ty::{self, TyCtxt, Ty};
use rustc::ty::cast::CastTy;
use rustc::mir::*;
use rustc::mir::traversal::ReversePostorder;
use rustc::mir::transform::{Pass, MirPass, MirSource};
use rustc::mir::visit::{LvalueContext, Visitor};
use rustc::util::nodemap::DefIdMap;
use rustc::middle::lang_items;
use syntax::abi::Abi;
use syntax::feature_gate::UnstableFeatures;
use syntax_pos::Span;

use std::collections::hash_map::Entry;
use std::fmt;
use std::usize;

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

pub fn is_const_fn(tcx: TyCtxt, def_id: DefId) -> bool {
    if let Some(node_id) = tcx.map.as_local_node_id(def_id) {
        if let Some(fn_like) = FnLikeNode::from_node(tcx.map.get(node_id)) {
            fn_like.constness() == hir::Constness::Const
        } else {
            false
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
    temp_qualif: IndexVec<Local, Option<Qualif>>,
    return_qualif: Option<Qualif>,
    qualif: Qualif,
    const_fn_arg_vars: BitVector,
    temp_promotion_state: IndexVec<Local, TempState>,
    promotion_candidates: Vec<Candidate>
}

impl<'a, 'tcx> Qualifier<'a, 'tcx, 'tcx> {
    fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
           param_env: ty::ParameterEnvironment<'tcx>,
           qualif_map: &'a mut DefIdMap<Qualif>,
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
            temp_qualif: IndexVec::from_elem(None, &mir.local_decls),
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
        } else {
            self.find_drop_implementation_method_span()
                .map(|span| err.span_label(span, &format!("destructor defined here")));

            err.span_label(self.span, &format!("constants cannot have destructors"));
        }

        err.emit();
    }

    fn find_drop_implementation_method_span(&self) -> Option<Span> {
        self.tcx.lang_items
            .drop_trait()
            .and_then(|drop_trait_id| {
                let mut span = None;

                self.tcx
                    .lookup_trait_def(drop_trait_id)
                    .for_each_relevant_impl(self.tcx, self.mir.return_ty, |impl_did| {
                        self.tcx.map
                            .as_local_node_id(impl_did)
                            .and_then(|impl_node_id| self.tcx.map.find(impl_node_id))
                            .map(|node| {
                                if let hir_map::NodeItem(item) = node {
                                    if let hir::ItemImpl(.., ref impl_item_refs) = item.node {
                                        span = impl_item_refs.first()
                                                             .map(|iiref| {
                                                                 self.tcx.map.impl_item(iiref.id)
                                                                             .span
                                                             });
                                    }
                                }
                            });
                    });

                span
            })
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
            struct_span_err!(self.tcx.sess, self.span, E0394, "{}", msg)
                .span_label(self.span, &format!("referring to another static by value"))
                .note(&format!("use the address-of operator or a constant instead"))
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
    fn assign(&mut self, dest: &Lvalue<'tcx>, location: Location) {
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
            if let Lvalue::Local(index) = *dest {
                if self.mir.local_kind(index) == LocalKind::Temp
                && self.temp_promotion_state[index].is_promotable() {
                    debug!("store to promotable temp {:?}", index);
                    store(&mut self.temp_qualif[index]);
                }
            }
            return;
        }

        match *dest {
            Lvalue::Local(index) if self.mir.local_kind(index) == LocalKind::Temp => {
                debug!("store to temp {:?}", index);
                store(&mut self.temp_qualif[index])
            }
            Lvalue::Local(index) if self.mir.local_kind(index) == LocalKind::ReturnPointer => {
                debug!("store to return pointer {:?}", index);
                store(&mut self.return_qualif)
            }

            Lvalue::Projection(box Projection {
                base: Lvalue::Local(index),
                elem: ProjectionElem::Deref
            }) if self.mir.local_kind(index) == LocalKind::Temp
               && self.mir.local_decls[index].ty.is_unique()
               && self.temp_qualif[index].map_or(false, |qualif| {
                    qualif.intersects(Qualif::NOT_CONST)
               }) => {
                // Part of `box expr`, we should've errored
                // already for the Box allocation Rvalue.
            }

            // This must be an explicit assignment.
            _ => {
                // Catch more errors in the destination.
                self.visit_lvalue(dest, LvalueContext::Store, location);
                self.statement_like();
            }
        }
    }

    /// Qualify a whole const, static initializer or const fn.
    fn qualify_const(&mut self) -> Qualif {
        debug!("qualifying {} {}", self.mode, self.tcx.item_path_str(self.def_id));

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
                            self.assign(&Lvalue::Local(index), Location {
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

        let return_ty = mir.return_ty;
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
    fn visit_lvalue(&mut self,
                    lvalue: &Lvalue<'tcx>,
                    context: LvalueContext<'tcx>,
                    location: Location) {
        match *lvalue {
            Lvalue::Local(local) => match self.mir.local_kind(local) {
                LocalKind::ReturnPointer => {
                    self.not_const();
                }
                LocalKind::Arg => {
                    self.add(Qualif::FN_ARGUMENT);
                }
                LocalKind::Var => {
                    self.add(Qualif::NOT_CONST);
                }
                LocalKind::Temp => {
                    if !self.temp_promotion_state[local].is_promotable() {
                        self.add(Qualif::NOT_PROMOTABLE);
                    }

                    if let Some(qualif) = self.temp_qualif[local] {
                        self.add(qualif);
                    } else {
                        self.not_const();
                    }
                }
            },
            Lvalue::Static(_) => {
                self.add(Qualif::STATIC);
                if self.mode == Mode::Const || self.mode == Mode::ConstFn {
                    span_err!(self.tcx.sess, self.span, E0013,
                              "{}s cannot refer to statics, use \
                               a constant instead", self.mode);
                }
            }
            Lvalue::Projection(ref proj) => {
                self.nest(|this| {
                    this.super_lvalue(lvalue, context, location);
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
                                        &format!("dereference of raw pointer in constant"))
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
                            let ty = lvalue.ty(this.mir, this.tcx).to_ty(this.tcx);
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

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        match *operand {
            Operand::Consume(_) => {
                self.nest(|this| {
                    this.super_operand(operand, location);
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
                    if substs.types().next().is_some() {
                        self.add_type(constant.ty);
                    } else {
                        let qualif = qualify_const_item_cached(self.tcx,
                                                               self.qualif_map,
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

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        // Recurse through operands and lvalues.
        self.super_rvalue(rvalue, location);

        match *rvalue {
            Rvalue::Use(_) |
            Rvalue::Repeat(..) |
            Rvalue::UnaryOp(..) |
            Rvalue::CheckedBinaryOp(..) |
            Rvalue::Cast(CastKind::ReifyFnPointer, ..) |
            Rvalue::Cast(CastKind::UnsafeFnPointer, ..) |
            Rvalue::Cast(CastKind::Unsize, ..) => {}

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

                let ty = lvalue.ty(self.mir, self.tcx).to_ty(self.tcx);
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
                            struct_span_err!(self.tcx.sess,  self.span, E0017,
                                             "references in {}s may only refer \
                                              to immutable values", self.mode)
                                .span_label(self.span, &format!("{}s require immutable values",
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
                                      "cannot borrow a constant which contains \
                                       interior mutability, create a static instead");
                        }
                    }
                }

                // We might have a candidate for promotion.
                let candidate = Candidate::Ref(location);
                if self.mode == Mode::Fn || self.mode == Mode::ConstFn {
                    if !self.qualif.intersects(Qualif::NEVER_PROMOTE) {
                        // We can only promote direct borrows of temps.
                        if let Lvalue::Local(local) = *lvalue {
                            if self.mir.local_kind(local) == LocalKind::Temp {
                                self.promotion_candidates.push(candidate);
                            }
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
                            op == BinOp::Ge || op == BinOp::Gt);

                    self.add(Qualif::NOT_CONST);
                    if self.mode != Mode::Fn {
                        struct_span_err!(
                            self.tcx.sess, self.span, E0395,
                            "raw pointers cannot be compared in {}s",
                            self.mode)
                        .span_label(
                            self.span,
                            &format!("comparing raw pointers in static"))
                        .emit();
                    }
                }
            }

            Rvalue::Box(_) => {
                self.add(Qualif::NOT_CONST);
                if self.mode != Mode::Fn {
                    struct_span_err!(self.tcx.sess, self.span, E0010,
                                     "allocations are not allowed in {}s", self.mode)
                        .span_label(self.span, &format!("allocation not allowed in {}s", self.mode))
                        .emit();
                }
            }

            Rvalue::Aggregate(ref kind, _) => {
                if let AggregateKind::Adt(def, ..) = *kind {
                    if def.has_dtor() {
                        self.add(Qualif::NEEDS_DROP);
                        self.deny_drop();
                    }

                    if Some(def.did) == self.tcx.lang_items.unsafe_cell_type() {
                        let ty = rvalue.ty(self.mir, self.tcx).unwrap();
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

    fn visit_terminator_kind(&mut self,
                             bb: BasicBlock,
                             kind: &TerminatorKind<'tcx>,
                             location: Location) {
        if let TerminatorKind::Call { ref func, ref args, ref destination, .. } = *kind {
            self.visit_operand(func, location);

            let fn_ty = func.ty(self.mir, self.tcx);
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
                    this.visit_operand(arg, location);
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
                    let ty = dest.ty(self.mir, tcx).to_ty(tcx);
                    self.qualif = Qualif::empty();
                    self.add_type(ty);

                    // Let `const fn` transitively have destructors,
                    // but they do get stopped in `const` or `static`.
                    if self.mode != Mode::ConstFn {
                        self.deny_drop();
                    }
                }
                self.assign(dest, location);
            }
        } else {
            // Qualify any operands inside other terminators.
            self.super_terminator_kind(bb, kind, location);
        }
    }

    fn visit_assign(&mut self,
                    _: BasicBlock,
                    dest: &Lvalue<'tcx>,
                    rvalue: &Rvalue<'tcx>,
                    location: Location) {
        self.visit_rvalue(rvalue, location);

        // Check the allowed const fn argument forms.
        if let (Mode::ConstFn, &Lvalue::Local(index)) = (self.mode, dest) {
            if self.mir.local_kind(index) == LocalKind::Var &&
               self.const_fn_arg_vars.insert(index.index()) {

                // Direct use of an argument is permitted.
                if let Rvalue::Use(Operand::Consume(Lvalue::Local(local))) = *rvalue {
                    if self.mir.local_kind(local) == LocalKind::Arg {
                        return;
                    }
                }

                // Avoid a generic error for other uses of arguments.
                if self.qualif.intersects(Qualif::FN_ARGUMENT) {
                    let decl = &self.mir.local_decls[index];
                    span_err!(self.tcx.sess, decl.source_info.unwrap().span, E0022,
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
                StatementKind::Assign(ref lvalue, ref rvalue) => {
                    this.visit_assign(bb, lvalue, rvalue, location);
                }
                StatementKind::SetDiscriminant { .. } |
                StatementKind::StorageLive(_) |
                StatementKind::StorageDead(_) |
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

fn qualify_const_item_cached<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       qualif_map: &mut DefIdMap<Qualif>,
                                       def_id: DefId)
                                       -> Qualif {
    match qualif_map.entry(def_id) {
        Entry::Occupied(entry) => return *entry.get(),
        Entry::Vacant(entry) => {
            // Guard against `const` recursion.
            entry.insert(Qualif::RECURSIVE);
        }
    }

    let param_env = if def_id.is_local() {
        let node_id = tcx.map.as_local_node_id(def_id).unwrap();
        ty::ParameterEnvironment::for_item(tcx, node_id)
    } else {
        // These should only be monomorphic constants.
        tcx.empty_parameter_environment()
    };

    let mir = &tcx.item_mir(def_id);
    let mut qualifier = Qualifier::new(tcx, param_env, qualif_map, def_id, mir, Mode::Const);
    let qualif = qualifier.qualify_const();
    qualifier.qualif_map.insert(def_id, qualif);
    qualif
}

#[derive(Default)]
pub struct QualifyAndPromoteConstants {
    qualif_map: DefIdMap<Qualif>
}

impl Pass for QualifyAndPromoteConstants {}

impl<'tcx> MirPass<'tcx> for QualifyAndPromoteConstants {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    src: MirSource, mir: &mut Mir<'tcx>) {
        let id = src.item_id();
        let def_id = tcx.map.local_def_id(id);
        let mode = match src {
            MirSource::Fn(_) => {
                if is_const_fn(tcx, def_id) {
                    Mode::ConstFn
                } else {
                    Mode::Fn
                }
            }
            MirSource::Const(_) => {
                match self.qualif_map.entry(def_id) {
                    Entry::Occupied(_) => return,
                    Entry::Vacant(entry) => {
                        // Guard against `const` recursion.
                        entry.insert(Qualif::RECURSIVE);
                        Mode::Const
                    }
                }
            }
            MirSource::Static(_, hir::MutImmutable) => Mode::Static,
            MirSource::Static(_, hir::MutMutable) => Mode::StaticMut,
            MirSource::Promoted(..) => return
        };
        let param_env = ty::ParameterEnvironment::for_item(tcx, id);

        if mode == Mode::Fn || mode == Mode::ConstFn {
            // This is ugly because Qualifier holds onto mir,
            // which can't be mutated until its scope ends.
            let (temps, candidates) = {
                let mut qualifier = Qualifier::new(tcx, param_env,
                                                   &mut self.qualif_map,
                                                   def_id, mir, mode);
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
            let mut qualifier = Qualifier::new(tcx, param_env,
                                               &mut self.qualif_map,
                                               def_id, mir, mode);
            let qualif = qualifier.qualify_const();

            if mode == Mode::Const {
                qualifier.qualif_map.insert(def_id, qualif);
            }
        }

        // Statics must be Sync.
        if mode == Mode::Static {
            let ty = mir.return_ty;
            tcx.infer_ctxt((), Reveal::NotSpecializable).enter(|infcx| {
                let cause = traits::ObligationCause::new(mir.span, id, traits::SharedStatic);
                let mut fulfillment_cx = traits::FulfillmentContext::new();
                fulfillment_cx.register_bound(&infcx, ty,
                                              tcx.require_lang_item(lang_items::SyncTraitLangItem),
                                              cause);
                if let Err(err) = fulfillment_cx.select_all_or_error(&infcx) {
                    infcx.report_fulfillment_errors(&err);
                }
            });
        }
    }
}
