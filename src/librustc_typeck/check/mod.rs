// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*

# check.rs

Within the check phase of type check, we check each item one at a time
(bodies of function expressions are checked as part of the containing
function).  Inference is used to supply types wherever they are
unknown.

By far the most complex case is checking the body of a function. This
can be broken down into several distinct phases:

- gather: creates type variables to represent the type of each local
  variable and pattern binding.

- main: the main pass does the lion's share of the work: it
  determines the types of all expressions, resolves
  methods, checks for most invalid conditions, and so forth.  In
  some cases, where a type is unknown, it may create a type or region
  variable and use that as the type of an expression.

  In the process of checking, various constraints will be placed on
  these type variables through the subtyping relationships requested
  through the `demand` module.  The `infer` module is in charge
  of resolving those constraints.

- regionck: after main is complete, the regionck pass goes over all
  types looking for regions and making sure that they did not escape
  into places they are not in scope.  This may also influence the
  final assignments of the various region variables if there is some
  flexibility.

- vtable: find and records the impls to use for each trait bound that
  appears on a type parameter.

- writeback: writes the final types within a function body, replacing
  type variables with their final inferred types.  These final types
  are written into the `tcx.node_types` table, which should *never* contain
  any reference to a type variable.

## Intermediate types

While type checking a function, the intermediate types for the
expressions, blocks, and so forth contained within the function are
stored in `fcx.node_types` and `fcx.item_substs`.  These types
may contain unresolved type variables.  After type checking is
complete, the functions in the writeback module are used to take the
types from this table, resolve them, and then write them into their
permanent home in the type context `ccx.tcx`.

This means that during inferencing you should use `fcx.write_ty()`
and `fcx.expr_ty()` / `fcx.node_ty()` to write/obtain the types of
nodes within the function.

The types of top-level items, which never contain unbound type
variables, are stored directly into the `tcx` tables.

n.b.: A type variable is not the same thing as a type parameter.  A
type variable is rather an "instance" of a type parameter: that is,
given a generic function `fn foo<T>(t: T)`: while checking the
function `foo`, the type `ty_param(0)` refers to the type `T`, which
is treated in abstract.  When `foo()` is called, however, `T` will be
substituted for a fresh type variable `N`.  This variable will
eventually be resolved to some concrete type (which might itself be
type parameter).

*/

pub use self::Expectation::*;
pub use self::compare_method::{compare_impl_method, compare_const_impl};
use self::TupleArgumentsFlag::*;

use astconv::{AstConv, ast_region_to_region};
use dep_graph::DepNode;
use fmt_macros::{Parser, Piece, Position};
use hir::def::{Def, CtorKind};
use hir::def_id::{DefId, LOCAL_CRATE};
use rustc::infer::{self, InferCtxt, InferOk, RegionVariableOrigin, TypeTrace};
use rustc::infer::type_variable::{self, TypeVariableOrigin};
use rustc::ty::subst::{Kind, Subst, Substs};
use rustc::traits::{self, ObligationCause, ObligationCauseCode, Reveal};
use rustc::ty::{ParamTy, ParameterEnvironment};
use rustc::ty::{LvaluePreference, NoPreference, PreferMutLvalue};
use rustc::ty::{self, ToPolyTraitRef, Ty, TyCtxt, Visibility};
use rustc::ty::{MethodCall, MethodCallee};
use rustc::ty::adjustment;
use rustc::ty::fold::{BottomUpFolder, TypeFoldable};
use rustc::ty::util::{Representability, IntTypeExt};
use require_c_abi_if_variadic;
use rscope::{ElisionFailureInfo, RegionScope};
use session::{Session, CompileResult};
use CrateCtxt;
use TypeAndSubsts;
use lint;
use util::common::{ErrorReported, indenter};
use util::nodemap::{DefIdMap, FxHashMap, FxHashSet, NodeMap};

use std::cell::{Cell, RefCell};
use std::cmp;
use std::mem::replace;
use std::ops::{self, Deref};
use syntax::abi::Abi;
use syntax::ast;
use syntax::attr;
use syntax::codemap::{self, original_sp, Spanned};
use syntax::feature_gate::{GateIssue, emit_feature_err};
use syntax::ptr::P;
use syntax::symbol::{Symbol, InternedString, keywords};
use syntax::util::lev_distance::find_best_match_for_name;
use syntax_pos::{self, BytePos, Span, DUMMY_SP};

use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir::{self, PatKind};
use rustc::middle::lang_items;
use rustc_back::slice;
use rustc_const_eval::eval_length;

mod assoc;
mod autoderef;
pub mod dropck;
pub mod _match;
pub mod writeback;
pub mod regionck;
pub mod coercion;
pub mod demand;
pub mod method;
mod upvar;
mod wfcheck;
mod cast;
mod closure;
mod callee;
mod compare_method;
mod intrinsic;
mod op;

/// closures defined within the function.  For example:
///
///     fn foo() {
///         bar(move|| { ... })
///     }
///
/// Here, the function `foo()` and the closure passed to
/// `bar()` will each have their own `FnCtxt`, but they will
/// share the inherited fields.
pub struct Inherited<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    ccx: &'a CrateCtxt<'a, 'gcx>,
    infcx: InferCtxt<'a, 'gcx, 'tcx>,
    locals: RefCell<NodeMap<Ty<'tcx>>>,

    fulfillment_cx: RefCell<traits::FulfillmentContext<'tcx>>,

    // When we process a call like `c()` where `c` is a closure type,
    // we may not have decided yet whether `c` is a `Fn`, `FnMut`, or
    // `FnOnce` closure. In that case, we defer full resolution of the
    // call until upvar inference can kick in and make the
    // decision. We keep these deferred resolutions grouped by the
    // def-id of the closure, so that once we decide, we can easily go
    // back and process them.
    deferred_call_resolutions: RefCell<DefIdMap<Vec<DeferredCallResolutionHandler<'gcx, 'tcx>>>>,

    deferred_cast_checks: RefCell<Vec<cast::CastCheck<'tcx>>>,

    // Anonymized types found in explicit return types and their
    // associated fresh inference variable. Writeback resolves these
    // variables to get the concrete type, which can be used to
    // deanonymize TyAnon, after typeck is done with all functions.
    anon_types: RefCell<DefIdMap<Ty<'tcx>>>,

    // Obligations which will have to be checked at the end of
    // type-checking, after all functions have been inferred.
    deferred_obligations: RefCell<Vec<traits::DeferredObligation<'tcx>>>,
}

impl<'a, 'gcx, 'tcx> Deref for Inherited<'a, 'gcx, 'tcx> {
    type Target = InferCtxt<'a, 'gcx, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.infcx
    }
}

trait DeferredCallResolution<'gcx, 'tcx> {
    fn resolve<'a>(&mut self, fcx: &FnCtxt<'a, 'gcx, 'tcx>);
}

type DeferredCallResolutionHandler<'gcx, 'tcx> = Box<DeferredCallResolution<'gcx, 'tcx>+'tcx>;

/// When type-checking an expression, we propagate downward
/// whatever type hint we are able in the form of an `Expectation`.
#[derive(Copy, Clone, Debug)]
pub enum Expectation<'tcx> {
    /// We know nothing about what type this expression should have.
    NoExpectation,

    /// This expression should have the type given (or some subtype)
    ExpectHasType(Ty<'tcx>),

    /// This expression will be cast to the `Ty`
    ExpectCastableToType(Ty<'tcx>),

    /// This rvalue expression will be wrapped in `&` or `Box` and coerced
    /// to `&Ty` or `Box<Ty>`, respectively. `Ty` is `[A]` or `Trait`.
    ExpectRvalueLikeUnsized(Ty<'tcx>),
}

impl<'a, 'gcx, 'tcx> Expectation<'tcx> {
    // Disregard "castable to" expectations because they
    // can lead us astray. Consider for example `if cond
    // {22} else {c} as u8` -- if we propagate the
    // "castable to u8" constraint to 22, it will pick the
    // type 22u8, which is overly constrained (c might not
    // be a u8). In effect, the problem is that the
    // "castable to" expectation is not the tightest thing
    // we can say, so we want to drop it in this case.
    // The tightest thing we can say is "must unify with
    // else branch". Note that in the case of a "has type"
    // constraint, this limitation does not hold.

    // If the expected type is just a type variable, then don't use
    // an expected type. Otherwise, we might write parts of the type
    // when checking the 'then' block which are incompatible with the
    // 'else' branch.
    fn adjust_for_branches(&self, fcx: &FnCtxt<'a, 'gcx, 'tcx>) -> Expectation<'tcx> {
        match *self {
            ExpectHasType(ety) => {
                let ety = fcx.shallow_resolve(ety);
                if !ety.is_ty_var() {
                    ExpectHasType(ety)
                } else {
                    NoExpectation
                }
            }
            ExpectRvalueLikeUnsized(ety) => {
                ExpectRvalueLikeUnsized(ety)
            }
            _ => NoExpectation
        }
    }

    /// Provide an expectation for an rvalue expression given an *optional*
    /// hint, which is not required for type safety (the resulting type might
    /// be checked higher up, as is the case with `&expr` and `box expr`), but
    /// is useful in determining the concrete type.
    ///
    /// The primary use case is where the expected type is a fat pointer,
    /// like `&[isize]`. For example, consider the following statement:
    ///
    ///    let x: &[isize] = &[1, 2, 3];
    ///
    /// In this case, the expected type for the `&[1, 2, 3]` expression is
    /// `&[isize]`. If however we were to say that `[1, 2, 3]` has the
    /// expectation `ExpectHasType([isize])`, that would be too strong --
    /// `[1, 2, 3]` does not have the type `[isize]` but rather `[isize; 3]`.
    /// It is only the `&[1, 2, 3]` expression as a whole that can be coerced
    /// to the type `&[isize]`. Therefore, we propagate this more limited hint,
    /// which still is useful, because it informs integer literals and the like.
    /// See the test case `test/run-pass/coerce-expect-unsized.rs` and #20169
    /// for examples of where this comes up,.
    fn rvalue_hint(fcx: &FnCtxt<'a, 'gcx, 'tcx>, ty: Ty<'tcx>) -> Expectation<'tcx> {
        match fcx.tcx.struct_tail(ty).sty {
            ty::TySlice(_) | ty::TyStr | ty::TyDynamic(..) => {
                ExpectRvalueLikeUnsized(ty)
            }
            _ => ExpectHasType(ty)
        }
    }

    // Resolves `expected` by a single level if it is a variable. If
    // there is no expected type or resolution is not possible (e.g.,
    // no constraints yet present), just returns `None`.
    fn resolve(self, fcx: &FnCtxt<'a, 'gcx, 'tcx>) -> Expectation<'tcx> {
        match self {
            NoExpectation => {
                NoExpectation
            }
            ExpectCastableToType(t) => {
                ExpectCastableToType(fcx.resolve_type_vars_if_possible(&t))
            }
            ExpectHasType(t) => {
                ExpectHasType(fcx.resolve_type_vars_if_possible(&t))
            }
            ExpectRvalueLikeUnsized(t) => {
                ExpectRvalueLikeUnsized(fcx.resolve_type_vars_if_possible(&t))
            }
        }
    }

    fn to_option(self, fcx: &FnCtxt<'a, 'gcx, 'tcx>) -> Option<Ty<'tcx>> {
        match self.resolve(fcx) {
            NoExpectation => None,
            ExpectCastableToType(ty) |
            ExpectHasType(ty) |
            ExpectRvalueLikeUnsized(ty) => Some(ty),
        }
    }

    fn only_has_type(self, fcx: &FnCtxt<'a, 'gcx, 'tcx>) -> Option<Ty<'tcx>> {
        match self.resolve(fcx) {
            ExpectHasType(ty) => Some(ty),
            _ => None
        }
    }
}

#[derive(Copy, Clone)]
pub struct UnsafetyState {
    pub def: ast::NodeId,
    pub unsafety: hir::Unsafety,
    pub unsafe_push_count: u32,
    from_fn: bool
}

impl UnsafetyState {
    pub fn function(unsafety: hir::Unsafety, def: ast::NodeId) -> UnsafetyState {
        UnsafetyState { def: def, unsafety: unsafety, unsafe_push_count: 0, from_fn: true }
    }

    pub fn recurse(&mut self, blk: &hir::Block) -> UnsafetyState {
        match self.unsafety {
            // If this unsafe, then if the outer function was already marked as
            // unsafe we shouldn't attribute the unsafe'ness to the block. This
            // way the block can be warned about instead of ignoring this
            // extraneous block (functions are never warned about).
            hir::Unsafety::Unsafe if self.from_fn => *self,

            unsafety => {
                let (unsafety, def, count) = match blk.rules {
                    hir::PushUnsafeBlock(..) =>
                        (unsafety, blk.id, self.unsafe_push_count.checked_add(1).unwrap()),
                    hir::PopUnsafeBlock(..) =>
                        (unsafety, blk.id, self.unsafe_push_count.checked_sub(1).unwrap()),
                    hir::UnsafeBlock(..) =>
                        (hir::Unsafety::Unsafe, blk.id, self.unsafe_push_count),
                    hir::DefaultBlock =>
                        (unsafety, self.def, self.unsafe_push_count),
                };
                UnsafetyState{ def: def,
                               unsafety: unsafety,
                               unsafe_push_count: count,
                               from_fn: false }
            }
        }
    }
}

/// Whether a node ever exits normally or not.
/// Tracked semi-automatically (through type variables
/// marked as diverging), with some manual adjustments
/// for control-flow primitives (approximating a CFG).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Diverges {
    /// Potentially unknown, some cases converge,
    /// others require a CFG to determine them.
    Maybe,

    /// Definitely known to diverge and therefore
    /// not reach the next sibling or its parent.
    Always,

    /// Same as `Always` but with a reachability
    /// warning already emitted
    WarnedAlways
}

// Convenience impls for combinig `Diverges`.

impl ops::BitAnd for Diverges {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        cmp::min(self, other)
    }
}

impl ops::BitOr for Diverges {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        cmp::max(self, other)
    }
}

impl ops::BitAndAssign for Diverges {
    fn bitand_assign(&mut self, other: Self) {
        *self = *self & other;
    }
}

impl ops::BitOrAssign for Diverges {
    fn bitor_assign(&mut self, other: Self) {
        *self = *self | other;
    }
}

impl Diverges {
    fn always(self) -> bool {
        self >= Diverges::Always
    }
}

#[derive(Clone)]
pub struct LoopCtxt<'gcx, 'tcx> {
    unified: Ty<'tcx>,
    coerce_to: Ty<'tcx>,
    break_exprs: Vec<&'gcx hir::Expr>,
    may_break: bool,
}

#[derive(Clone)]
pub struct EnclosingLoops<'gcx, 'tcx> {
    stack: Vec<LoopCtxt<'gcx, 'tcx>>,
    by_id: NodeMap<usize>,
}

impl<'gcx, 'tcx> EnclosingLoops<'gcx, 'tcx> {
    fn find_loop(&mut self, id: Option<ast::NodeId>) -> Option<&mut LoopCtxt<'gcx, 'tcx>> {
        if let Some(id) = id {
            if let Some(ix) = self.by_id.get(&id).cloned() {
                Some(&mut self.stack[ix])
            } else {
                None
            }
        } else {
            self.stack.last_mut()
        }
    }
}

#[derive(Clone)]
pub struct FnCtxt<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    ast_ty_to_ty_cache: RefCell<NodeMap<Ty<'tcx>>>,

    body_id: ast::NodeId,

    // This flag is set to true if, during the writeback phase, we encounter
    // a type error in this function.
    writeback_errors: Cell<bool>,

    // Number of errors that had been reported when we started
    // checking this function. On exit, if we find that *more* errors
    // have been reported, we will skip regionck and other work that
    // expects the types within the function to be consistent.
    err_count_on_creation: usize,

    ret_ty: Option<Ty<'tcx>>,

    ps: RefCell<UnsafetyState>,

    /// Whether the last checked node can ever exit.
    diverges: Cell<Diverges>,

    /// Whether any child nodes have any type errors.
    has_errors: Cell<bool>,

    enclosing_loops: RefCell<EnclosingLoops<'gcx, 'tcx>>,

    inh: &'a Inherited<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> Deref for FnCtxt<'a, 'gcx, 'tcx> {
    type Target = Inherited<'a, 'gcx, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.inh
    }
}

/// Helper type of a temporary returned by ccx.inherited(...).
/// Necessary because we can't write the following bound:
/// F: for<'b, 'tcx> where 'gcx: 'tcx FnOnce(Inherited<'b, 'gcx, 'tcx>).
pub struct InheritedBuilder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    ccx: &'a CrateCtxt<'a, 'gcx>,
    infcx: infer::InferCtxtBuilder<'a, 'gcx, 'tcx>
}

impl<'a, 'gcx, 'tcx> CrateCtxt<'a, 'gcx> {
    pub fn inherited(&'a self, id: ast::NodeId)
                     -> InheritedBuilder<'a, 'gcx, 'tcx> {
        let tables = ty::Tables::empty();
        let param_env = ParameterEnvironment::for_item(self.tcx, id);
        InheritedBuilder {
            ccx: self,
            infcx: self.tcx.infer_ctxt((tables, param_env), Reveal::NotSpecializable)
        }
    }
}

impl<'a, 'gcx, 'tcx> InheritedBuilder<'a, 'gcx, 'tcx> {
    fn enter<F, R>(&'tcx mut self, f: F) -> R
        where F: for<'b> FnOnce(Inherited<'b, 'gcx, 'tcx>) -> R
    {
        let ccx = self.ccx;
        self.infcx.enter(|infcx| f(Inherited::new(ccx, infcx)))
    }
}

impl<'a, 'gcx, 'tcx> Inherited<'a, 'gcx, 'tcx> {
    pub fn new(ccx: &'a CrateCtxt<'a, 'gcx>,
               infcx: InferCtxt<'a, 'gcx, 'tcx>)
               -> Self {
        Inherited {
            ccx: ccx,
            infcx: infcx,
            fulfillment_cx: RefCell::new(traits::FulfillmentContext::new()),
            locals: RefCell::new(NodeMap()),
            deferred_call_resolutions: RefCell::new(DefIdMap()),
            deferred_cast_checks: RefCell::new(Vec::new()),
            anon_types: RefCell::new(DefIdMap()),
            deferred_obligations: RefCell::new(Vec::new()),
        }
    }

    fn normalize_associated_types_in<T>(&self,
                                        span: Span,
                                        body_id: ast::NodeId,
                                        value: &T)
                                        -> T
        where T : TypeFoldable<'tcx>
    {
        assoc::normalize_associated_types_in(self,
                                             &mut self.fulfillment_cx.borrow_mut(),
                                             span,
                                             body_id,
                                             value)
    }

}

struct CheckItemTypesVisitor<'a, 'tcx: 'a> { ccx: &'a CrateCtxt<'a, 'tcx> }
struct CheckItemBodiesVisitor<'a, 'tcx: 'a> { ccx: &'a CrateCtxt<'a, 'tcx> }

impl<'a, 'tcx> Visitor<'tcx> for CheckItemTypesVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.ccx.tcx.map)
    }

    fn visit_item(&mut self, i: &'tcx hir::Item) {
        check_item_type(self.ccx, i);
        intravisit::walk_item(self, i);
    }

    fn visit_ty(&mut self, t: &'tcx hir::Ty) {
        match t.node {
            hir::TyArray(_, length) => {
                check_const_with_type(self.ccx, length, self.ccx.tcx.types.usize, length.node_id);
            }
            _ => {}
        }

        intravisit::walk_ty(self, t);
    }

    fn visit_expr(&mut self, e: &'tcx hir::Expr) {
        match e.node {
            hir::ExprRepeat(_, count) => {
                check_const_with_type(self.ccx, count, self.ccx.tcx.types.usize, count.node_id);
            }
            _ => {}
        }

        intravisit::walk_expr(self, e);
    }
}

impl<'a, 'tcx> ItemLikeVisitor<'tcx> for CheckItemBodiesVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &'tcx hir::Item) {
        check_item_body(self.ccx, i);
    }

    fn visit_trait_item(&mut self, _item: &'tcx hir::TraitItem) {
        // done as part of `visit_item` above
    }

    fn visit_impl_item(&mut self, _item: &'tcx hir::ImplItem) {
        // done as part of `visit_item` above
    }
}

pub fn check_wf_new(ccx: &CrateCtxt) -> CompileResult {
    ccx.tcx.sess.track_errors(|| {
        let mut visit = wfcheck::CheckTypeWellFormedVisitor::new(ccx);
        ccx.tcx.visit_all_item_likes_in_krate(DepNode::WfCheck, &mut visit.as_deep_visitor());
    })
}

pub fn check_item_types(ccx: &CrateCtxt) -> CompileResult {
    ccx.tcx.sess.track_errors(|| {
        let mut visit = CheckItemTypesVisitor { ccx: ccx };
        ccx.tcx.visit_all_item_likes_in_krate(DepNode::TypeckItemType,
                                              &mut visit.as_deep_visitor());
    })
}

pub fn check_item_bodies(ccx: &CrateCtxt) -> CompileResult {
    ccx.tcx.sess.track_errors(|| {
        let mut visit = CheckItemBodiesVisitor { ccx: ccx };
        ccx.tcx.visit_all_item_likes_in_krate(DepNode::TypeckItemBody, &mut visit);

        // Process deferred obligations, now that all functions
        // bodies have been fully inferred.
        for (&item_id, obligations) in ccx.deferred_obligations.borrow().iter() {
            // Use the same DepNode as for the body of the original function/item.
            let def_id = ccx.tcx.map.local_def_id(item_id);
            let _task = ccx.tcx.dep_graph.in_task(DepNode::TypeckItemBody(def_id));

            let param_env = ParameterEnvironment::for_item(ccx.tcx, item_id);
            ccx.tcx.infer_ctxt(param_env, Reveal::NotSpecializable).enter(|infcx| {
                let mut fulfillment_cx = traits::FulfillmentContext::new();
                for obligation in obligations.iter().map(|o| o.to_obligation()) {
                    fulfillment_cx.register_predicate_obligation(&infcx, obligation);
                }

                if let Err(errors) = fulfillment_cx.select_all_or_error(&infcx) {
                    infcx.report_fulfillment_errors(&errors);
                }
            });
        }
    })
}

pub fn check_drop_impls(ccx: &CrateCtxt) -> CompileResult {
    ccx.tcx.sess.track_errors(|| {
        let _task = ccx.tcx.dep_graph.in_task(DepNode::Dropck);
        let drop_trait = match ccx.tcx.lang_items.drop_trait() {
            Some(id) => ccx.tcx.lookup_trait_def(id), None => { return }
        };
        drop_trait.for_each_impl(ccx.tcx, |drop_impl_did| {
            let _task = ccx.tcx.dep_graph.in_task(DepNode::DropckImpl(drop_impl_did));
            if drop_impl_did.is_local() {
                match dropck::check_drop_impl(ccx, drop_impl_did) {
                    Ok(()) => {}
                    Err(()) => {
                        assert!(ccx.tcx.sess.has_errors());
                    }
                }
            }
        });
    })
}

fn check_bare_fn<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                           decl: &'tcx hir::FnDecl,
                           body_id: hir::BodyId,
                           fn_id: ast::NodeId,
                           span: Span) {
    let body = ccx.tcx.map.body(body_id);

    let raw_fty = ccx.tcx.item_type(ccx.tcx.map.local_def_id(fn_id));
    let fn_ty = match raw_fty.sty {
        ty::TyFnDef(.., f) => f,
        _ => span_bug!(body.value.span, "check_bare_fn: function type expected")
    };

    check_abi(ccx, span, fn_ty.abi);

    ccx.inherited(fn_id).enter(|inh| {
        // Compute the fty from point of view of inside fn.
        let fn_scope = inh.tcx.region_maps.call_site_extent(fn_id, body_id.node_id);
        let fn_sig =
            fn_ty.sig.subst(inh.tcx, &inh.parameter_environment.free_substs);
        let fn_sig =
            inh.tcx.liberate_late_bound_regions(fn_scope, &fn_sig);
        let fn_sig =
            inh.normalize_associated_types_in(body.value.span, body_id.node_id, &fn_sig);

        let fcx = check_fn(&inh, fn_ty.unsafety, fn_id, &fn_sig, decl, fn_id, body);

        fcx.select_all_obligations_and_apply_defaults();
        fcx.closure_analyze(body);
        fcx.select_obligations_where_possible();
        fcx.check_casts();
        fcx.select_all_obligations_or_error(); // Casts can introduce new obligations.

        fcx.regionck_fn(fn_id, body);
        fcx.resolve_type_vars_in_body(body);
    });
}

fn check_abi<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>, span: Span, abi: Abi) {
    if !ccx.tcx.sess.target.target.is_abi_supported(abi) {
        struct_span_err!(ccx.tcx.sess, span, E0570,
            "The ABI `{}` is not supported for the current target", abi).emit()
    }
}

struct GatherLocalsVisitor<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>
}

impl<'a, 'gcx, 'tcx> GatherLocalsVisitor<'a, 'gcx, 'tcx> {
    fn assign(&mut self, span: Span, nid: ast::NodeId, ty_opt: Option<Ty<'tcx>>) -> Ty<'tcx> {
        match ty_opt {
            None => {
                // infer the variable's type
                let var_ty = self.fcx.next_ty_var(TypeVariableOrigin::TypeInference(span));
                self.fcx.locals.borrow_mut().insert(nid, var_ty);
                var_ty
            }
            Some(typ) => {
                // take type that the user specified
                self.fcx.locals.borrow_mut().insert(nid, typ);
                typ
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> Visitor<'gcx> for GatherLocalsVisitor<'a, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'gcx> {
        NestedVisitorMap::None
    }

    // Add explicitly-declared locals.
    fn visit_local(&mut self, local: &'gcx hir::Local) {
        let o_ty = match local.ty {
            Some(ref ty) => Some(self.fcx.to_ty(&ty)),
            None => None
        };
        self.assign(local.span, local.id, o_ty);
        debug!("Local variable {:?} is assigned type {}",
               local.pat,
               self.fcx.ty_to_string(
                   self.fcx.locals.borrow().get(&local.id).unwrap().clone()));
        intravisit::walk_local(self, local);
    }

    // Add pattern bindings.
    fn visit_pat(&mut self, p: &'gcx hir::Pat) {
        if let PatKind::Binding(_, _, ref path1, _) = p.node {
            let var_ty = self.assign(p.span, p.id, None);

            self.fcx.require_type_is_sized(var_ty, p.span,
                                           traits::VariableType(p.id));

            debug!("Pattern binding {} is assigned to {} with type {:?}",
                   path1.node,
                   self.fcx.ty_to_string(
                       self.fcx.locals.borrow().get(&p.id).unwrap().clone()),
                   var_ty);
        }
        intravisit::walk_pat(self, p);
    }

    // Don't descend into the bodies of nested closures
    fn visit_fn(&mut self, _: intravisit::FnKind<'gcx>, _: &'gcx hir::FnDecl,
                _: hir::BodyId, _: Span, _: ast::NodeId) { }
}

/// Helper used by check_bare_fn and check_expr_fn. Does the grungy work of checking a function
/// body and returns the function context used for that purpose, since in the case of a fn item
/// there is still a bit more to do.
///
/// * ...
/// * inherited: other fields inherited from the enclosing fn (if any)
fn check_fn<'a, 'gcx, 'tcx>(inherited: &'a Inherited<'a, 'gcx, 'tcx>,
                            unsafety: hir::Unsafety,
                            unsafety_id: ast::NodeId,
                            fn_sig: &ty::FnSig<'tcx>,
                            decl: &'gcx hir::FnDecl,
                            fn_id: ast::NodeId,
                            body: &'gcx hir::Body)
                            -> FnCtxt<'a, 'gcx, 'tcx>
{
    let mut fn_sig = fn_sig.clone();

    debug!("check_fn(sig={:?}, fn_id={})", fn_sig, fn_id);

    // Create the function context.  This is either derived from scratch or,
    // in the case of function expressions, based on the outer context.
    let mut fcx = FnCtxt::new(inherited, None, body.value.id);
    let ret_ty = fn_sig.output();
    *fcx.ps.borrow_mut() = UnsafetyState::function(unsafety, unsafety_id);

    fcx.require_type_is_sized(ret_ty, decl.output.span(), traits::ReturnType);
    fcx.ret_ty = fcx.instantiate_anon_types(&Some(ret_ty));
    fn_sig = fcx.tcx.mk_fn_sig(fn_sig.inputs().iter().cloned(), &fcx.ret_ty.unwrap(),
                               fn_sig.variadic);

    GatherLocalsVisitor { fcx: &fcx, }.visit_body(body);

    // Add formal parameters.
    for (arg_ty, arg) in fn_sig.inputs().iter().zip(&body.arguments) {
        // The type of the argument must be well-formed.
        //
        // NB -- this is now checked in wfcheck, but that
        // currently only results in warnings, so we issue an
        // old-style WF obligation here so that we still get the
        // errors that we used to get.
        fcx.register_old_wf_obligation(arg_ty, arg.pat.span, traits::MiscObligation);

        // Check the pattern.
        fcx.check_pat_arg(&arg.pat, arg_ty, true);
        fcx.write_ty(arg.id, arg_ty);
    }

    inherited.tables.borrow_mut().liberated_fn_sigs.insert(fn_id, fn_sig);

    fcx.check_expr_coercable_to_type(&body.value, fcx.ret_ty.unwrap());

    fcx
}

fn check_struct(ccx: &CrateCtxt, id: ast::NodeId, span: Span) {
    let def_id = ccx.tcx.map.local_def_id(id);
    check_representable(ccx.tcx, span, def_id);

    if ccx.tcx.lookup_simd(def_id) {
        check_simd(ccx.tcx, span, def_id);
    }
}

fn check_union(ccx: &CrateCtxt, id: ast::NodeId, span: Span) {
    check_representable(ccx.tcx, span, ccx.tcx.map.local_def_id(id));
}

pub fn check_item_type<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>, it: &'tcx hir::Item) {
    debug!("check_item_type(it.id={}, it.name={})",
           it.id,
           ccx.tcx.item_path_str(ccx.tcx.map.local_def_id(it.id)));
    let _indenter = indenter();
    match it.node {
      // Consts can play a role in type-checking, so they are included here.
      hir::ItemStatic(.., e) |
      hir::ItemConst(_, e) => check_const(ccx, e, it.id),
      hir::ItemEnum(ref enum_definition, _) => {
        check_enum_variants(ccx,
                            it.span,
                            &enum_definition.variants,
                            it.id);
      }
      hir::ItemFn(..) => {} // entirely within check_item_body
      hir::ItemImpl(.., ref impl_item_refs) => {
          debug!("ItemImpl {} with id {}", it.name, it.id);
          let impl_def_id = ccx.tcx.map.local_def_id(it.id);
          if let Some(impl_trait_ref) = ccx.tcx.impl_trait_ref(impl_def_id) {
              check_impl_items_against_trait(ccx,
                                             it.span,
                                             impl_def_id,
                                             impl_trait_ref,
                                             impl_item_refs);
              let trait_def_id = impl_trait_ref.def_id;
              check_on_unimplemented(ccx, trait_def_id, it);
          }
      }
      hir::ItemTrait(..) => {
        let def_id = ccx.tcx.map.local_def_id(it.id);
        check_on_unimplemented(ccx, def_id, it);
      }
      hir::ItemStruct(..) => {
        check_struct(ccx, it.id, it.span);
      }
      hir::ItemUnion(..) => {
        check_union(ccx, it.id, it.span);
      }
      hir::ItemTy(_, ref generics) => {
        let def_id = ccx.tcx.map.local_def_id(it.id);
        let pty_ty = ccx.tcx.item_type(def_id);
        check_bounds_are_used(ccx, generics, pty_ty);
      }
      hir::ItemForeignMod(ref m) => {
        check_abi(ccx, it.span, m.abi);

        if m.abi == Abi::RustIntrinsic {
            for item in &m.items {
                intrinsic::check_intrinsic_type(ccx, item);
            }
        } else if m.abi == Abi::PlatformIntrinsic {
            for item in &m.items {
                intrinsic::check_platform_intrinsic_type(ccx, item);
            }
        } else {
            for item in &m.items {
                let generics = ccx.tcx.item_generics(ccx.tcx.map.local_def_id(item.id));
                if !generics.types.is_empty() {
                    let mut err = struct_span_err!(ccx.tcx.sess, item.span, E0044,
                        "foreign items may not have type parameters");
                    span_help!(&mut err, item.span,
                        "consider using specialization instead of \
                        type parameters");
                    err.emit();
                }

                if let hir::ForeignItemFn(ref fn_decl, _, _) = item.node {
                    require_c_abi_if_variadic(ccx.tcx, fn_decl, m.abi, item.span);
                }
            }
        }
      }
      _ => {/* nothing to do */ }
    }
}

pub fn check_item_body<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>, it: &'tcx hir::Item) {
    debug!("check_item_body(it.id={}, it.name={})",
           it.id,
           ccx.tcx.item_path_str(ccx.tcx.map.local_def_id(it.id)));
    let _indenter = indenter();
    match it.node {
      hir::ItemFn(ref decl, .., body_id) => {
        check_bare_fn(ccx, &decl, body_id, it.id, it.span);
      }
      hir::ItemImpl(.., ref impl_item_refs) => {
        debug!("ItemImpl {} with id {}", it.name, it.id);

        for impl_item_ref in impl_item_refs {
            let impl_item = ccx.tcx.map.impl_item(impl_item_ref.id);
            match impl_item.node {
                hir::ImplItemKind::Const(_, expr) => {
                    check_const(ccx, expr, impl_item.id)
                }
                hir::ImplItemKind::Method(ref sig, body_id) => {
                    check_bare_fn(ccx, &sig.decl, body_id, impl_item.id, impl_item.span);
                }
                hir::ImplItemKind::Type(_) => {
                    // Nothing to do here.
                }
            }
        }
      }
      hir::ItemTrait(.., ref trait_item_refs) => {
        for trait_item_ref in trait_item_refs {
            let trait_item = ccx.tcx.map.trait_item(trait_item_ref.id);
            match trait_item.node {
                hir::TraitItemKind::Const(_, Some(expr)) => {
                    check_const(ccx, expr, trait_item.id)
                }
                hir::TraitItemKind::Method(ref sig, hir::TraitMethod::Provided(body_id)) => {
                    check_bare_fn(ccx, &sig.decl, body_id, trait_item.id, trait_item.span);
                }
                hir::TraitItemKind::Method(_, hir::TraitMethod::Required(_)) |
                hir::TraitItemKind::Const(_, None) |
                hir::TraitItemKind::Type(..) => {
                    // Nothing to do.
                }
            }
        }
      }
      _ => {/* nothing to do */ }
    }
}

fn check_on_unimplemented<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                    def_id: DefId,
                                    item: &hir::Item) {
    let generics = ccx.tcx.item_generics(def_id);
    if let Some(ref attr) = item.attrs.iter().find(|a| {
        a.check_name("rustc_on_unimplemented")
    }) {
        if let Some(istring) = attr.value_str() {
            let istring = istring.as_str();
            let parser = Parser::new(&istring);
            let types = &generics.types;
            for token in parser {
                match token {
                    Piece::String(_) => (), // Normal string, no need to check it
                    Piece::NextArgument(a) => match a.position {
                        // `{Self}` is allowed
                        Position::ArgumentNamed(s) if s == "Self" => (),
                        // So is `{A}` if A is a type parameter
                        Position::ArgumentNamed(s) => match types.iter().find(|t| {
                            t.name == s
                        }) {
                            Some(_) => (),
                            None => {
                                let name = ccx.tcx.item_name(def_id);
                                span_err!(ccx.tcx.sess, attr.span, E0230,
                                                 "there is no type parameter \
                                                          {} on trait {}",
                                                           s, name);
                            }
                        },
                        // `{:1}` and `{}` are not to be used
                        Position::ArgumentIs(_) => {
                            span_err!(ccx.tcx.sess, attr.span, E0231,
                                                  "only named substitution \
                                                   parameters are allowed");
                        }
                    }
                }
            }
        } else {
            struct_span_err!(
                ccx.tcx.sess, attr.span, E0232,
                "this attribute must have a value")
                .span_label(attr.span, &format!("attribute requires a value"))
                .note(&format!("eg `#[rustc_on_unimplemented = \"foo\"]`"))
                .emit();
        }
    }
}

fn report_forbidden_specialization<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                             impl_item: &hir::ImplItem,
                                             parent_impl: DefId)
{
    let mut err = struct_span_err!(
        tcx.sess, impl_item.span, E0520,
        "`{}` specializes an item from a parent `impl`, but \
         that item is not marked `default`",
        impl_item.name);
    err.span_label(impl_item.span, &format!("cannot specialize default item `{}`",
                                            impl_item.name));

    match tcx.span_of_impl(parent_impl) {
        Ok(span) => {
            err.span_label(span, &"parent `impl` is here");
            err.note(&format!("to specialize, `{}` in the parent `impl` must be marked `default`",
                              impl_item.name));
        }
        Err(cname) => {
            err.note(&format!("parent implementation is in crate `{}`", cname));
        }
    }

    err.emit();
}

fn check_specialization_validity<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                           trait_def: &ty::TraitDef,
                                           impl_id: DefId,
                                           impl_item: &hir::ImplItem)
{
    let ancestors = trait_def.ancestors(impl_id);

    let kind = match impl_item.node {
        hir::ImplItemKind::Const(..) => ty::AssociatedKind::Const,
        hir::ImplItemKind::Method(..) => ty::AssociatedKind::Method,
        hir::ImplItemKind::Type(_) => ty::AssociatedKind::Type
    };
    let parent = ancestors.defs(tcx, impl_item.name, kind).skip(1).next()
        .map(|node_item| node_item.map(|parent| parent.defaultness));

    if let Some(parent) = parent {
        if parent.item.is_final() {
            report_forbidden_specialization(tcx, impl_item, parent.node.def_id());
        }
    }

}

fn check_impl_items_against_trait<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                            impl_span: Span,
                                            impl_id: DefId,
                                            impl_trait_ref: ty::TraitRef<'tcx>,
                                            impl_item_refs: &[hir::ImplItemRef]) {
    // If the trait reference itself is erroneous (so the compilation is going
    // to fail), skip checking the items here -- the `impl_item` table in `tcx`
    // isn't populated for such impls.
    if impl_trait_ref.references_error() { return; }

    // Locate trait definition and items
    let tcx = ccx.tcx;
    let trait_def = tcx.lookup_trait_def(impl_trait_ref.def_id);
    let mut overridden_associated_type = None;

    let impl_items = || impl_item_refs.iter().map(|iiref| ccx.tcx.map.impl_item(iiref.id));

    // Check existing impl methods to see if they are both present in trait
    // and compatible with trait signature
    for impl_item in impl_items() {
        let ty_impl_item = tcx.associated_item(tcx.map.local_def_id(impl_item.id));
        let ty_trait_item = tcx.associated_items(impl_trait_ref.def_id)
            .find(|ac| ac.name == ty_impl_item.name);

        // Check that impl definition matches trait definition
        if let Some(ty_trait_item) = ty_trait_item {
            match impl_item.node {
                hir::ImplItemKind::Const(..) => {
                    // Find associated const definition.
                    if ty_trait_item.kind == ty::AssociatedKind::Const {
                        compare_const_impl(ccx,
                                           &ty_impl_item,
                                           impl_item.span,
                                           &ty_trait_item,
                                           impl_trait_ref);
                    } else {
                         let mut err = struct_span_err!(tcx.sess, impl_item.span, E0323,
                                  "item `{}` is an associated const, \
                                  which doesn't match its trait `{}`",
                                  ty_impl_item.name,
                                  impl_trait_ref);
                         err.span_label(impl_item.span, &format!("does not match trait"));
                         // We can only get the spans from local trait definition
                         // Same for E0324 and E0325
                         if let Some(trait_span) = tcx.map.span_if_local(ty_trait_item.def_id) {
                            err.span_label(trait_span, &format!("item in trait"));
                         }
                         err.emit()
                    }
                }
                hir::ImplItemKind::Method(_, body_id) => {
                    let trait_span = tcx.map.span_if_local(ty_trait_item.def_id);
                    if ty_trait_item.kind == ty::AssociatedKind::Method {
                        let err_count = tcx.sess.err_count();
                        compare_impl_method(ccx,
                                            &ty_impl_item,
                                            impl_item.span,
                                            body_id.node_id,
                                            &ty_trait_item,
                                            impl_trait_ref,
                                            trait_span,
                                            true); // start with old-broken-mode
                        if err_count == tcx.sess.err_count() {
                            // old broken mode did not report an error. Try with the new mode.
                            compare_impl_method(ccx,
                                                &ty_impl_item,
                                                impl_item.span,
                                                body_id.node_id,
                                                &ty_trait_item,
                                                impl_trait_ref,
                                                trait_span,
                                                false); // use the new mode
                        }
                    } else {
                        let mut err = struct_span_err!(tcx.sess, impl_item.span, E0324,
                                  "item `{}` is an associated method, \
                                  which doesn't match its trait `{}`",
                                  ty_impl_item.name,
                                  impl_trait_ref);
                         err.span_label(impl_item.span, &format!("does not match trait"));
                         if let Some(trait_span) = tcx.map.span_if_local(ty_trait_item.def_id) {
                            err.span_label(trait_span, &format!("item in trait"));
                         }
                         err.emit()
                    }
                }
                hir::ImplItemKind::Type(_) => {
                    if ty_trait_item.kind == ty::AssociatedKind::Type {
                        if ty_trait_item.defaultness.has_value() {
                            overridden_associated_type = Some(impl_item);
                        }
                    } else {
                        let mut err = struct_span_err!(tcx.sess, impl_item.span, E0325,
                                  "item `{}` is an associated type, \
                                  which doesn't match its trait `{}`",
                                  ty_impl_item.name,
                                  impl_trait_ref);
                         err.span_label(impl_item.span, &format!("does not match trait"));
                         if let Some(trait_span) = tcx.map.span_if_local(ty_trait_item.def_id) {
                            err.span_label(trait_span, &format!("item in trait"));
                         }
                         err.emit()
                    }
                }
            }
        }

        check_specialization_validity(tcx, trait_def, impl_id, impl_item);
    }

    // Check for missing items from trait
    let mut missing_items = Vec::new();
    let mut invalidated_items = Vec::new();
    let associated_type_overridden = overridden_associated_type.is_some();
    for trait_item in tcx.associated_items(impl_trait_ref.def_id) {
        let is_implemented = trait_def.ancestors(impl_id)
            .defs(tcx, trait_item.name, trait_item.kind)
            .next()
            .map(|node_item| !node_item.node.is_from_trait())
            .unwrap_or(false);

        if !is_implemented {
            if !trait_item.defaultness.has_value() {
                missing_items.push(trait_item);
            } else if associated_type_overridden {
                invalidated_items.push(trait_item.name);
            }
        }
    }

    let signature = |item: &ty::AssociatedItem| {
        match item.kind {
            ty::AssociatedKind::Method => {
                format!("{}", tcx.item_type(item.def_id).fn_sig().0)
            }
            ty::AssociatedKind::Type => format!("type {};", item.name.to_string()),
            ty::AssociatedKind::Const => {
                format!("const {}: {:?};", item.name.to_string(), tcx.item_type(item.def_id))
            }
        }
    };

    if !missing_items.is_empty() {
        let mut err = struct_span_err!(tcx.sess, impl_span, E0046,
            "not all trait items implemented, missing: `{}`",
            missing_items.iter()
                  .map(|trait_item| trait_item.name.to_string())
                  .collect::<Vec<_>>().join("`, `"));
        err.span_label(impl_span, &format!("missing `{}` in implementation",
                missing_items.iter()
                    .map(|trait_item| trait_item.name.to_string())
                    .collect::<Vec<_>>().join("`, `")));
        for trait_item in missing_items {
            if let Some(span) = tcx.map.span_if_local(trait_item.def_id) {
                err.span_label(span, &format!("`{}` from trait", trait_item.name));
            } else {
                err.note(&format!("`{}` from trait: `{}`",
                                  trait_item.name,
                                  signature(&trait_item)));
            }
        }
        err.emit();
    }

    if !invalidated_items.is_empty() {
        let invalidator = overridden_associated_type.unwrap();
        span_err!(tcx.sess, invalidator.span, E0399,
                  "the following trait items need to be reimplemented \
                   as `{}` was overridden: `{}`",
                  invalidator.name,
                  invalidated_items.iter()
                                   .map(|name| name.to_string())
                                   .collect::<Vec<_>>().join("`, `"))
    }
}

/// Checks a constant with a given type.
fn check_const_with_type<'a, 'tcx>(ccx: &'a CrateCtxt<'a, 'tcx>,
                                   body: hir::BodyId,
                                   expected_type: Ty<'tcx>,
                                   id: ast::NodeId) {
    let body = ccx.tcx.map.body(body);
    ccx.inherited(id).enter(|inh| {
        let fcx = FnCtxt::new(&inh, None, body.value.id);
        fcx.require_type_is_sized(expected_type, body.value.span, traits::ConstSized);

        // Gather locals in statics (because of block expressions).
        // This is technically unnecessary because locals in static items are forbidden,
        // but prevents type checking from blowing up before const checking can properly
        // emit an error.
        GatherLocalsVisitor { fcx: &fcx }.visit_body(body);

        fcx.check_expr_coercable_to_type(&body.value, expected_type);

        fcx.select_all_obligations_and_apply_defaults();
        fcx.closure_analyze(body);
        fcx.select_obligations_where_possible();
        fcx.check_casts();
        fcx.select_all_obligations_or_error();

        fcx.regionck_expr(body);
        fcx.resolve_type_vars_in_body(body);
    });
}

fn check_const<'a, 'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                         body: hir::BodyId,
                         id: ast::NodeId) {
    let decl_ty = ccx.tcx.item_type(ccx.tcx.map.local_def_id(id));
    check_const_with_type(ccx, body, decl_ty, id);
}

/// Checks whether a type can be represented in memory. In particular, it
/// identifies types that contain themselves without indirection through a
/// pointer, which would mean their size is unbounded.
fn check_representable<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 sp: Span,
                                 item_def_id: DefId)
                                 -> bool {
    let rty = tcx.item_type(item_def_id);

    // Check that it is possible to represent this type. This call identifies
    // (1) types that contain themselves and (2) types that contain a different
    // recursive type. It is only necessary to throw an error on those that
    // contain themselves. For case 2, there must be an inner type that will be
    // caught by case 1.
    match rty.is_representable(tcx, sp) {
        Representability::SelfRecursive => {
            tcx.recursive_type_with_infinite_size_error(item_def_id).emit();
            return false
        }
        Representability::Representable | Representability::ContainsRecursive => (),
    }
    return true
}

pub fn check_simd<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, sp: Span, def_id: DefId) {
    let t = tcx.item_type(def_id);
    match t.sty {
        ty::TyAdt(def, substs) if def.is_struct() => {
            let fields = &def.struct_variant().fields;
            if fields.is_empty() {
                span_err!(tcx.sess, sp, E0075, "SIMD vector cannot be empty");
                return;
            }
            let e = fields[0].ty(tcx, substs);
            if !fields.iter().all(|f| f.ty(tcx, substs) == e) {
                struct_span_err!(tcx.sess, sp, E0076, "SIMD vector should be homogeneous")
                                .span_label(sp, &format!("SIMD elements must have the same type"))
                                .emit();
                return;
            }
            match e.sty {
                ty::TyParam(_) => { /* struct<T>(T, T, T, T) is ok */ }
                _ if e.is_machine()  => { /* struct(u8, u8, u8, u8) is ok */ }
                _ => {
                    span_err!(tcx.sess, sp, E0077,
                              "SIMD vector element type should be machine type");
                    return;
                }
            }
        }
        _ => ()
    }
}

#[allow(trivial_numeric_casts)]
pub fn check_enum_variants<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                    sp: Span,
                                    vs: &'tcx [hir::Variant],
                                    id: ast::NodeId) {
    let def_id = ccx.tcx.map.local_def_id(id);
    let hint = *ccx.tcx.lookup_repr_hints(def_id).get(0).unwrap_or(&attr::ReprAny);

    if hint != attr::ReprAny && vs.is_empty() {
        struct_span_err!(
            ccx.tcx.sess, sp, E0084,
            "unsupported representation for zero-variant enum")
            .span_label(sp, &format!("unsupported enum representation"))
            .emit();
    }

    let repr_type_ty = ccx.tcx.enum_repr_type(Some(&hint)).to_ty(ccx.tcx);
    if repr_type_ty == ccx.tcx.types.i128 || repr_type_ty == ccx.tcx.types.u128 {
        if !ccx.tcx.sess.features.borrow().i128_type {
            emit_feature_err(&ccx.tcx.sess.parse_sess,
                             "i128_type", sp, GateIssue::Language, "128-bit type is unstable");
        }
    }

    for v in vs {
        if let Some(e) = v.node.disr_expr {
            check_const_with_type(ccx, e, repr_type_ty, e.node_id);
        }
    }

    let def_id = ccx.tcx.map.local_def_id(id);

    let variants = &ccx.tcx.lookup_adt_def(def_id).variants;
    let mut disr_vals: Vec<ty::Disr> = Vec::new();
    for (v, variant) in vs.iter().zip(variants.iter()) {
        let current_disr_val = variant.disr_val;

        // Check for duplicate discriminant values
        if let Some(i) = disr_vals.iter().position(|&x| x == current_disr_val) {
            let variant_i_node_id = ccx.tcx.map.as_local_node_id(variants[i].did).unwrap();
            let variant_i = ccx.tcx.map.expect_variant(variant_i_node_id);
            let i_span = match variant_i.node.disr_expr {
                Some(expr) => ccx.tcx.map.span(expr.node_id),
                None => ccx.tcx.map.span(variant_i_node_id)
            };
            let span = match v.node.disr_expr {
                Some(expr) => ccx.tcx.map.span(expr.node_id),
                None => v.span
            };
            struct_span_err!(ccx.tcx.sess, span, E0081,
                             "discriminant value `{}` already exists", disr_vals[i])
                .span_label(i_span, &format!("first use of `{}`", disr_vals[i]))
                .span_label(span , &format!("enum already has `{}`", disr_vals[i]))
                .emit();
        }
        disr_vals.push(current_disr_val);
    }

    check_representable(ccx.tcx, sp, def_id);
}

impl<'a, 'gcx, 'tcx> AstConv<'gcx, 'tcx> for FnCtxt<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn ast_ty_to_ty_cache(&self) -> &RefCell<NodeMap<Ty<'tcx>>> {
        &self.ast_ty_to_ty_cache
    }

    fn get_generics(&self, _: Span, id: DefId)
                    -> Result<&'tcx ty::Generics<'tcx>, ErrorReported>
    {
        Ok(self.tcx().item_generics(id))
    }

    fn get_item_type(&self, _: Span, id: DefId) -> Result<Ty<'tcx>, ErrorReported>
    {
        Ok(self.tcx().item_type(id))
    }

    fn get_trait_def(&self, _: Span, id: DefId)
                     -> Result<&'tcx ty::TraitDef, ErrorReported>
    {
        Ok(self.tcx().lookup_trait_def(id))
    }

    fn ensure_super_predicates(&self, _: Span, _: DefId) -> Result<(), ErrorReported> {
        // all super predicates are ensured during collect pass
        Ok(())
    }

    fn get_free_substs(&self) -> Option<&Substs<'tcx>> {
        Some(&self.parameter_environment.free_substs)
    }

    fn get_type_parameter_bounds(&self,
                                 _: Span,
                                 node_id: ast::NodeId)
                                 -> Result<Vec<ty::PolyTraitRef<'tcx>>, ErrorReported>
    {
        let def = self.tcx.type_parameter_def(node_id);
        let r = self.parameter_environment
                                  .caller_bounds
                                  .iter()
                                  .filter_map(|predicate| {
                                      match *predicate {
                                          ty::Predicate::Trait(ref data) => {
                                              if data.0.self_ty().is_param(def.index) {
                                                  Some(data.to_poly_trait_ref())
                                              } else {
                                                  None
                                              }
                                          }
                                          _ => {
                                              None
                                          }
                                      }
                                  })
                                  .collect();
        Ok(r)
    }

    fn ty_infer(&self, span: Span) -> Ty<'tcx> {
        self.next_ty_var(TypeVariableOrigin::TypeInference(span))
    }

    fn ty_infer_for_def(&self,
                        ty_param_def: &ty::TypeParameterDef<'tcx>,
                        substs: &[Kind<'tcx>],
                        span: Span) -> Ty<'tcx> {
        self.type_var_for_def(span, ty_param_def, substs)
    }

    fn projected_ty_from_poly_trait_ref(&self,
                                        span: Span,
                                        poly_trait_ref: ty::PolyTraitRef<'tcx>,
                                        item_name: ast::Name)
                                        -> Ty<'tcx>
    {
        let (trait_ref, _) =
            self.replace_late_bound_regions_with_fresh_var(
                span,
                infer::LateBoundRegionConversionTime::AssocTypeProjection(item_name),
                &poly_trait_ref);

        self.normalize_associated_type(span, trait_ref, item_name)
    }

    fn projected_ty(&self,
                    span: Span,
                    trait_ref: ty::TraitRef<'tcx>,
                    item_name: ast::Name)
                    -> Ty<'tcx>
    {
        self.normalize_associated_type(span, trait_ref, item_name)
    }

    fn set_tainted_by_errors(&self) {
        self.infcx.set_tainted_by_errors()
    }
}

impl<'a, 'gcx, 'tcx> RegionScope for FnCtxt<'a, 'gcx, 'tcx> {
    fn object_lifetime_default(&self, span: Span) -> Option<ty::Region> {
        Some(self.base_object_lifetime_default(span))
    }

    fn base_object_lifetime_default(&self, span: Span) -> ty::Region {
        // RFC #599 specifies that object lifetime defaults take
        // precedence over other defaults. But within a fn body we
        // don't have a *default* region, rather we use inference to
        // find the *correct* region, which is strictly more general
        // (and anyway, within a fn body the right region may not even
        // be something the user can write explicitly, since it might
        // be some expression).
        *self.next_region_var(infer::MiscVariable(span))
    }

    fn anon_regions(&self, span: Span, count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<ElisionFailureInfo>>> {
        Ok((0..count).map(|_| {
            *self.next_region_var(infer::MiscVariable(span))
        }).collect())
    }
}

/// Controls whether the arguments are tupled. This is used for the call
/// operator.
///
/// Tupling means that all call-side arguments are packed into a tuple and
/// passed as a single parameter. For example, if tupling is enabled, this
/// function:
///
///     fn f(x: (isize, isize))
///
/// Can be called as:
///
///     f(1, 2);
///
/// Instead of:
///
///     f((1, 2));
#[derive(Clone, Eq, PartialEq)]
enum TupleArgumentsFlag {
    DontTupleArguments,
    TupleArguments,
}

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    pub fn new(inh: &'a Inherited<'a, 'gcx, 'tcx>,
               rty: Option<Ty<'tcx>>,
               body_id: ast::NodeId)
               -> FnCtxt<'a, 'gcx, 'tcx> {
        FnCtxt {
            ast_ty_to_ty_cache: RefCell::new(NodeMap()),
            body_id: body_id,
            writeback_errors: Cell::new(false),
            err_count_on_creation: inh.tcx.sess.err_count(),
            ret_ty: rty,
            ps: RefCell::new(UnsafetyState::function(hir::Unsafety::Normal,
                                                     ast::CRATE_NODE_ID)),
            diverges: Cell::new(Diverges::Maybe),
            has_errors: Cell::new(false),
            enclosing_loops: RefCell::new(EnclosingLoops {
                stack: Vec::new(),
                by_id: NodeMap(),
            }),
            inh: inh,
        }
    }

    pub fn param_env(&self) -> &ty::ParameterEnvironment<'gcx> {
        &self.parameter_environment
    }

    pub fn sess(&self) -> &Session {
        &self.tcx.sess
    }

    pub fn err_count_since_creation(&self) -> usize {
        self.tcx.sess.err_count() - self.err_count_on_creation
    }

    /// Produce warning on the given node, if the current point in the
    /// function is unreachable, and there hasn't been another warning.
    fn warn_if_unreachable(&self, id: ast::NodeId, span: Span, kind: &str) {
        if self.diverges.get() == Diverges::Always {
            self.diverges.set(Diverges::WarnedAlways);

            self.tcx.sess.add_lint(lint::builtin::UNREACHABLE_CODE,
                                   id, span,
                                   format!("unreachable {}", kind));
        }
    }

    pub fn cause(&self,
                 span: Span,
                 code: ObligationCauseCode<'tcx>)
                 -> ObligationCause<'tcx> {
        ObligationCause::new(span, self.body_id, code)
    }

    pub fn misc(&self, span: Span) -> ObligationCause<'tcx> {
        self.cause(span, ObligationCauseCode::MiscObligation)
    }

    /// Resolves type variables in `ty` if possible. Unlike the infcx
    /// version (resolve_type_vars_if_possible), this version will
    /// also select obligations if it seems useful, in an effort
    /// to get more type information.
    fn resolve_type_vars_with_obligations(&self, mut ty: Ty<'tcx>) -> Ty<'tcx> {
        debug!("resolve_type_vars_with_obligations(ty={:?})", ty);

        // No TyInfer()? Nothing needs doing.
        if !ty.has_infer_types() {
            debug!("resolve_type_vars_with_obligations: ty={:?}", ty);
            return ty;
        }

        // If `ty` is a type variable, see whether we already know what it is.
        ty = self.resolve_type_vars_if_possible(&ty);
        if !ty.has_infer_types() {
            debug!("resolve_type_vars_with_obligations: ty={:?}", ty);
            return ty;
        }

        // If not, try resolving pending obligations as much as
        // possible. This can help substantially when there are
        // indirect dependencies that don't seem worth tracking
        // precisely.
        self.select_obligations_where_possible();
        ty = self.resolve_type_vars_if_possible(&ty);

        debug!("resolve_type_vars_with_obligations: ty={:?}", ty);
        ty
    }

    fn record_deferred_call_resolution(&self,
                                       closure_def_id: DefId,
                                       r: DeferredCallResolutionHandler<'gcx, 'tcx>) {
        let mut deferred_call_resolutions = self.deferred_call_resolutions.borrow_mut();
        deferred_call_resolutions.entry(closure_def_id).or_insert(vec![]).push(r);
    }

    fn remove_deferred_call_resolutions(&self,
                                        closure_def_id: DefId)
                                        -> Vec<DeferredCallResolutionHandler<'gcx, 'tcx>>
    {
        let mut deferred_call_resolutions = self.deferred_call_resolutions.borrow_mut();
        deferred_call_resolutions.remove(&closure_def_id).unwrap_or(Vec::new())
    }

    pub fn tag(&self) -> String {
        let self_ptr: *const FnCtxt = self;
        format!("{:?}", self_ptr)
    }

    pub fn local_ty(&self, span: Span, nid: ast::NodeId) -> Ty<'tcx> {
        match self.locals.borrow().get(&nid) {
            Some(&t) => t,
            None => {
                span_bug!(span, "no type for local variable {}",
                          self.tcx.map.node_to_string(nid));
            }
        }
    }

    #[inline]
    pub fn write_ty(&self, node_id: ast::NodeId, ty: Ty<'tcx>) {
        debug!("write_ty({}, {:?}) in fcx {}",
               node_id, ty, self.tag());
        self.tables.borrow_mut().node_types.insert(node_id, ty);

        if ty.references_error() {
            self.has_errors.set(true);
        }

        // FIXME(canndrew): This is_never should probably be an is_uninhabited
        if ty.is_never() || self.type_var_diverges(ty) {
            self.diverges.set(self.diverges.get() | Diverges::Always);
        }
    }

    pub fn write_substs(&self, node_id: ast::NodeId, substs: ty::ItemSubsts<'tcx>) {
        if !substs.substs.is_noop() {
            debug!("write_substs({}, {:?}) in fcx {}",
                   node_id,
                   substs,
                   self.tag());

            self.tables.borrow_mut().item_substs.insert(node_id, substs);
        }
    }

    pub fn write_autoderef_adjustment(&self,
                                      node_id: ast::NodeId,
                                      derefs: usize,
                                      adjusted_ty: Ty<'tcx>) {
        self.write_adjustment(node_id, adjustment::Adjustment {
            kind: adjustment::Adjust::DerefRef {
                autoderefs: derefs,
                autoref: None,
                unsize: false
            },
            target: adjusted_ty
        });
    }

    pub fn write_adjustment(&self,
                            node_id: ast::NodeId,
                            adj: adjustment::Adjustment<'tcx>) {
        debug!("write_adjustment(node_id={}, adj={:?})", node_id, adj);

        if adj.is_identity() {
            return;
        }

        self.tables.borrow_mut().adjustments.insert(node_id, adj);
    }

    /// Basically whenever we are converting from a type scheme into
    /// the fn body space, we always want to normalize associated
    /// types as well. This function combines the two.
    fn instantiate_type_scheme<T>(&self,
                                  span: Span,
                                  substs: &Substs<'tcx>,
                                  value: &T)
                                  -> T
        where T : TypeFoldable<'tcx>
    {
        let value = value.subst(self.tcx, substs);
        let result = self.normalize_associated_types_in(span, &value);
        debug!("instantiate_type_scheme(value={:?}, substs={:?}) = {:?}",
               value,
               substs,
               result);
        result
    }

    /// As `instantiate_type_scheme`, but for the bounds found in a
    /// generic type scheme.
    fn instantiate_bounds(&self, span: Span, def_id: DefId, substs: &Substs<'tcx>)
                          -> ty::InstantiatedPredicates<'tcx> {
        let bounds = self.tcx.item_predicates(def_id);
        let result = bounds.instantiate(self.tcx, substs);
        let result = self.normalize_associated_types_in(span, &result.predicates);
        debug!("instantiate_bounds(bounds={:?}, substs={:?}) = {:?}",
               bounds,
               substs,
               result);
        ty::InstantiatedPredicates {
            predicates: result
        }
    }

    /// Replace all anonymized types with fresh inference variables
    /// and record them for writeback.
    fn instantiate_anon_types<T: TypeFoldable<'tcx>>(&self, value: &T) -> T {
        value.fold_with(&mut BottomUpFolder { tcx: self.tcx, fldop: |ty| {
            if let ty::TyAnon(def_id, substs) = ty.sty {
                // Use the same type variable if the exact same TyAnon appears more
                // than once in the return type (e.g. if it's pased to a type alias).
                if let Some(ty_var) = self.anon_types.borrow().get(&def_id) {
                    return ty_var;
                }
                let span = self.tcx.def_span(def_id);
                let ty_var = self.next_ty_var(TypeVariableOrigin::TypeInference(span));
                self.anon_types.borrow_mut().insert(def_id, ty_var);

                let item_predicates = self.tcx.item_predicates(def_id);
                let bounds = item_predicates.instantiate(self.tcx, substs);

                for predicate in bounds.predicates {
                    // Change the predicate to refer to the type variable,
                    // which will be the concrete type, instead of the TyAnon.
                    // This also instantiates nested `impl Trait`.
                    let predicate = self.instantiate_anon_types(&predicate);

                    // Require that the predicate holds for the concrete type.
                    let cause = traits::ObligationCause::new(span, self.body_id,
                                                             traits::ReturnType);
                    self.register_predicate(traits::Obligation::new(cause, predicate));
                }

                ty_var
            } else {
                ty
            }
        }})
    }

    fn normalize_associated_types_in<T>(&self, span: Span, value: &T) -> T
        where T : TypeFoldable<'tcx>
    {
        self.inh.normalize_associated_types_in(span, self.body_id, value)
    }

    fn normalize_associated_type(&self,
                                 span: Span,
                                 trait_ref: ty::TraitRef<'tcx>,
                                 item_name: ast::Name)
                                 -> Ty<'tcx>
    {
        let cause = traits::ObligationCause::new(span,
                                                 self.body_id,
                                                 traits::ObligationCauseCode::MiscObligation);
        self.fulfillment_cx
            .borrow_mut()
            .normalize_projection_type(self,
                                       ty::ProjectionTy {
                                           trait_ref: trait_ref,
                                           item_name: item_name,
                                       },
                                       cause)
    }

    pub fn write_nil(&self, node_id: ast::NodeId) {
        self.write_ty(node_id, self.tcx.mk_nil());
    }

    pub fn write_never(&self, node_id: ast::NodeId) {
        self.write_ty(node_id, self.tcx.types.never);
    }

    pub fn write_error(&self, node_id: ast::NodeId) {
        self.write_ty(node_id, self.tcx.types.err);
    }

    pub fn require_type_meets(&self,
                              ty: Ty<'tcx>,
                              span: Span,
                              code: traits::ObligationCauseCode<'tcx>,
                              def_id: DefId)
    {
        self.register_bound(
            ty,
            def_id,
            traits::ObligationCause::new(span, self.body_id, code));
    }

    pub fn require_type_is_sized(&self,
                                 ty: Ty<'tcx>,
                                 span: Span,
                                 code: traits::ObligationCauseCode<'tcx>)
    {
        let lang_item = self.tcx.require_lang_item(lang_items::SizedTraitLangItem);
        self.require_type_meets(ty, span, code, lang_item);
    }

    pub fn register_bound(&self,
                                  ty: Ty<'tcx>,
                                  def_id: DefId,
                                  cause: traits::ObligationCause<'tcx>)
    {
        self.fulfillment_cx.borrow_mut()
            .register_bound(self, ty, def_id, cause);
    }

    pub fn register_predicate(&self,
                              obligation: traits::PredicateObligation<'tcx>)
    {
        debug!("register_predicate({:?})",
               obligation);
        self.fulfillment_cx
            .borrow_mut()
            .register_predicate_obligation(self, obligation);
    }

    pub fn register_predicates(&self,
                               obligations: Vec<traits::PredicateObligation<'tcx>>)
    {
        for obligation in obligations {
            self.register_predicate(obligation);
        }
    }

    pub fn register_infer_ok_obligations<T>(&self, infer_ok: InferOk<'tcx, T>) -> T {
        self.register_predicates(infer_ok.obligations);
        infer_ok.value
    }

    pub fn to_ty(&self, ast_t: &hir::Ty) -> Ty<'tcx> {
        let t = AstConv::ast_ty_to_ty(self, self, ast_t);
        self.register_wf_obligation(t, ast_t.span, traits::MiscObligation);
        t
    }

    pub fn node_ty(&self, id: ast::NodeId) -> Ty<'tcx> {
        match self.tables.borrow().node_types.get(&id) {
            Some(&t) => t,
            None if self.err_count_since_creation() != 0 => self.tcx.types.err,
            None => {
                bug!("no type for node {}: {} in fcx {}",
                     id, self.tcx.map.node_to_string(id),
                     self.tag());
            }
        }
    }

    pub fn opt_node_ty_substs<F>(&self,
                                 id: ast::NodeId,
                                 f: F) where
        F: FnOnce(&ty::ItemSubsts<'tcx>),
    {
        if let Some(s) = self.tables.borrow().item_substs.get(&id) {
            f(s);
        }
    }

    /// Registers an obligation for checking later, during regionck, that the type `ty` must
    /// outlive the region `r`.
    pub fn register_region_obligation(&self,
                                      ty: Ty<'tcx>,
                                      region: &'tcx ty::Region,
                                      cause: traits::ObligationCause<'tcx>)
    {
        let mut fulfillment_cx = self.fulfillment_cx.borrow_mut();
        fulfillment_cx.register_region_obligation(ty, region, cause);
    }

    /// Registers an obligation for checking later, during regionck, that the type `ty` must
    /// outlive the region `r`.
    pub fn register_wf_obligation(&self,
                                  ty: Ty<'tcx>,
                                  span: Span,
                                  code: traits::ObligationCauseCode<'tcx>)
    {
        // WF obligations never themselves fail, so no real need to give a detailed cause:
        let cause = traits::ObligationCause::new(span, self.body_id, code);
        self.register_predicate(traits::Obligation::new(cause, ty::Predicate::WellFormed(ty)));
    }

    pub fn register_old_wf_obligation(&self,
                                      ty: Ty<'tcx>,
                                      span: Span,
                                      code: traits::ObligationCauseCode<'tcx>)
    {
        // Registers an "old-style" WF obligation that uses the
        // implicator code.  This is basically a buggy version of
        // `register_wf_obligation` that is being kept around
        // temporarily just to help with phasing in the newer rules.
        //
        // FIXME(#27579) all uses of this should be migrated to register_wf_obligation eventually
        let cause = traits::ObligationCause::new(span, self.body_id, code);
        self.register_region_obligation(ty, self.tcx.mk_region(ty::ReEmpty), cause);
    }

    /// Registers obligations that all types appearing in `substs` are well-formed.
    pub fn add_wf_bounds(&self, substs: &Substs<'tcx>, expr: &hir::Expr)
    {
        for ty in substs.types() {
            self.register_wf_obligation(ty, expr.span, traits::MiscObligation);
        }
    }

    /// Given a fully substituted set of bounds (`generic_bounds`), and the values with which each
    /// type/region parameter was instantiated (`substs`), creates and registers suitable
    /// trait/region obligations.
    ///
    /// For example, if there is a function:
    ///
    /// ```
    /// fn foo<'a,T:'a>(...)
    /// ```
    ///
    /// and a reference:
    ///
    /// ```
    /// let f = foo;
    /// ```
    ///
    /// Then we will create a fresh region variable `'$0` and a fresh type variable `$1` for `'a`
    /// and `T`. This routine will add a region obligation `$1:'$0` and register it locally.
    pub fn add_obligations_for_parameters(&self,
                                          cause: traits::ObligationCause<'tcx>,
                                          predicates: &ty::InstantiatedPredicates<'tcx>)
    {
        assert!(!predicates.has_escaping_regions());

        debug!("add_obligations_for_parameters(predicates={:?})",
               predicates);

        for obligation in traits::predicates_for_generics(cause, predicates) {
            self.register_predicate(obligation);
        }
    }

    // FIXME(arielb1): use this instead of field.ty everywhere
    // Only for fields! Returns <none> for methods>
    // Indifferent to privacy flags
    pub fn field_ty(&self,
                    span: Span,
                    field: &'tcx ty::FieldDef,
                    substs: &Substs<'tcx>)
                    -> Ty<'tcx>
    {
        self.normalize_associated_types_in(span,
                                           &field.ty(self.tcx, substs))
    }

    fn check_casts(&self) {
        let mut deferred_cast_checks = self.deferred_cast_checks.borrow_mut();
        for cast in deferred_cast_checks.drain(..) {
            cast.check(self);
        }
    }

    /// Apply "fallbacks" to some types
    /// ! gets replaced with (), unconstrained ints with i32, and unconstrained floats with f64.
    fn default_type_parameters(&self) {
        use rustc::ty::error::UnconstrainedNumeric::Neither;
        use rustc::ty::error::UnconstrainedNumeric::{UnconstrainedInt, UnconstrainedFloat};

        // Defaulting inference variables becomes very dubious if we have
        // encountered type-checking errors. Therefore, if we think we saw
        // some errors in this function, just resolve all uninstanted type
        // varibles to TyError.
        if self.is_tainted_by_errors() {
            for ty in &self.unsolved_variables() {
                if let ty::TyInfer(_) = self.shallow_resolve(ty).sty {
                    debug!("default_type_parameters: defaulting `{:?}` to error", ty);
                    self.demand_eqtype(syntax_pos::DUMMY_SP, *ty, self.tcx().types.err);
                }
            }
            return;
        }

        for ty in &self.unsolved_variables() {
            let resolved = self.resolve_type_vars_if_possible(ty);
            if self.type_var_diverges(resolved) {
                debug!("default_type_parameters: defaulting `{:?}` to `!` because it diverges",
                       resolved);
                self.demand_eqtype(syntax_pos::DUMMY_SP, *ty,
                                   self.tcx.mk_diverging_default());
            } else {
                match self.type_is_unconstrained_numeric(resolved) {
                    UnconstrainedInt => {
                        debug!("default_type_parameters: defaulting `{:?}` to `i32`",
                               resolved);
                        self.demand_eqtype(syntax_pos::DUMMY_SP, *ty, self.tcx.types.i32)
                    },
                    UnconstrainedFloat => {
                        debug!("default_type_parameters: defaulting `{:?}` to `f32`",
                               resolved);
                        self.demand_eqtype(syntax_pos::DUMMY_SP, *ty, self.tcx.types.f64)
                    }
                    Neither => { }
                }
            }
        }
    }

    fn select_all_obligations_and_apply_defaults(&self) {
        if self.tcx.sess.features.borrow().default_type_parameter_fallback {
            self.new_select_all_obligations_and_apply_defaults();
        } else {
            self.old_select_all_obligations_and_apply_defaults();
        }
    }

    // Implements old type inference fallback algorithm
    fn old_select_all_obligations_and_apply_defaults(&self) {
        self.select_obligations_where_possible();
        self.default_type_parameters();
        self.select_obligations_where_possible();
    }

    fn new_select_all_obligations_and_apply_defaults(&self) {
        use rustc::ty::error::UnconstrainedNumeric::Neither;
        use rustc::ty::error::UnconstrainedNumeric::{UnconstrainedInt, UnconstrainedFloat};

        // For the time being this errs on the side of being memory wasteful but provides better
        // error reporting.
        // let type_variables = self.type_variables.clone();

        // There is a possibility that this algorithm will have to run an arbitrary number of times
        // to terminate so we bound it by the compiler's recursion limit.
        for _ in 0..self.tcx.sess.recursion_limit.get() {
            // First we try to solve all obligations, it is possible that the last iteration
            // has made it possible to make more progress.
            self.select_obligations_where_possible();

            let mut conflicts = Vec::new();

            // Collect all unsolved type, integral and floating point variables.
            let unsolved_variables = self.unsolved_variables();

            // We must collect the defaults *before* we do any unification. Because we have
            // directly attached defaults to the type variables any unification that occurs
            // will erase defaults causing conflicting defaults to be completely ignored.
            let default_map: FxHashMap<_, _> =
                unsolved_variables
                    .iter()
                    .filter_map(|t| self.default(t).map(|d| (t, d)))
                    .collect();

            let mut unbound_tyvars = FxHashSet();

            debug!("select_all_obligations_and_apply_defaults: defaults={:?}", default_map);

            // We loop over the unsolved variables, resolving them and if they are
            // and unconstrainted numeric type we add them to the set of unbound
            // variables. We do this so we only apply literal fallback to type
            // variables without defaults.
            for ty in &unsolved_variables {
                let resolved = self.resolve_type_vars_if_possible(ty);
                if self.type_var_diverges(resolved) {
                    self.demand_eqtype(syntax_pos::DUMMY_SP, *ty,
                                       self.tcx.mk_diverging_default());
                } else {
                    match self.type_is_unconstrained_numeric(resolved) {
                        UnconstrainedInt | UnconstrainedFloat => {
                            unbound_tyvars.insert(resolved);
                        },
                        Neither => {}
                    }
                }
            }

            // We now remove any numeric types that also have defaults, and instead insert
            // the type variable with a defined fallback.
            for ty in &unsolved_variables {
                if let Some(_default) = default_map.get(ty) {
                    let resolved = self.resolve_type_vars_if_possible(ty);

                    debug!("select_all_obligations_and_apply_defaults: \
                            ty: {:?} with default: {:?}",
                             ty, _default);

                    match resolved.sty {
                        ty::TyInfer(ty::TyVar(_)) => {
                            unbound_tyvars.insert(ty);
                        }

                        ty::TyInfer(ty::IntVar(_)) | ty::TyInfer(ty::FloatVar(_)) => {
                            unbound_tyvars.insert(ty);
                            if unbound_tyvars.contains(resolved) {
                                unbound_tyvars.remove(resolved);
                            }
                        }

                        _ => {}
                    }
                }
            }

            // If there are no more fallbacks to apply at this point we have applied all possible
            // defaults and type inference will proceed as normal.
            if unbound_tyvars.is_empty() {
                break;
            }

            // Finally we go through each of the unbound type variables and unify them with
            // the proper fallback, reporting a conflicting default error if any of the
            // unifications fail. We know it must be a conflicting default because the
            // variable would only be in `unbound_tyvars` and have a concrete value if
            // it had been solved by previously applying a default.

            // We wrap this in a transaction for error reporting, if we detect a conflict
            // we will rollback the inference context to its prior state so we can probe
            // for conflicts and correctly report them.


            let _ = self.commit_if_ok(|_: &infer::CombinedSnapshot| {
                for ty in &unbound_tyvars {
                    if self.type_var_diverges(ty) {
                        self.demand_eqtype(syntax_pos::DUMMY_SP, *ty,
                                           self.tcx.mk_diverging_default());
                    } else {
                        match self.type_is_unconstrained_numeric(ty) {
                            UnconstrainedInt => {
                                self.demand_eqtype(syntax_pos::DUMMY_SP, *ty, self.tcx.types.i32)
                            },
                            UnconstrainedFloat => {
                                self.demand_eqtype(syntax_pos::DUMMY_SP, *ty, self.tcx.types.f64)
                            }
                            Neither => {
                                if let Some(default) = default_map.get(ty) {
                                    let default = default.clone();
                                    match self.eq_types(false,
                                                        &self.misc(default.origin_span),
                                                        ty,
                                                        default.ty) {
                                        Ok(ok) => self.register_infer_ok_obligations(ok),
                                        Err(_) => conflicts.push((*ty, default)),
                                    }
                                }
                            }
                        }
                    }
                }

                // If there are conflicts we rollback, otherwise commit
                if conflicts.len() > 0 {
                    Err(())
                } else {
                    Ok(())
                }
            });

            if conflicts.len() > 0 {
                // Loop through each conflicting default, figuring out the default that caused
                // a unification failure and then report an error for each.
                for (conflict, default) in conflicts {
                    let conflicting_default =
                        self.find_conflicting_default(&unbound_tyvars, &default_map, conflict)
                            .unwrap_or(type_variable::Default {
                                ty: self.next_ty_var(
                                    TypeVariableOrigin::MiscVariable(syntax_pos::DUMMY_SP)),
                                origin_span: syntax_pos::DUMMY_SP,
                                // what do I put here?
                                def_id: self.tcx.map.local_def_id(ast::CRATE_NODE_ID)
                            });

                    // This is to ensure that we elimnate any non-determinism from the error
                    // reporting by fixing an order, it doesn't matter what order we choose
                    // just that it is consistent.
                    let (first_default, second_default) =
                        if default.def_id < conflicting_default.def_id {
                            (default, conflicting_default)
                        } else {
                            (conflicting_default, default)
                        };


                    self.report_conflicting_default_types(
                        first_default.origin_span,
                        self.body_id,
                        first_default,
                        second_default)
                }
            }
        }

        self.select_obligations_where_possible();
    }

    // For use in error handling related to default type parameter fallback. We explicitly
    // apply the default that caused conflict first to a local version of the type variable
    // table then apply defaults until we find a conflict. That default must be the one
    // that caused conflict earlier.
    fn find_conflicting_default(&self,
                                unbound_vars: &FxHashSet<Ty<'tcx>>,
                                default_map: &FxHashMap<&Ty<'tcx>, type_variable::Default<'tcx>>,
                                conflict: Ty<'tcx>)
                                -> Option<type_variable::Default<'tcx>> {
        use rustc::ty::error::UnconstrainedNumeric::Neither;
        use rustc::ty::error::UnconstrainedNumeric::{UnconstrainedInt, UnconstrainedFloat};

        // Ensure that we apply the conflicting default first
        let mut unbound_tyvars = Vec::with_capacity(unbound_vars.len() + 1);
        unbound_tyvars.push(conflict);
        unbound_tyvars.extend(unbound_vars.iter());

        let mut result = None;
        // We run the same code as above applying defaults in order, this time when
        // we find the conflict we just return it for error reporting above.

        // We also run this inside snapshot that never commits so we can do error
        // reporting for more then one conflict.
        for ty in &unbound_tyvars {
            if self.type_var_diverges(ty) {
                self.demand_eqtype(syntax_pos::DUMMY_SP, *ty,
                                   self.tcx.mk_diverging_default());
            } else {
                match self.type_is_unconstrained_numeric(ty) {
                    UnconstrainedInt => {
                        self.demand_eqtype(syntax_pos::DUMMY_SP, *ty, self.tcx.types.i32)
                    },
                    UnconstrainedFloat => {
                        self.demand_eqtype(syntax_pos::DUMMY_SP, *ty, self.tcx.types.f64)
                    },
                    Neither => {
                        if let Some(default) = default_map.get(ty) {
                            let default = default.clone();
                            match self.eq_types(false,
                                                &self.misc(default.origin_span),
                                                ty,
                                                default.ty) {
                                Ok(ok) => self.register_infer_ok_obligations(ok),
                                Err(_) => {
                                    result = Some(default);
                                }
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    fn select_all_obligations_or_error(&self) {
        debug!("select_all_obligations_or_error");

        // upvar inference should have ensured that all deferred call
        // resolutions are handled by now.
        assert!(self.deferred_call_resolutions.borrow().is_empty());

        self.select_all_obligations_and_apply_defaults();

        let mut fulfillment_cx = self.fulfillment_cx.borrow_mut();

        // Steal the deferred obligations before the fulfillment
        // context can turn all of them into errors.
        let obligations = fulfillment_cx.take_deferred_obligations();
        self.deferred_obligations.borrow_mut().extend(obligations);

        match fulfillment_cx.select_all_or_error(self) {
            Ok(()) => { }
            Err(errors) => { self.report_fulfillment_errors(&errors); }
        }
    }

    /// Select as many obligations as we can at present.
    fn select_obligations_where_possible(&self) {
        match self.fulfillment_cx.borrow_mut().select_where_possible(self) {
            Ok(()) => { }
            Err(errors) => { self.report_fulfillment_errors(&errors); }
        }
    }

    /// For the overloaded lvalue expressions (`*x`, `x[3]`), the trait
    /// returns a type of `&T`, but the actual type we assign to the
    /// *expression* is `T`. So this function just peels off the return
    /// type by one layer to yield `T`.
    fn make_overloaded_lvalue_return_type(&self,
                                          method: MethodCallee<'tcx>)
                                          -> ty::TypeAndMut<'tcx>
    {
        // extract method return type, which will be &T;
        // all LB regions should have been instantiated during method lookup
        let ret_ty = method.ty.fn_ret();
        let ret_ty = self.tcx.no_late_bound_regions(&ret_ty).unwrap();

        // method returns &T, but the type as visible to user is T, so deref
        ret_ty.builtin_deref(true, NoPreference).unwrap()
    }

    fn lookup_indexing(&self,
                       expr: &hir::Expr,
                       base_expr: &'gcx hir::Expr,
                       base_ty: Ty<'tcx>,
                       idx_ty: Ty<'tcx>,
                       lvalue_pref: LvaluePreference)
                       -> Option<(/*index type*/ Ty<'tcx>, /*element type*/ Ty<'tcx>)>
    {
        // FIXME(#18741) -- this is almost but not quite the same as the
        // autoderef that normal method probing does. They could likely be
        // consolidated.

        let mut autoderef = self.autoderef(base_expr.span, base_ty);

        while let Some((adj_ty, autoderefs)) = autoderef.next() {
            if let Some(final_mt) = self.try_index_step(
                MethodCall::expr(expr.id),
                expr, base_expr, adj_ty, autoderefs,
                false, lvalue_pref, idx_ty)
            {
                autoderef.finalize(lvalue_pref, Some(base_expr));
                return Some(final_mt);
            }

            if let ty::TyArray(element_ty, _) = adj_ty.sty {
                autoderef.finalize(lvalue_pref, Some(base_expr));
                let adjusted_ty = self.tcx.mk_slice(element_ty);
                return self.try_index_step(
                    MethodCall::expr(expr.id), expr, base_expr,
                    adjusted_ty, autoderefs, true, lvalue_pref, idx_ty);
            }
        }
        autoderef.unambiguous_final_ty();
        None
    }

    /// To type-check `base_expr[index_expr]`, we progressively autoderef
    /// (and otherwise adjust) `base_expr`, looking for a type which either
    /// supports builtin indexing or overloaded indexing.
    /// This loop implements one step in that search; the autoderef loop
    /// is implemented by `lookup_indexing`.
    fn try_index_step(&self,
                      method_call: MethodCall,
                      expr: &hir::Expr,
                      base_expr: &'gcx hir::Expr,
                      adjusted_ty: Ty<'tcx>,
                      autoderefs: usize,
                      unsize: bool,
                      lvalue_pref: LvaluePreference,
                      index_ty: Ty<'tcx>)
                      -> Option<(/*index type*/ Ty<'tcx>, /*element type*/ Ty<'tcx>)>
    {
        let tcx = self.tcx;
        debug!("try_index_step(expr={:?}, base_expr.id={:?}, adjusted_ty={:?}, \
                               autoderefs={}, unsize={}, index_ty={:?})",
               expr,
               base_expr,
               adjusted_ty,
               autoderefs,
               unsize,
               index_ty);

        let input_ty = self.next_ty_var(TypeVariableOrigin::AutoDeref(base_expr.span));

        // First, try built-in indexing.
        match (adjusted_ty.builtin_index(), &index_ty.sty) {
            (Some(ty), &ty::TyUint(ast::UintTy::Us)) | (Some(ty), &ty::TyInfer(ty::IntVar(_))) => {
                debug!("try_index_step: success, using built-in indexing");
                // If we had `[T; N]`, we should've caught it before unsizing to `[T]`.
                assert!(!unsize);
                self.write_autoderef_adjustment(base_expr.id, autoderefs, adjusted_ty);
                return Some((tcx.types.usize, ty));
            }
            _ => {}
        }

        // Try `IndexMut` first, if preferred.
        let method = match (lvalue_pref, tcx.lang_items.index_mut_trait()) {
            (PreferMutLvalue, Some(trait_did)) => {
                self.lookup_method_in_trait_adjusted(expr.span,
                                                     Some(&base_expr),
                                                     Symbol::intern("index_mut"),
                                                     trait_did,
                                                     autoderefs,
                                                     unsize,
                                                     adjusted_ty,
                                                     Some(vec![input_ty]))
            }
            _ => None,
        };

        // Otherwise, fall back to `Index`.
        let method = match (method, tcx.lang_items.index_trait()) {
            (None, Some(trait_did)) => {
                self.lookup_method_in_trait_adjusted(expr.span,
                                                     Some(&base_expr),
                                                     Symbol::intern("index"),
                                                     trait_did,
                                                     autoderefs,
                                                     unsize,
                                                     adjusted_ty,
                                                     Some(vec![input_ty]))
            }
            (method, _) => method,
        };

        // If some lookup succeeds, write callee into table and extract index/element
        // type from the method signature.
        // If some lookup succeeded, install method in table
        method.map(|method| {
            debug!("try_index_step: success, using overloaded indexing");
            self.tables.borrow_mut().method_map.insert(method_call, method);
            (input_ty, self.make_overloaded_lvalue_return_type(method).ty)
        })
    }

    fn check_method_argument_types(&self,
                                   sp: Span,
                                   method_fn_ty: Ty<'tcx>,
                                   callee_expr: &'gcx hir::Expr,
                                   args_no_rcvr: &'gcx [hir::Expr],
                                   tuple_arguments: TupleArgumentsFlag,
                                   expected: Expectation<'tcx>)
                                   -> Ty<'tcx> {
        if method_fn_ty.references_error() {
            let err_inputs = self.err_args(args_no_rcvr.len());

            let err_inputs = match tuple_arguments {
                DontTupleArguments => err_inputs,
                TupleArguments => vec![self.tcx.intern_tup(&err_inputs[..])],
            };

            self.check_argument_types(sp, &err_inputs[..], &[], args_no_rcvr,
                                      false, tuple_arguments, None);
            self.tcx.types.err
        } else {
            match method_fn_ty.sty {
                ty::TyFnDef(def_id, .., ref fty) => {
                    // HACK(eddyb) ignore self in the definition (see above).
                    let expected_arg_tys = self.expected_types_for_fn_args(
                        sp,
                        expected,
                        fty.sig.0.output(),
                        &fty.sig.0.inputs()[1..]
                    );
                    self.check_argument_types(sp, &fty.sig.0.inputs()[1..], &expected_arg_tys[..],
                                              args_no_rcvr, fty.sig.0.variadic, tuple_arguments,
                                              self.tcx.map.span_if_local(def_id));
                    fty.sig.0.output()
                }
                _ => {
                    span_bug!(callee_expr.span, "method without bare fn type");
                }
            }
        }
    }

    /// Generic function that factors out common logic from function calls,
    /// method calls and overloaded operators.
    fn check_argument_types(&self,
                            sp: Span,
                            fn_inputs: &[Ty<'tcx>],
                            expected_arg_tys: &[Ty<'tcx>],
                            args: &'gcx [hir::Expr],
                            variadic: bool,
                            tuple_arguments: TupleArgumentsFlag,
                            def_span: Option<Span>) {
        let tcx = self.tcx;

        // Grab the argument types, supplying fresh type variables
        // if the wrong number of arguments were supplied
        let supplied_arg_count = if tuple_arguments == DontTupleArguments {
            args.len()
        } else {
            1
        };

        // All the input types from the fn signature must outlive the call
        // so as to validate implied bounds.
        for &fn_input_ty in fn_inputs {
            self.register_wf_obligation(fn_input_ty, sp, traits::MiscObligation);
        }

        let mut expected_arg_tys = expected_arg_tys;
        let expected_arg_count = fn_inputs.len();

        let sp_args = if args.len() > 0 {
            let (first, args) = args.split_at(1);
            let mut sp_tmp = first[0].span;
            for arg in args {
                let sp_opt = self.sess().codemap().merge_spans(sp_tmp, arg.span);
                if ! sp_opt.is_some() {
                    break;
                }
                sp_tmp = sp_opt.unwrap();
            };
            sp_tmp
        } else {
            sp
        };

        fn parameter_count_error<'tcx>(sess: &Session, sp: Span, expected_count: usize,
                                       arg_count: usize, error_code: &str, variadic: bool,
                                       def_span: Option<Span>) {
            let mut err = sess.struct_span_err_with_code(sp,
                &format!("this function takes {}{} parameter{} but {} parameter{} supplied",
                    if variadic {"at least "} else {""},
                    expected_count,
                    if expected_count == 1 {""} else {"s"},
                    arg_count,
                    if arg_count == 1 {" was"} else {"s were"}),
                error_code);

            err.span_label(sp, &format!("expected {}{} parameter{}",
                                        if variadic {"at least "} else {""},
                                        expected_count,
                                        if expected_count == 1 {""} else {"s"}));
            if let Some(def_s) = def_span {
                err.span_label(def_s, &format!("defined here"));
            }
            err.emit();
        }

        let formal_tys = if tuple_arguments == TupleArguments {
            let tuple_type = self.structurally_resolved_type(sp, fn_inputs[0]);
            match tuple_type.sty {
                ty::TyTuple(arg_types) if arg_types.len() != args.len() => {
                    parameter_count_error(tcx.sess, sp_args, arg_types.len(), args.len(),
                                          "E0057", false, def_span);
                    expected_arg_tys = &[];
                    self.err_args(args.len())
                }
                ty::TyTuple(arg_types) => {
                    expected_arg_tys = match expected_arg_tys.get(0) {
                        Some(&ty) => match ty.sty {
                            ty::TyTuple(ref tys) => &tys,
                            _ => &[]
                        },
                        None => &[]
                    };
                    arg_types.to_vec()
                }
                _ => {
                    span_err!(tcx.sess, sp, E0059,
                        "cannot use call notation; the first type parameter \
                         for the function trait is neither a tuple nor unit");
                    expected_arg_tys = &[];
                    self.err_args(args.len())
                }
            }
        } else if expected_arg_count == supplied_arg_count {
            fn_inputs.to_vec()
        } else if variadic {
            if supplied_arg_count >= expected_arg_count {
                fn_inputs.to_vec()
            } else {
                parameter_count_error(tcx.sess, sp_args, expected_arg_count,
                                      supplied_arg_count, "E0060", true, def_span);
                expected_arg_tys = &[];
                self.err_args(supplied_arg_count)
            }
        } else {
            parameter_count_error(tcx.sess, sp_args, expected_arg_count,
                                  supplied_arg_count, "E0061", false, def_span);
            expected_arg_tys = &[];
            self.err_args(supplied_arg_count)
        };

        debug!("check_argument_types: formal_tys={:?}",
               formal_tys.iter().map(|t| self.ty_to_string(*t)).collect::<Vec<String>>());

        // Check the arguments.
        // We do this in a pretty awful way: first we typecheck any arguments
        // that are not closures, then we typecheck the closures. This is so
        // that we have more information about the types of arguments when we
        // typecheck the functions. This isn't really the right way to do this.
        for &check_closures in &[false, true] {
            debug!("check_closures={}", check_closures);

            // More awful hacks: before we check argument types, try to do
            // an "opportunistic" vtable resolution of any trait bounds on
            // the call. This helps coercions.
            if check_closures {
                self.select_obligations_where_possible();
            }

            // For variadic functions, we don't have a declared type for all of
            // the arguments hence we only do our usual type checking with
            // the arguments who's types we do know.
            let t = if variadic {
                expected_arg_count
            } else if tuple_arguments == TupleArguments {
                args.len()
            } else {
                supplied_arg_count
            };
            for (i, arg) in args.iter().take(t).enumerate() {
                // Warn only for the first loop (the "no closures" one).
                // Closure arguments themselves can't be diverging, but
                // a previous argument can, e.g. `foo(panic!(), || {})`.
                if !check_closures {
                    self.warn_if_unreachable(arg.id, arg.span, "expression");
                }

                let is_closure = match arg.node {
                    hir::ExprClosure(..) => true,
                    _ => false
                };

                if is_closure != check_closures {
                    continue;
                }

                debug!("checking the argument");
                let formal_ty = formal_tys[i];

                // The special-cased logic below has three functions:
                // 1. Provide as good of an expected type as possible.
                let expected = expected_arg_tys.get(i).map(|&ty| {
                    Expectation::rvalue_hint(self, ty)
                });

                let checked_ty = self.check_expr_with_expectation(&arg,
                                        expected.unwrap_or(ExpectHasType(formal_ty)));
                // 2. Coerce to the most detailed type that could be coerced
                //    to, which is `expected_ty` if `rvalue_hint` returns an
                //    `ExpectHasType(expected_ty)`, or the `formal_ty` otherwise.
                let coerce_ty = expected.and_then(|e| e.only_has_type(self));
                self.demand_coerce(&arg, checked_ty, coerce_ty.unwrap_or(formal_ty));

                // 3. Relate the expected type and the formal one,
                //    if the expected type was used for the coercion.
                coerce_ty.map(|ty| self.demand_suptype(arg.span, formal_ty, ty));
            }
        }

        // We also need to make sure we at least write the ty of the other
        // arguments which we skipped above.
        if variadic {
            for arg in args.iter().skip(expected_arg_count) {
                let arg_ty = self.check_expr(&arg);

                // There are a few types which get autopromoted when passed via varargs
                // in C but we just error out instead and require explicit casts.
                let arg_ty = self.structurally_resolved_type(arg.span,
                                                             arg_ty);
                match arg_ty.sty {
                    ty::TyFloat(ast::FloatTy::F32) => {
                        self.type_error_message(arg.span, |t| {
                            format!("can't pass an `{}` to variadic \
                                     function, cast to `c_double`", t)
                        }, arg_ty);
                    }
                    ty::TyInt(ast::IntTy::I8) | ty::TyInt(ast::IntTy::I16) | ty::TyBool => {
                        self.type_error_message(arg.span, |t| {
                            format!("can't pass `{}` to variadic \
                                     function, cast to `c_int`",
                                           t)
                        }, arg_ty);
                    }
                    ty::TyUint(ast::UintTy::U8) | ty::TyUint(ast::UintTy::U16) => {
                        self.type_error_message(arg.span, |t| {
                            format!("can't pass `{}` to variadic \
                                     function, cast to `c_uint`",
                                           t)
                        }, arg_ty);
                    }
                    ty::TyFnDef(.., f) => {
                        let ptr_ty = self.tcx.mk_fn_ptr(f);
                        let ptr_ty = self.resolve_type_vars_if_possible(&ptr_ty);
                        self.type_error_message(arg.span,
                                                |t| {
                            format!("can't pass `{}` to variadic \
                                     function, cast to `{}`", t, ptr_ty)
                        }, arg_ty);
                    }
                    _ => {}
                }
            }
        }
    }

    fn err_args(&self, len: usize) -> Vec<Ty<'tcx>> {
        (0..len).map(|_| self.tcx.types.err).collect()
    }

    // AST fragment checking
    fn check_lit(&self,
                 lit: &ast::Lit,
                 expected: Expectation<'tcx>)
                 -> Ty<'tcx>
    {
        let tcx = self.tcx;

        match lit.node {
            ast::LitKind::Str(..) => tcx.mk_static_str(),
            ast::LitKind::ByteStr(ref v) => {
                tcx.mk_imm_ref(tcx.mk_region(ty::ReStatic),
                                tcx.mk_array(tcx.types.u8, v.len()))
            }
            ast::LitKind::Byte(_) => tcx.types.u8,
            ast::LitKind::Char(_) => tcx.types.char,
            ast::LitKind::Int(_, ast::LitIntType::Signed(t)) => tcx.mk_mach_int(t),
            ast::LitKind::Int(_, ast::LitIntType::Unsigned(t)) => tcx.mk_mach_uint(t),
            ast::LitKind::Int(_, ast::LitIntType::Unsuffixed) => {
                let opt_ty = expected.to_option(self).and_then(|ty| {
                    match ty.sty {
                        ty::TyInt(_) | ty::TyUint(_) => Some(ty),
                        ty::TyChar => Some(tcx.types.u8),
                        ty::TyRawPtr(..) => Some(tcx.types.usize),
                        ty::TyFnDef(..) | ty::TyFnPtr(_) => Some(tcx.types.usize),
                        _ => None
                    }
                });
                opt_ty.unwrap_or_else(
                    || tcx.mk_int_var(self.next_int_var_id()))
            }
            ast::LitKind::Float(_, t) => tcx.mk_mach_float(t),
            ast::LitKind::FloatUnsuffixed(_) => {
                let opt_ty = expected.to_option(self).and_then(|ty| {
                    match ty.sty {
                        ty::TyFloat(_) => Some(ty),
                        _ => None
                    }
                });
                opt_ty.unwrap_or_else(
                    || tcx.mk_float_var(self.next_float_var_id()))
            }
            ast::LitKind::Bool(_) => tcx.types.bool
        }
    }

    fn check_expr_eq_type(&self,
                          expr: &'gcx hir::Expr,
                          expected: Ty<'tcx>) {
        let ty = self.check_expr_with_hint(expr, expected);
        self.demand_eqtype(expr.span, expected, ty);
    }

    pub fn check_expr_has_type(&self,
                               expr: &'gcx hir::Expr,
                               expected: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.check_expr_with_hint(expr, expected);
        self.demand_suptype(expr.span, expected, ty);
        ty
    }

    fn check_expr_coercable_to_type(&self,
                                    expr: &'gcx hir::Expr,
                                    expected: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.check_expr_with_hint(expr, expected);
        self.demand_coerce(expr, ty, expected);
        ty
    }

    fn check_expr_with_hint(&self, expr: &'gcx hir::Expr,
                            expected: Ty<'tcx>) -> Ty<'tcx> {
        self.check_expr_with_expectation(expr, ExpectHasType(expected))
    }

    fn check_expr_with_expectation(&self,
                                   expr: &'gcx hir::Expr,
                                   expected: Expectation<'tcx>) -> Ty<'tcx> {
        self.check_expr_with_expectation_and_lvalue_pref(expr, expected, NoPreference)
    }

    fn check_expr(&self, expr: &'gcx hir::Expr) -> Ty<'tcx> {
        self.check_expr_with_expectation(expr, NoExpectation)
    }

    fn check_expr_with_lvalue_pref(&self, expr: &'gcx hir::Expr,
                                   lvalue_pref: LvaluePreference) -> Ty<'tcx> {
        self.check_expr_with_expectation_and_lvalue_pref(expr, NoExpectation, lvalue_pref)
    }

    // determine the `self` type, using fresh variables for all variables
    // declared on the impl declaration e.g., `impl<A,B> for Vec<(A,B)>`
    // would return ($0, $1) where $0 and $1 are freshly instantiated type
    // variables.
    pub fn impl_self_ty(&self,
                        span: Span, // (potential) receiver for this impl
                        did: DefId)
                        -> TypeAndSubsts<'tcx> {
        let ity = self.tcx.item_type(did);
        debug!("impl_self_ty: ity={:?}", ity);

        let substs = self.fresh_substs_for_item(span, did);
        let substd_ty = self.instantiate_type_scheme(span, &substs, &ity);

        TypeAndSubsts { substs: substs, ty: substd_ty }
    }

    /// Unifies the return type with the expected type early, for more coercions
    /// and forward type information on the argument expressions.
    fn expected_types_for_fn_args(&self,
                                  call_span: Span,
                                  expected_ret: Expectation<'tcx>,
                                  formal_ret: Ty<'tcx>,
                                  formal_args: &[Ty<'tcx>])
                                  -> Vec<Ty<'tcx>> {
        let expected_args = expected_ret.only_has_type(self).and_then(|ret_ty| {
            self.fudge_regions_if_ok(&RegionVariableOrigin::Coercion(call_span), || {
                // Attempt to apply a subtyping relationship between the formal
                // return type (likely containing type variables if the function
                // is polymorphic) and the expected return type.
                // No argument expectations are produced if unification fails.
                let origin = self.misc(call_span);
                let ures = self.sub_types(false, &origin, formal_ret, ret_ty);
                // FIXME(#15760) can't use try! here, FromError doesn't default
                // to identity so the resulting type is not constrained.
                match ures {
                    Ok(ok) => self.register_infer_ok_obligations(ok),
                    Err(e) => return Err(e),
                }

                // Record all the argument types, with the substitutions
                // produced from the above subtyping unification.
                Ok(formal_args.iter().map(|ty| {
                    self.resolve_type_vars_if_possible(ty)
                }).collect())
            }).ok()
        }).unwrap_or(vec![]);
        debug!("expected_types_for_fn_args(formal={:?} -> {:?}, expected={:?} -> {:?})",
               formal_args, formal_ret,
               expected_args, expected_ret);
        expected_args
    }

    // Checks a method call.
    fn check_method_call(&self,
                         expr: &'gcx hir::Expr,
                         method_name: Spanned<ast::Name>,
                         args: &'gcx [hir::Expr],
                         tps: &[P<hir::Ty>],
                         expected: Expectation<'tcx>,
                         lvalue_pref: LvaluePreference) -> Ty<'tcx> {
        let rcvr = &args[0];
        let rcvr_t = self.check_expr_with_lvalue_pref(&rcvr, lvalue_pref);

        // no need to check for bot/err -- callee does that
        let expr_t = self.structurally_resolved_type(expr.span, rcvr_t);

        let tps = tps.iter().map(|ast_ty| self.to_ty(&ast_ty)).collect::<Vec<_>>();
        let fn_ty = match self.lookup_method(method_name.span,
                                             method_name.node,
                                             expr_t,
                                             tps,
                                             expr,
                                             rcvr) {
            Ok(method) => {
                let method_ty = method.ty;
                let method_call = MethodCall::expr(expr.id);
                self.tables.borrow_mut().method_map.insert(method_call, method);
                method_ty
            }
            Err(error) => {
                if method_name.node != keywords::Invalid.name() {
                    self.report_method_error(method_name.span,
                                             expr_t,
                                             method_name.node,
                                             Some(rcvr),
                                             error,
                                             Some(args));
                }
                self.write_error(expr.id);
                self.tcx.types.err
            }
        };

        // Call the generic checker.
        let ret_ty = self.check_method_argument_types(method_name.span, fn_ty,
                                                      expr, &args[1..],
                                                      DontTupleArguments,
                                                      expected);

        ret_ty
    }

    // A generic function for checking the then and else in an if
    // or if-else.
    fn check_then_else(&self,
                       cond_expr: &'gcx hir::Expr,
                       then_blk: &'gcx hir::Block,
                       opt_else_expr: Option<&'gcx hir::Expr>,
                       sp: Span,
                       expected: Expectation<'tcx>) -> Ty<'tcx> {
        let cond_ty = self.check_expr_has_type(cond_expr, self.tcx.types.bool);
        let cond_diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);

        let expected = expected.adjust_for_branches(self);
        let then_ty = self.check_block_with_expected(then_blk, expected);
        let then_diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);

        let unit = self.tcx.mk_nil();
        let (cause, expected_ty, found_ty, result);
        if let Some(else_expr) = opt_else_expr {
            let else_ty = self.check_expr_with_expectation(else_expr, expected);
            let else_diverges = self.diverges.get();
            cause = self.cause(sp, ObligationCauseCode::IfExpression);

            // Only try to coerce-unify if we have a then expression
            // to assign coercions to, otherwise it's () or diverging.
            expected_ty = then_ty;
            found_ty = else_ty;
            result = if let Some(ref then) = then_blk.expr {
                let res = self.try_find_coercion_lub(&cause, || Some(&**then),
                                                     then_ty, else_expr, else_ty);

                // In case we did perform an adjustment, we have to update
                // the type of the block, because old trans still uses it.
                if res.is_ok() {
                    let adj = self.tables.borrow().adjustments.get(&then.id).cloned();
                    if let Some(adj) = adj {
                        self.write_ty(then_blk.id, adj.target);
                    }
                }

                res
            } else {
                self.commit_if_ok(|_| {
                    let trace = TypeTrace::types(&cause, true, then_ty, else_ty);
                    self.lub(true, trace, &then_ty, &else_ty)
                        .map(|ok| self.register_infer_ok_obligations(ok))
                })
            };

            // We won't diverge unless both branches do (or the condition does).
            self.diverges.set(cond_diverges | then_diverges & else_diverges);
        } else {
            // If the condition is false we can't diverge.
            self.diverges.set(cond_diverges);

            cause = self.cause(sp, ObligationCauseCode::IfExpressionWithNoElse);
            expected_ty = unit;
            found_ty = then_ty;
            result = self.eq_types(true, &cause, unit, then_ty)
                         .map(|ok| {
                             self.register_infer_ok_obligations(ok);
                             unit
                         });
        }

        match result {
            Ok(ty) => {
                if cond_ty.references_error() {
                    self.tcx.types.err
                } else {
                    ty
                }
            }
            Err(e) => {
                self.report_mismatched_types(&cause, expected_ty, found_ty, e).emit();
                self.tcx.types.err
            }
        }
    }

    // Check field access expressions
    fn check_field(&self,
                   expr: &'gcx hir::Expr,
                   lvalue_pref: LvaluePreference,
                   base: &'gcx hir::Expr,
                   field: &Spanned<ast::Name>) -> Ty<'tcx> {
        let expr_t = self.check_expr_with_lvalue_pref(base, lvalue_pref);
        let expr_t = self.structurally_resolved_type(expr.span,
                                                     expr_t);
        let mut private_candidate = None;
        let mut autoderef = self.autoderef(expr.span, expr_t);
        while let Some((base_t, autoderefs)) = autoderef.next() {
            match base_t.sty {
                ty::TyAdt(base_def, substs) if !base_def.is_enum() => {
                    debug!("struct named {:?}",  base_t);
                    if let Some(field) = base_def.struct_variant().find_field_named(field.node) {
                        let field_ty = self.field_ty(expr.span, field, substs);
                        if self.tcx.vis_is_accessible_from(field.vis, self.body_id) {
                            autoderef.finalize(lvalue_pref, Some(base));
                            self.write_autoderef_adjustment(base.id, autoderefs, base_t);

                            self.tcx.check_stability(field.did, expr.id, expr.span);

                            return field_ty;
                        }
                        private_candidate = Some((base_def.did, field_ty));
                    }
                }
                _ => {}
            }
        }
        autoderef.unambiguous_final_ty();

        if let Some((did, field_ty)) = private_candidate {
            let struct_path = self.tcx().item_path_str(did);
            let msg = format!("field `{}` of struct `{}` is private", field.node, struct_path);
            let mut err = self.tcx().sess.struct_span_err(expr.span, &msg);
            // Also check if an accessible method exists, which is often what is meant.
            if self.method_exists(field.span, field.node, expr_t, expr.id, false) {
                err.note(&format!("a method `{}` also exists, perhaps you wish to call it",
                                  field.node));
            }
            err.emit();
            field_ty
        } else if field.node == keywords::Invalid.name() {
            self.tcx().types.err
        } else if self.method_exists(field.span, field.node, expr_t, expr.id, true) {
            self.type_error_struct(field.span, |actual| {
                format!("attempted to take value of method `{}` on type \
                         `{}`", field.node, actual)
            }, expr_t)
                .help("maybe a `()` to call it is missing? \
                       If not, try an anonymous function")
                .emit();
            self.tcx().types.err
        } else {
            let mut err = self.type_error_struct(field.span, |actual| {
                format!("no field `{}` on type `{}`",
                        field.node, actual)
            }, expr_t);
            match expr_t.sty {
                ty::TyAdt(def, _) if !def.is_enum() => {
                    if let Some(suggested_field_name) =
                        Self::suggest_field_name(def.struct_variant(), field, vec![]) {
                            err.span_label(field.span,
                                           &format!("did you mean `{}`?", suggested_field_name));
                        } else {
                            err.span_label(field.span,
                                           &format!("unknown field"));
                        };
                }
                ty::TyRawPtr(..) => {
                    err.note(&format!("`{0}` is a native pointer; perhaps you need to deref with \
                                      `(*{0}).{1}`",
                                      self.tcx.map.node_to_pretty_string(base.id),
                                      field.node));
                }
                _ => {}
            }
            err.emit();
            self.tcx().types.err
        }
    }

    // Return an hint about the closest match in field names
    fn suggest_field_name(variant: &'tcx ty::VariantDef,
                          field: &Spanned<ast::Name>,
                          skip : Vec<InternedString>)
                          -> Option<Symbol> {
        let name = field.node.as_str();
        let names = variant.fields.iter().filter_map(|field| {
            // ignore already set fields and private fields from non-local crates
            if skip.iter().any(|x| *x == field.name.as_str()) ||
               (variant.did.krate != LOCAL_CRATE && field.vis != Visibility::Public) {
                None
            } else {
                Some(&field.name)
            }
        });

        // only find fits with at least one matching letter
        find_best_match_for_name(names, &name, Some(name.len()))
    }

    // Check tuple index expressions
    fn check_tup_field(&self,
                       expr: &'gcx hir::Expr,
                       lvalue_pref: LvaluePreference,
                       base: &'gcx hir::Expr,
                       idx: codemap::Spanned<usize>) -> Ty<'tcx> {
        let expr_t = self.check_expr_with_lvalue_pref(base, lvalue_pref);
        let expr_t = self.structurally_resolved_type(expr.span,
                                                     expr_t);
        let mut private_candidate = None;
        let mut tuple_like = false;
        let mut autoderef = self.autoderef(expr.span, expr_t);
        while let Some((base_t, autoderefs)) = autoderef.next() {
            let field = match base_t.sty {
                ty::TyAdt(base_def, substs) if base_def.is_struct() => {
                    tuple_like = base_def.struct_variant().ctor_kind == CtorKind::Fn;
                    if !tuple_like { continue }

                    debug!("tuple struct named {:?}",  base_t);
                    base_def.struct_variant().fields.get(idx.node).and_then(|field| {
                        let field_ty = self.field_ty(expr.span, field, substs);
                        private_candidate = Some((base_def.did, field_ty));
                        if self.tcx.vis_is_accessible_from(field.vis, self.body_id) {
                            self.tcx.check_stability(field.did, expr.id, expr.span);
                            Some(field_ty)
                        } else {
                            None
                        }
                    })
                }
                ty::TyTuple(ref v) => {
                    tuple_like = true;
                    v.get(idx.node).cloned()
                }
                _ => continue
            };

            if let Some(field_ty) = field {
                autoderef.finalize(lvalue_pref, Some(base));
                self.write_autoderef_adjustment(base.id, autoderefs, base_t);
                return field_ty;
            }
        }
        autoderef.unambiguous_final_ty();

        if let Some((did, field_ty)) = private_candidate {
            let struct_path = self.tcx().item_path_str(did);
            let msg = format!("field `{}` of struct `{}` is private", idx.node, struct_path);
            self.tcx().sess.span_err(expr.span, &msg);
            return field_ty;
        }

        self.type_error_message(
            expr.span,
            |actual| {
                if tuple_like {
                    format!("attempted out-of-bounds tuple index `{}` on \
                                    type `{}`",
                                   idx.node,
                                   actual)
                } else {
                    format!("attempted tuple index `{}` on type `{}`, but the \
                                     type was not a tuple or tuple struct",
                                    idx.node,
                                    actual)
                }
            },
            expr_t);

        self.tcx().types.err
    }

    fn report_unknown_field(&self,
                            ty: Ty<'tcx>,
                            variant: &'tcx ty::VariantDef,
                            field: &hir::Field,
                            skip_fields: &[hir::Field],
                            kind_name: &str) {
        let mut err = self.type_error_struct_with_diag(
            field.name.span,
            |actual| match ty.sty {
                ty::TyAdt(adt, ..) if adt.is_enum() => {
                    struct_span_err!(self.tcx.sess, field.name.span, E0559,
                                    "{} `{}::{}` has no field named `{}`",
                                    kind_name, actual, variant.name, field.name.node)
                }
                _ => {
                    struct_span_err!(self.tcx.sess, field.name.span, E0560,
                                    "{} `{}` has no field named `{}`",
                                    kind_name, actual, field.name.node)
                }
            },
            ty);
        // prevent all specified fields from being suggested
        let skip_fields = skip_fields.iter().map(|ref x| x.name.node.as_str());
        if let Some(field_name) = Self::suggest_field_name(variant,
                                                           &field.name,
                                                           skip_fields.collect()) {
            err.span_label(field.name.span,
                           &format!("field does not exist - did you mean `{}`?", field_name));
        } else {
            match ty.sty {
                ty::TyAdt(adt, ..) if adt.is_enum() => {
                    err.span_label(field.name.span, &format!("`{}::{}` does not have this field",
                                                             ty, variant.name));
                }
                _ => {
                    err.span_label(field.name.span, &format!("`{}` does not have this field", ty));
                }
            }
        };
        err.emit();
    }

    fn check_expr_struct_fields(&self,
                                adt_ty: Ty<'tcx>,
                                expr_id: ast::NodeId,
                                span: Span,
                                variant: &'tcx ty::VariantDef,
                                ast_fields: &'gcx [hir::Field],
                                check_completeness: bool) {
        let tcx = self.tcx;
        let (substs, adt_kind, kind_name) = match adt_ty.sty {
            ty::TyAdt(adt, substs) => (substs, adt.adt_kind(), adt.variant_descr()),
            _ => span_bug!(span, "non-ADT passed to check_expr_struct_fields")
        };

        let mut remaining_fields = FxHashMap();
        for field in &variant.fields {
            remaining_fields.insert(field.name, field);
        }

        let mut seen_fields = FxHashMap();

        let mut error_happened = false;

        // Typecheck each field.
        for field in ast_fields {
            let expected_field_type;

            if let Some(v_field) = remaining_fields.remove(&field.name.node) {
                expected_field_type = self.field_ty(field.span, v_field, substs);

                seen_fields.insert(field.name.node, field.span);

                // we don't look at stability attributes on
                // struct-like enums (yet...), but it's definitely not
                // a bug to have construct one.
                if adt_kind != ty::AdtKind::Enum {
                    tcx.check_stability(v_field.did, expr_id, field.span);
                }
            } else {
                error_happened = true;
                expected_field_type = tcx.types.err;
                if let Some(_) = variant.find_field_named(field.name.node) {
                    let mut err = struct_span_err!(self.tcx.sess,
                                                field.name.span,
                                                E0062,
                                                "field `{}` specified more than once",
                                                field.name.node);

                    err.span_label(field.name.span, &format!("used more than once"));

                    if let Some(prev_span) = seen_fields.get(&field.name.node) {
                        err.span_label(*prev_span, &format!("first use of `{}`", field.name.node));
                    }

                    err.emit();
                } else {
                    self.report_unknown_field(adt_ty, variant, field, ast_fields, kind_name);
                }
            }

            // Make sure to give a type to the field even if there's
            // an error, so we can continue typechecking
            self.check_expr_coercable_to_type(&field.expr, expected_field_type);
        }

        // Make sure the programmer specified correct number of fields.
        if kind_name == "union" {
            if ast_fields.len() != 1 {
                tcx.sess.span_err(span, "union expressions should have exactly one field");
            }
        } else if check_completeness && !error_happened && !remaining_fields.is_empty() {
            let len = remaining_fields.len();

            let mut displayable_field_names = remaining_fields
                                              .keys()
                                              .map(|x| x.as_str())
                                              .collect::<Vec<_>>();

            displayable_field_names.sort();

            let truncated_fields_error = if len <= 3 {
                "".to_string()
            } else {
                format!(" and {} other field{}", (len - 3), if len - 3 == 1 {""} else {"s"})
            };

            let remaining_fields_names = displayable_field_names.iter().take(3)
                                        .map(|n| format!("`{}`", n))
                                        .collect::<Vec<_>>()
                                        .join(", ");

            struct_span_err!(tcx.sess, span, E0063,
                        "missing field{} {}{} in initializer of `{}`",
                        if remaining_fields.len() == 1 {""} else {"s"},
                        remaining_fields_names,
                        truncated_fields_error,
                        adt_ty)
                        .span_label(span, &format!("missing {}{}",
                            remaining_fields_names,
                            truncated_fields_error))
                        .emit();
        }
    }

    fn check_struct_fields_on_error(&self,
                                    fields: &'gcx [hir::Field],
                                    base_expr: &'gcx Option<P<hir::Expr>>) {
        for field in fields {
            self.check_expr(&field.expr);
        }
        match *base_expr {
            Some(ref base) => {
                self.check_expr(&base);
            },
            None => {}
        }
    }

    pub fn check_struct_path(&self,
                             qpath: &hir::QPath,
                             node_id: ast::NodeId)
                             -> Option<(&'tcx ty::VariantDef,  Ty<'tcx>)> {
        let path_span = match *qpath {
            hir::QPath::Resolved(_, ref path) => path.span,
            hir::QPath::TypeRelative(ref qself, _) => qself.span
        };
        let (def, ty) = self.finish_resolving_struct_path(qpath, path_span, node_id);
        let variant = match def {
            Def::Err => {
                self.set_tainted_by_errors();
                return None;
            }
            Def::Variant(..) => {
                match ty.sty {
                    ty::TyAdt(adt, substs) => {
                        Some((adt.variant_of_def(def), adt.did, substs))
                    }
                    _ => bug!("unexpected type: {:?}", ty.sty)
                }
            }
            Def::Struct(..) | Def::Union(..) | Def::TyAlias(..) |
            Def::AssociatedTy(..) | Def::SelfTy(..) => {
                match def {
                    Def::AssociatedTy(..) | Def::SelfTy(..)
                            if !self.tcx.sess.features.borrow().more_struct_aliases => {
                        emit_feature_err(&self.tcx.sess.parse_sess,
                                         "more_struct_aliases", path_span, GateIssue::Language,
                                         "`Self` and associated types in struct \
                                          expressions and patterns are unstable");
                    }
                    _ => {}
                }
                match ty.sty {
                    ty::TyAdt(adt, substs) if !adt.is_enum() => {
                        Some((adt.struct_variant(), adt.did, substs))
                    }
                    _ => None,
                }
            }
            _ => bug!("unexpected definition: {:?}", def)
        };

        if let Some((variant, did, substs)) = variant {
            // Check bounds on type arguments used in the path.
            let bounds = self.instantiate_bounds(path_span, did, substs);
            let cause = traits::ObligationCause::new(path_span, self.body_id,
                                                     traits::ItemObligation(did));
            self.add_obligations_for_parameters(cause, &bounds);

            Some((variant, ty))
        } else {
            struct_span_err!(self.tcx.sess, path_span, E0071,
                             "expected struct, variant or union type, found {}",
                             ty.sort_string(self.tcx))
                .span_label(path_span, &format!("not a struct"))
                .emit();
            None
        }
    }

    fn check_expr_struct(&self,
                         expr: &hir::Expr,
                         qpath: &hir::QPath,
                         fields: &'gcx [hir::Field],
                         base_expr: &'gcx Option<P<hir::Expr>>) -> Ty<'tcx>
    {
        // Find the relevant variant
        let (variant, struct_ty) =
        if let Some(variant_ty) = self.check_struct_path(qpath, expr.id) {
            variant_ty
        } else {
            self.check_struct_fields_on_error(fields, base_expr);
            return self.tcx.types.err;
        };

        let path_span = match *qpath {
            hir::QPath::Resolved(_, ref path) => path.span,
            hir::QPath::TypeRelative(ref qself, _) => qself.span
        };

        self.check_expr_struct_fields(struct_ty, expr.id, path_span, variant, fields,
                                      base_expr.is_none());
        if let &Some(ref base_expr) = base_expr {
            self.check_expr_has_type(base_expr, struct_ty);
            match struct_ty.sty {
                ty::TyAdt(adt, substs) if adt.is_struct() => {
                    self.tables.borrow_mut().fru_field_types.insert(
                        expr.id,
                        adt.struct_variant().fields.iter().map(|f| {
                            self.normalize_associated_types_in(
                                expr.span, &f.ty(self.tcx, substs)
                            )
                        }).collect()
                    );
                }
                _ => {
                    span_err!(self.tcx.sess, base_expr.span, E0436,
                              "functional record update syntax requires a struct");
                }
            }
        }
        self.require_type_is_sized(struct_ty, expr.span, traits::StructInitializerSized);
        struct_ty
    }


    /// Invariant:
    /// If an expression has any sub-expressions that result in a type error,
    /// inspecting that expression's type with `ty.references_error()` will return
    /// true. Likewise, if an expression is known to diverge, inspecting its
    /// type with `ty::type_is_bot` will return true (n.b.: since Rust is
    /// strict, _|_ can appear in the type of an expression that does not,
    /// itself, diverge: for example, fn() -> _|_.)
    /// Note that inspecting a type's structure *directly* may expose the fact
    /// that there are actually multiple representations for `TyError`, so avoid
    /// that when err needs to be handled differently.
    fn check_expr_with_expectation_and_lvalue_pref(&self,
                                                   expr: &'gcx hir::Expr,
                                                   expected: Expectation<'tcx>,
                                                   lvalue_pref: LvaluePreference) -> Ty<'tcx> {
        debug!(">> typechecking: expr={:?} expected={:?}",
               expr, expected);

        // Warn for expressions after diverging siblings.
        self.warn_if_unreachable(expr.id, expr.span, "expression");

        // Hide the outer diverging and has_errors flags.
        let old_diverges = self.diverges.get();
        let old_has_errors = self.has_errors.get();
        self.diverges.set(Diverges::Maybe);
        self.has_errors.set(false);

        let ty = self.check_expr_kind(expr, expected, lvalue_pref);

        // Warn for non-block expressions with diverging children.
        match expr.node {
            hir::ExprBlock(_) |
            hir::ExprLoop(..) | hir::ExprWhile(..) |
            hir::ExprIf(..) | hir::ExprMatch(..) => {}

            _ => self.warn_if_unreachable(expr.id, expr.span, "expression")
        }

        // Record the type, which applies it effects.
        // We need to do this after the warning above, so that
        // we don't warn for the diverging expression itself.
        self.write_ty(expr.id, ty);

        // Combine the diverging and has_error flags.
        self.diverges.set(self.diverges.get() | old_diverges);
        self.has_errors.set(self.has_errors.get() | old_has_errors);

        debug!("type of {} is...", self.tcx.map.node_to_string(expr.id));
        debug!("... {:?}, expected is {:?}", ty, expected);

        // Add adjustments to !-expressions
        if ty.is_never() {
            if let Some(hir::map::NodeExpr(node_expr)) = self.tcx.map.find(expr.id) {
                let adj_ty = self.next_diverging_ty_var(
                    TypeVariableOrigin::AdjustmentType(node_expr.span));
                self.write_adjustment(expr.id, adjustment::Adjustment {
                    kind: adjustment::Adjust::NeverToAny,
                    target: adj_ty
                });
                return adj_ty;
            }
        }
        ty
    }

    fn check_expr_kind(&self,
                       expr: &'gcx hir::Expr,
                       expected: Expectation<'tcx>,
                       lvalue_pref: LvaluePreference) -> Ty<'tcx> {
        let tcx = self.tcx;
        let id = expr.id;
        match expr.node {
          hir::ExprBox(ref subexpr) => {
            let expected_inner = expected.to_option(self).map_or(NoExpectation, |ty| {
                match ty.sty {
                    ty::TyBox(ty) => Expectation::rvalue_hint(self, ty),
                    _ => NoExpectation
                }
            });
            let referent_ty = self.check_expr_with_expectation(subexpr, expected_inner);
            tcx.mk_box(referent_ty)
          }

          hir::ExprLit(ref lit) => {
            self.check_lit(&lit, expected)
          }
          hir::ExprBinary(op, ref lhs, ref rhs) => {
            self.check_binop(expr, op, lhs, rhs)
          }
          hir::ExprAssignOp(op, ref lhs, ref rhs) => {
            self.check_binop_assign(expr, op, lhs, rhs)
          }
          hir::ExprUnary(unop, ref oprnd) => {
            let expected_inner = match unop {
                hir::UnNot | hir::UnNeg => {
                    expected
                }
                hir::UnDeref => {
                    NoExpectation
                }
            };
            let lvalue_pref = match unop {
                hir::UnDeref => lvalue_pref,
                _ => NoPreference
            };
            let mut oprnd_t = self.check_expr_with_expectation_and_lvalue_pref(&oprnd,
                                                                               expected_inner,
                                                                               lvalue_pref);

            if !oprnd_t.references_error() {
                match unop {
                    hir::UnDeref => {
                        oprnd_t = self.structurally_resolved_type(expr.span, oprnd_t);

                        if let Some(mt) = oprnd_t.builtin_deref(true, NoPreference) {
                            oprnd_t = mt.ty;
                        } else if let Some(method) = self.try_overloaded_deref(
                                expr.span, Some(&oprnd), oprnd_t, lvalue_pref) {
                            oprnd_t = self.make_overloaded_lvalue_return_type(method).ty;
                            self.tables.borrow_mut().method_map.insert(MethodCall::expr(expr.id),
                                                                           method);
                        } else {
                            self.type_error_message(expr.span, |actual| {
                                format!("type `{}` cannot be \
                                        dereferenced", actual)
                            }, oprnd_t);
                            oprnd_t = tcx.types.err;
                        }
                    }
                    hir::UnNot => {
                        oprnd_t = self.structurally_resolved_type(oprnd.span,
                                                                  oprnd_t);
                        let result = self.check_user_unop("!", "not",
                                                          tcx.lang_items.not_trait(),
                                                          expr, &oprnd, oprnd_t, unop);
                        // If it's builtin, we can reuse the type, this helps inference.
                        if !(oprnd_t.is_integral() || oprnd_t.sty == ty::TyBool) {
                            oprnd_t = result;
                        }
                    }
                    hir::UnNeg => {
                        oprnd_t = self.structurally_resolved_type(oprnd.span,
                                                                  oprnd_t);
                        let result = self.check_user_unop("-", "neg",
                                                          tcx.lang_items.neg_trait(),
                                                          expr, &oprnd, oprnd_t, unop);
                        // If it's builtin, we can reuse the type, this helps inference.
                        if !(oprnd_t.is_integral() || oprnd_t.is_fp()) {
                            oprnd_t = result;
                        }
                    }
                }
            }
            oprnd_t
          }
          hir::ExprAddrOf(mutbl, ref oprnd) => {
            let hint = expected.only_has_type(self).map_or(NoExpectation, |ty| {
                match ty.sty {
                    ty::TyRef(_, ref mt) | ty::TyRawPtr(ref mt) => {
                        if self.tcx.expr_is_lval(&oprnd) {
                            // Lvalues may legitimately have unsized types.
                            // For example, dereferences of a fat pointer and
                            // the last field of a struct can be unsized.
                            ExpectHasType(mt.ty)
                        } else {
                            Expectation::rvalue_hint(self, mt.ty)
                        }
                    }
                    _ => NoExpectation
                }
            });
            let lvalue_pref = LvaluePreference::from_mutbl(mutbl);
            let ty = self.check_expr_with_expectation_and_lvalue_pref(&oprnd, hint, lvalue_pref);

            let tm = ty::TypeAndMut { ty: ty, mutbl: mutbl };
            if tm.ty.references_error() {
                tcx.types.err
            } else {
                // Note: at this point, we cannot say what the best lifetime
                // is to use for resulting pointer.  We want to use the
                // shortest lifetime possible so as to avoid spurious borrowck
                // errors.  Moreover, the longest lifetime will depend on the
                // precise details of the value whose address is being taken
                // (and how long it is valid), which we don't know yet until type
                // inference is complete.
                //
                // Therefore, here we simply generate a region variable.  The
                // region inferencer will then select the ultimate value.
                // Finally, borrowck is charged with guaranteeing that the
                // value whose address was taken can actually be made to live
                // as long as it needs to live.
                let region = self.next_region_var(infer::AddrOfRegion(expr.span));
                tcx.mk_ref(region, tm)
            }
          }
          hir::ExprPath(ref qpath) => {
              let (def, opt_ty, segments) = self.resolve_ty_and_def_ufcs(qpath,
                                                                         expr.id, expr.span);
              let ty = if def != Def::Err {
                  self.instantiate_value_path(segments, opt_ty, def, expr.span, id)
              } else {
                  self.set_tainted_by_errors();
                  tcx.types.err
              };

              // We always require that the type provided as the value for
              // a type parameter outlives the moment of instantiation.
              self.opt_node_ty_substs(expr.id, |item_substs| {
                  self.add_wf_bounds(&item_substs.substs, expr);
              });

              ty
          }
          hir::ExprInlineAsm(_, ref outputs, ref inputs) => {
              for output in outputs {
                  self.check_expr(output);
              }
              for input in inputs {
                  self.check_expr(input);
              }
              tcx.mk_nil()
          }
          hir::ExprBreak(label, ref expr_opt) => {
            let loop_id = label.map(|l| l.loop_id);
            let coerce_to = {
                let mut enclosing_loops = self.enclosing_loops.borrow_mut();
                enclosing_loops.find_loop(loop_id).map(|ctxt| ctxt.coerce_to)
            };
            if let Some(coerce_to) = coerce_to {
                let e_ty;
                let cause;
                if let Some(ref e) = *expr_opt {
                    // Recurse without `enclosing_loops` borrowed.
                    e_ty = self.check_expr_with_hint(e, coerce_to);
                    cause = self.misc(e.span);
                    // Notably, the recursive call may alter coerce_to - must not keep using it!
                } else {
                    // `break` without argument acts like `break ()`.
                    e_ty = tcx.mk_nil();
                    cause = self.misc(expr.span);
                }
                let mut enclosing_loops = self.enclosing_loops.borrow_mut();
                let ctxt = enclosing_loops.find_loop(loop_id).unwrap();

                let result = if let Some(ref e) = *expr_opt {
                    // Special-case the first element, as it has no "previous expressions".
                    let result = if !ctxt.may_break {
                        self.try_coerce(e, e_ty, ctxt.coerce_to)
                    } else {
                        self.try_find_coercion_lub(&cause, || ctxt.break_exprs.iter().cloned(),
                                                   ctxt.unified, e, e_ty)
                    };

                    ctxt.break_exprs.push(e);
                    result
                } else {
                    self.eq_types(true, &cause, e_ty, ctxt.unified)
                        .map(|InferOk { obligations, .. }| {
                            // FIXME(#32730) propagate obligations
                            assert!(obligations.is_empty());
                            e_ty
                        })
                };
                match result {
                    Ok(ty) => ctxt.unified = ty,
                    Err(err) => {
                        self.report_mismatched_types(&cause, ctxt.unified, e_ty, err).emit();
                    }
                }

                ctxt.may_break = true;
            }
            // Otherwise, we failed to find the enclosing loop; this can only happen if the
            // `break` was not inside a loop at all, which is caught by the loop-checking pass.
            tcx.types.never
          }
          hir::ExprAgain(_) => { tcx.types.never }
          hir::ExprRet(ref expr_opt) => {
            if self.ret_ty.is_none() {
                struct_span_err!(self.tcx.sess, expr.span, E0572,
                                 "return statement outside of function body").emit();
            } else if let Some(ref e) = *expr_opt {
                self.check_expr_coercable_to_type(&e, self.ret_ty.unwrap());
            } else {
                match self.eq_types(false,
                                    &self.misc(expr.span),
                                    self.ret_ty.unwrap(),
                                    tcx.mk_nil()) {
                    Ok(ok) => self.register_infer_ok_obligations(ok),
                    Err(_) => {
                        struct_span_err!(tcx.sess, expr.span, E0069,
                                         "`return;` in a function whose return type is not `()`")
                            .span_label(expr.span, &format!("return type is not ()"))
                            .emit();
                    }
                }
            }
            tcx.types.never
          }
          hir::ExprAssign(ref lhs, ref rhs) => {
            let lhs_ty = self.check_expr_with_lvalue_pref(&lhs, PreferMutLvalue);

            let tcx = self.tcx;
            if !tcx.expr_is_lval(&lhs) {
                struct_span_err!(
                    tcx.sess, expr.span, E0070,
                    "invalid left-hand side expression")
                .span_label(
                    expr.span,
                    &format!("left-hand of expression not valid"))
                .emit();
            }

            let rhs_ty = self.check_expr_coercable_to_type(&rhs, lhs_ty);

            self.require_type_is_sized(lhs_ty, lhs.span, traits::AssignmentLhsSized);

            if lhs_ty.references_error() || rhs_ty.references_error() {
                tcx.types.err
            } else {
                tcx.mk_nil()
            }
          }
          hir::ExprIf(ref cond, ref then_blk, ref opt_else_expr) => {
            self.check_then_else(&cond, &then_blk, opt_else_expr.as_ref().map(|e| &**e),
                                 expr.span, expected)
          }
          hir::ExprWhile(ref cond, ref body, _) => {
            let unified = self.tcx.mk_nil();
            let coerce_to = unified;
            let ctxt = LoopCtxt {
                unified: unified,
                coerce_to: coerce_to,
                break_exprs: vec![],
                may_break: true,
            };
            self.with_loop_ctxt(expr.id, ctxt, || {
                self.check_expr_has_type(&cond, tcx.types.bool);
                let cond_diverging = self.diverges.get();
                self.check_block_no_value(&body);

                // We may never reach the body so it diverging means nothing.
                self.diverges.set(cond_diverging);
            });

            if self.has_errors.get() {
                tcx.types.err
            } else {
                tcx.mk_nil()
            }
          }
          hir::ExprLoop(ref body, _, _) => {
            let unified = self.next_ty_var(TypeVariableOrigin::TypeInference(body.span));
            let coerce_to = expected.only_has_type(self).unwrap_or(unified);
            let ctxt = LoopCtxt {
                unified: unified,
                coerce_to: coerce_to,
                break_exprs: vec![],
                may_break: false,
            };

            let ctxt = self.with_loop_ctxt(expr.id, ctxt, || {
                self.check_block_no_value(&body);
            });
            if ctxt.may_break {
                // No way to know whether it's diverging because
                // of a `break` or an outer `break` or `return.
                self.diverges.set(Diverges::Maybe);

                ctxt.unified
            } else {
                tcx.types.never
            }
          }
          hir::ExprMatch(ref discrim, ref arms, match_src) => {
            self.check_match(expr, &discrim, arms, expected, match_src)
          }
          hir::ExprClosure(capture, ref decl, body_id, _) => {
              self.check_expr_closure(expr, capture, &decl, body_id, expected)
          }
          hir::ExprBlock(ref b) => {
            self.check_block_with_expected(&b, expected)
          }
          hir::ExprCall(ref callee, ref args) => {
              self.check_call(expr, &callee, args, expected)
          }
          hir::ExprMethodCall(name, ref tps, ref args) => {
              self.check_method_call(expr, name, args, &tps[..], expected, lvalue_pref)
          }
          hir::ExprCast(ref e, ref t) => {
            // Find the type of `e`. Supply hints based on the type we are casting to,
            // if appropriate.
            let t_cast = self.to_ty(t);
            let t_cast = self.resolve_type_vars_if_possible(&t_cast);
            let t_expr = self.check_expr_with_expectation(e, ExpectCastableToType(t_cast));
            let t_cast = self.resolve_type_vars_if_possible(&t_cast);

            // Eagerly check for some obvious errors.
            if t_expr.references_error() || t_cast.references_error() {
                tcx.types.err
            } else {
                // Defer other checks until we're done type checking.
                let mut deferred_cast_checks = self.deferred_cast_checks.borrow_mut();
                match cast::CastCheck::new(self, e, t_expr, t_cast, t.span, expr.span) {
                    Ok(cast_check) => {
                        deferred_cast_checks.push(cast_check);
                        t_cast
                    }
                    Err(ErrorReported) => {
                        tcx.types.err
                    }
                }
            }
          }
          hir::ExprType(ref e, ref t) => {
            let typ = self.to_ty(&t);
            self.check_expr_eq_type(&e, typ);
            typ
          }
          hir::ExprArray(ref args) => {
            let uty = expected.to_option(self).and_then(|uty| {
                match uty.sty {
                    ty::TyArray(ty, _) | ty::TySlice(ty) => Some(ty),
                    _ => None
                }
            });

            let mut unified = self.next_ty_var(TypeVariableOrigin::TypeInference(expr.span));
            let coerce_to = uty.unwrap_or(unified);

            for (i, e) in args.iter().enumerate() {
                let e_ty = self.check_expr_with_hint(e, coerce_to);
                let cause = self.misc(e.span);

                // Special-case the first element, as it has no "previous expressions".
                let result = if i == 0 {
                    self.try_coerce(e, e_ty, coerce_to)
                } else {
                    let prev_elems = || args[..i].iter().map(|e| &*e);
                    self.try_find_coercion_lub(&cause, prev_elems, unified, e, e_ty)
                };

                match result {
                    Ok(ty) => unified = ty,
                    Err(e) => {
                        self.report_mismatched_types(&cause, unified, e_ty, e).emit();
                    }
                }
            }
            tcx.mk_array(unified, args.len())
          }
          hir::ExprRepeat(ref element, count) => {
            let count = eval_length(self.tcx.global_tcx(), count, "repeat count")
                  .unwrap_or(0);

            let uty = match expected {
                ExpectHasType(uty) => {
                    match uty.sty {
                        ty::TyArray(ty, _) | ty::TySlice(ty) => Some(ty),
                        _ => None
                    }
                }
                _ => None
            };

            let (element_ty, t) = match uty {
                Some(uty) => {
                    self.check_expr_coercable_to_type(&element, uty);
                    (uty, uty)
                }
                None => {
                    let t: Ty = self.next_ty_var(TypeVariableOrigin::MiscVariable(element.span));
                    let element_ty = self.check_expr_has_type(&element, t);
                    (element_ty, t)
                }
            };

            if count > 1 {
                // For [foo, ..n] where n > 1, `foo` must have
                // Copy type:
                let lang_item = self.tcx.require_lang_item(lang_items::CopyTraitLangItem);
                self.require_type_meets(t, expr.span, traits::RepeatVec, lang_item);
            }

            if element_ty.references_error() {
                tcx.types.err
            } else {
                tcx.mk_array(t, count)
            }
          }
          hir::ExprTup(ref elts) => {
            let flds = expected.only_has_type(self).and_then(|ty| {
                match ty.sty {
                    ty::TyTuple(ref flds) => Some(&flds[..]),
                    _ => None
                }
            });

            let elt_ts_iter = elts.iter().enumerate().map(|(i, e)| {
                let t = match flds {
                    Some(ref fs) if i < fs.len() => {
                        let ety = fs[i];
                        self.check_expr_coercable_to_type(&e, ety);
                        ety
                    }
                    _ => {
                        self.check_expr_with_expectation(&e, NoExpectation)
                    }
                };
                t
            });
            let tuple = tcx.mk_tup(elt_ts_iter);
            if tuple.references_error() {
                tcx.types.err
            } else {
                tuple
            }
          }
          hir::ExprStruct(ref qpath, ref fields, ref base_expr) => {
            self.check_expr_struct(expr, qpath, fields, base_expr)
          }
          hir::ExprField(ref base, ref field) => {
            self.check_field(expr, lvalue_pref, &base, field)
          }
          hir::ExprTupField(ref base, idx) => {
            self.check_tup_field(expr, lvalue_pref, &base, idx)
          }
          hir::ExprIndex(ref base, ref idx) => {
              let base_t = self.check_expr_with_lvalue_pref(&base, lvalue_pref);
              let idx_t = self.check_expr(&idx);

              if base_t.references_error() {
                  base_t
              } else if idx_t.references_error() {
                  idx_t
              } else {
                  let base_t = self.structurally_resolved_type(expr.span, base_t);
                  match self.lookup_indexing(expr, base, base_t, idx_t, lvalue_pref) {
                      Some((index_ty, element_ty)) => {
                          self.demand_eqtype(expr.span, index_ty, idx_t);
                          element_ty
                      }
                      None => {
                          self.check_expr_has_type(&idx, self.tcx.types.err);
                          let mut err = self.type_error_struct(
                              expr.span,
                              |actual| {
                                  format!("cannot index a value of type `{}`",
                                          actual)
                              },
                              base_t);
                          // Try to give some advice about indexing tuples.
                          if let ty::TyTuple(_) = base_t.sty {
                              let mut needs_note = true;
                              // If the index is an integer, we can show the actual
                              // fixed expression:
                              if let hir::ExprLit(ref lit) = idx.node {
                                  if let ast::LitKind::Int(i,
                                            ast::LitIntType::Unsuffixed) = lit.node {
                                      let snip = tcx.sess.codemap().span_to_snippet(base.span);
                                      if let Ok(snip) = snip {
                                          err.span_suggestion(expr.span,
                                                              "to access tuple elements, \
                                                               use tuple indexing syntax \
                                                               as shown",
                                                              format!("{}.{}", snip, i));
                                          needs_note = false;
                                      }
                                  }
                              }
                              if needs_note {
                                  err.help("to access tuple elements, use tuple indexing \
                                            syntax (e.g. `tuple.0`)");
                              }
                          }
                          err.emit();
                          self.tcx.types.err
                      }
                  }
              }
           }
        }
    }

    // Finish resolving a path in a struct expression or pattern `S::A { .. }` if necessary.
    // The newly resolved definition is written into `type_relative_path_defs`.
    fn finish_resolving_struct_path(&self,
                                    qpath: &hir::QPath,
                                    path_span: Span,
                                    node_id: ast::NodeId)
                                    -> (Def, Ty<'tcx>)
    {
        match *qpath {
            hir::QPath::Resolved(ref maybe_qself, ref path) => {
                let opt_self_ty = maybe_qself.as_ref().map(|qself| self.to_ty(qself));
                let ty = AstConv::def_to_ty(self, self, opt_self_ty, path, node_id, true);
                (path.def, ty)
            }
            hir::QPath::TypeRelative(ref qself, ref segment) => {
                let ty = self.to_ty(qself);

                let def = if let hir::TyPath(hir::QPath::Resolved(_, ref path)) = qself.node {
                    path.def
                } else {
                    Def::Err
                };
                let (ty, def) = AstConv::associated_path_def_to_ty(self, node_id, path_span,
                                                                   ty, def, segment);

                // Write back the new resolution.
                self.tables.borrow_mut().type_relative_path_defs.insert(node_id, def);

                (def, ty)
            }
        }
    }

    // Resolve associated value path into a base type and associated constant or method definition.
    // The newly resolved definition is written into `type_relative_path_defs`.
    pub fn resolve_ty_and_def_ufcs<'b>(&self,
                                       qpath: &'b hir::QPath,
                                       node_id: ast::NodeId,
                                       span: Span)
                                       -> (Def, Option<Ty<'tcx>>, &'b [hir::PathSegment])
    {
        let (ty, item_segment) = match *qpath {
            hir::QPath::Resolved(ref opt_qself, ref path) => {
                return (path.def,
                        opt_qself.as_ref().map(|qself| self.to_ty(qself)),
                        &path.segments[..]);
            }
            hir::QPath::TypeRelative(ref qself, ref segment) => {
                (self.to_ty(qself), segment)
            }
        };
        let item_name = item_segment.name;
        let def = match self.resolve_ufcs(span, item_name, ty, node_id) {
            Ok(def) => def,
            Err(error) => {
                let def = match error {
                    method::MethodError::PrivateMatch(def) => def,
                    _ => Def::Err,
                };
                if item_name != keywords::Invalid.name() {
                    self.report_method_error(span, ty, item_name, None, error, None);
                }
                def
            }
        };

        // Write back the new resolution.
        self.tables.borrow_mut().type_relative_path_defs.insert(node_id, def);
        (def, Some(ty), slice::ref_slice(&**item_segment))
    }

    pub fn check_decl_initializer(&self,
                                  local: &'gcx hir::Local,
                                  init: &'gcx hir::Expr) -> Ty<'tcx>
    {
        let ref_bindings = local.pat.contains_ref_binding();

        let local_ty = self.local_ty(init.span, local.id);
        if let Some(m) = ref_bindings {
            // Somewhat subtle: if we have a `ref` binding in the pattern,
            // we want to avoid introducing coercions for the RHS. This is
            // both because it helps preserve sanity and, in the case of
            // ref mut, for soundness (issue #23116). In particular, in
            // the latter case, we need to be clear that the type of the
            // referent for the reference that results is *equal to* the
            // type of the lvalue it is referencing, and not some
            // supertype thereof.
            let init_ty = self.check_expr_with_lvalue_pref(init, LvaluePreference::from_mutbl(m));
            self.demand_eqtype(init.span, init_ty, local_ty);
            init_ty
        } else {
            self.check_expr_coercable_to_type(init, local_ty)
        }
    }

    pub fn check_decl_local(&self, local: &'gcx hir::Local)  {
        let t = self.local_ty(local.span, local.id);
        self.write_ty(local.id, t);

        if let Some(ref init) = local.init {
            let init_ty = self.check_decl_initializer(local, &init);
            if init_ty.references_error() {
                self.write_ty(local.id, init_ty);
            }
        }

        self.check_pat(&local.pat, t);
        let pat_ty = self.node_ty(local.pat.id);
        if pat_ty.references_error() {
            self.write_ty(local.id, pat_ty);
        }
    }

    pub fn check_stmt(&self, stmt: &'gcx hir::Stmt) {
        // Don't do all the complex logic below for DeclItem.
        match stmt.node {
            hir::StmtDecl(ref decl, id) => {
                match decl.node {
                    hir::DeclLocal(_) => {}
                    hir::DeclItem(_) => {
                        self.write_nil(id);
                        return;
                    }
                }
            }
            hir::StmtExpr(..) | hir::StmtSemi(..) => {}
        }

        self.warn_if_unreachable(stmt.node.id(), stmt.span, "statement");

        // Hide the outer diverging and has_errors flags.
        let old_diverges = self.diverges.get();
        let old_has_errors = self.has_errors.get();
        self.diverges.set(Diverges::Maybe);
        self.has_errors.set(false);

        let (node_id, span) = match stmt.node {
            hir::StmtDecl(ref decl, id) => {
                let span = match decl.node {
                    hir::DeclLocal(ref l) => {
                        self.check_decl_local(&l);
                        l.span
                    }
                    hir::DeclItem(_) => {/* ignore for now */
                        DUMMY_SP
                    }
                };
                (id, span)
            }
            hir::StmtExpr(ref expr, id) => {
                // Check with expected type of ()
                self.check_expr_has_type(&expr, self.tcx.mk_nil());
                (id, expr.span)
            }
            hir::StmtSemi(ref expr, id) => {
                self.check_expr(&expr);
                (id, expr.span)
            }
        };

        if self.has_errors.get() {
            self.write_error(node_id);
        } else if self.diverges.get().always() {
            self.write_ty(node_id, self.next_diverging_ty_var(
                TypeVariableOrigin::DivergingStmt(span)));
        } else {
            self.write_nil(node_id);
        }

        // Combine the diverging and has_error flags.
        self.diverges.set(self.diverges.get() | old_diverges);
        self.has_errors.set(self.has_errors.get() | old_has_errors);
    }

    pub fn check_block_no_value(&self, blk: &'gcx hir::Block)  {
        let unit = self.tcx.mk_nil();
        let ty = self.check_block_with_expected(blk, ExpectHasType(unit));
        self.demand_suptype(blk.span, unit, ty);
    }

    fn check_block_with_expected(&self,
                                 blk: &'gcx hir::Block,
                                 expected: Expectation<'tcx>) -> Ty<'tcx> {
        let prev = {
            let mut fcx_ps = self.ps.borrow_mut();
            let unsafety_state = fcx_ps.recurse(blk);
            replace(&mut *fcx_ps, unsafety_state)
        };

        for s in &blk.stmts {
            self.check_stmt(s);
        }

        let mut ty = match blk.expr {
            Some(ref e) => self.check_expr_with_expectation(e, expected),
            None => self.tcx.mk_nil()
        };

        if self.diverges.get().always() {
            if let ExpectHasType(ety) = expected {
                // Avoid forcing a type (only `!` for now) in unreachable code.
                // FIXME(aburka) do we need this special case? and should it be is_uninhabited?
                if !ety.is_never() {
                    if let Some(ref e) = blk.expr {
                        // Coerce the tail expression to the right type.
                        self.demand_coerce(e, ty, ety);
                    }
                }
            }

            ty = self.next_diverging_ty_var(TypeVariableOrigin::DivergingBlockExpr(blk.span));
        } else if let ExpectHasType(ety) = expected {
            if let Some(ref e) = blk.expr {
                // Coerce the tail expression to the right type.
                self.demand_coerce(e, ty, ety);
            } else {
                // We're not diverging and there's an expected type, which,
                // in case it's not `()`, could result in an error higher-up.
                // We have a chance to error here early and be more helpful.
                let cause = self.misc(blk.span);
                let trace = TypeTrace::types(&cause, false, ty, ety);
                match self.sub_types(false, &cause, ty, ety) {
                    Ok(InferOk { obligations, .. }) => {
                        // FIXME(#32730) propagate obligations
                        assert!(obligations.is_empty());
                    },
                    Err(err) => {
                        let mut err = self.report_and_explain_type_error(trace, &err);

                        // Be helpful when the user wrote `{... expr;}` and
                        // taking the `;` off is enough to fix the error.
                        let mut extra_semi = None;
                        if let Some(stmt) = blk.stmts.last() {
                            if let hir::StmtSemi(ref e, _) = stmt.node {
                                if self.can_sub_types(self.node_ty(e.id), ety).is_ok() {
                                    extra_semi = Some(stmt);
                                }
                            }
                        }
                        if let Some(last_stmt) = extra_semi {
                            let original_span = original_sp(self.tcx.sess.codemap(),
                                                            last_stmt.span, blk.span);
                            let span_semi = Span {
                                lo: original_span.hi - BytePos(1),
                                hi: original_span.hi,
                                expn_id: original_span.expn_id
                            };
                            err.span_help(span_semi, "consider removing this semicolon:");
                        }

                        err.emit();
                    }
                }
            }

            // We already applied the type (and potentially errored),
            // use the expected type to avoid further errors out.
            ty = ety;
        }

        if self.has_errors.get() || ty.references_error() {
            ty = self.tcx.types.err
        }

        self.write_ty(blk.id, ty);

        *self.ps.borrow_mut() = prev;
        ty
    }

    // Instantiates the given path, which must refer to an item with the given
    // number of type parameters and type.
    pub fn instantiate_value_path(&self,
                                  segments: &[hir::PathSegment],
                                  opt_self_ty: Option<Ty<'tcx>>,
                                  def: Def,
                                  span: Span,
                                  node_id: ast::NodeId)
                                  -> Ty<'tcx> {
        debug!("instantiate_value_path(path={:?}, def={:?}, node_id={})",
               segments,
               def,
               node_id);

        // We need to extract the type parameters supplied by the user in
        // the path `path`. Due to the current setup, this is a bit of a
        // tricky-process; the problem is that resolve only tells us the
        // end-point of the path resolution, and not the intermediate steps.
        // Luckily, we can (at least for now) deduce the intermediate steps
        // just from the end-point.
        //
        // There are basically four cases to consider:
        //
        // 1. Reference to a constructor of enum variant or struct:
        //
        //        struct Foo<T>(...)
        //        enum E<T> { Foo(...) }
        //
        //    In these cases, the parameters are declared in the type
        //    space.
        //
        // 2. Reference to a fn item or a free constant:
        //
        //        fn foo<T>() { }
        //
        //    In this case, the path will again always have the form
        //    `a::b::foo::<T>` where only the final segment should have
        //    type parameters. However, in this case, those parameters are
        //    declared on a value, and hence are in the `FnSpace`.
        //
        // 3. Reference to a method or an associated constant:
        //
        //        impl<A> SomeStruct<A> {
        //            fn foo<B>(...)
        //        }
        //
        //    Here we can have a path like
        //    `a::b::SomeStruct::<A>::foo::<B>`, in which case parameters
        //    may appear in two places. The penultimate segment,
        //    `SomeStruct::<A>`, contains parameters in TypeSpace, and the
        //    final segment, `foo::<B>` contains parameters in fn space.
        //
        // 4. Reference to a local variable
        //
        //    Local variables can't have any type parameters.
        //
        // The first step then is to categorize the segments appropriately.

        assert!(!segments.is_empty());

        let mut ufcs_associated = None;
        let mut type_segment = None;
        let mut fn_segment = None;
        match def {
            // Case 1. Reference to a struct/variant constructor.
            Def::StructCtor(def_id, ..) |
            Def::VariantCtor(def_id, ..) => {
                // Everything but the final segment should have no
                // parameters at all.
                let mut generics = self.tcx.item_generics(def_id);
                if let Some(def_id) = generics.parent {
                    // Variant and struct constructors use the
                    // generics of their parent type definition.
                    generics = self.tcx.item_generics(def_id);
                }
                type_segment = Some((segments.last().unwrap(), generics));
            }

            // Case 2. Reference to a top-level value.
            Def::Fn(def_id) |
            Def::Const(def_id) |
            Def::Static(def_id, _) => {
                fn_segment = Some((segments.last().unwrap(),
                                   self.tcx.item_generics(def_id)));
            }

            // Case 3. Reference to a method or associated const.
            Def::Method(def_id) |
            Def::AssociatedConst(def_id) => {
                let container = self.tcx.associated_item(def_id).container;
                match container {
                    ty::TraitContainer(trait_did) => {
                        callee::check_legal_trait_for_method_call(self.ccx, span, trait_did)
                    }
                    ty::ImplContainer(_) => {}
                }

                let generics = self.tcx.item_generics(def_id);
                if segments.len() >= 2 {
                    let parent_generics = self.tcx.item_generics(generics.parent.unwrap());
                    type_segment = Some((&segments[segments.len() - 2], parent_generics));
                } else {
                    // `<T>::assoc` will end up here, and so can `T::assoc`.
                    let self_ty = opt_self_ty.expect("UFCS sugared assoc missing Self");
                    ufcs_associated = Some((container, self_ty));
                }
                fn_segment = Some((segments.last().unwrap(), generics));
            }

            // Case 4. Local variable, no generics.
            Def::Local(..) | Def::Upvar(..) => {}

            _ => bug!("unexpected definition: {:?}", def),
        }

        debug!("type_segment={:?} fn_segment={:?}", type_segment, fn_segment);

        // Now that we have categorized what space the parameters for each
        // segment belong to, let's sort out the parameters that the user
        // provided (if any) into their appropriate spaces. We'll also report
        // errors if type parameters are provided in an inappropriate place.
        let poly_segments = type_segment.is_some() as usize +
                            fn_segment.is_some() as usize;
        self.tcx.prohibit_type_params(&segments[..segments.len() - poly_segments]);

        match def {
            Def::Local(def_id) | Def::Upvar(def_id, ..) => {
                let nid = self.tcx.map.as_local_node_id(def_id).unwrap();
                let ty = self.local_ty(span, nid);
                let ty = self.normalize_associated_types_in(span, &ty);
                self.write_ty(node_id, ty);
                self.write_substs(node_id, ty::ItemSubsts {
                    substs: self.tcx.intern_substs(&[])
                });
                return ty;
            }
            _ => {}
        }

        // Now we have to compare the types that the user *actually*
        // provided against the types that were *expected*. If the user
        // did not provide any types, then we want to substitute inference
        // variables. If the user provided some types, we may still need
        // to add defaults. If the user provided *too many* types, that's
        // a problem.
        self.check_path_parameter_count(span, &mut type_segment);
        self.check_path_parameter_count(span, &mut fn_segment);

        let (fn_start, has_self) = match (type_segment, fn_segment) {
            (_, Some((_, generics))) => {
                (generics.parent_count(), generics.has_self)
            }
            (Some((_, generics)), None) => {
                (generics.own_count(), generics.has_self)
            }
            (None, None) => (0, false)
        };
        let substs = Substs::for_item(self.tcx, def.def_id(), |def, _| {
            let mut i = def.index as usize;

            let segment = if i < fn_start {
                i -= has_self as usize;
                type_segment
            } else {
                i -= fn_start;
                fn_segment
            };
            let lifetimes = match segment.map(|(s, _)| &s.parameters) {
                Some(&hir::AngleBracketedParameters(ref data)) => &data.lifetimes[..],
                Some(&hir::ParenthesizedParameters(_)) => bug!(),
                None => &[]
            };

            if let Some(ast_lifetime) = lifetimes.get(i) {
                ast_region_to_region(self.tcx, ast_lifetime)
            } else {
                self.region_var_for_def(span, def)
            }
        }, |def, substs| {
            let mut i = def.index as usize;

            let segment = if i < fn_start {
                // Handle Self first, so we can adjust the index to match the AST.
                if has_self && i == 0 {
                    return opt_self_ty.unwrap_or_else(|| {
                        self.type_var_for_def(span, def, substs)
                    });
                }
                i -= has_self as usize;
                type_segment
            } else {
                i -= fn_start;
                fn_segment
            };
            let (types, infer_types) = match segment.map(|(s, _)| &s.parameters) {
                Some(&hir::AngleBracketedParameters(ref data)) => {
                    (&data.types[..], data.infer_types)
                }
                Some(&hir::ParenthesizedParameters(_)) => bug!(),
                None => (&[][..], true)
            };

            // Skip over the lifetimes in the same segment.
            if let Some((_, generics)) = segment {
                i -= generics.regions.len();
            }

            if let Some(ast_ty) = types.get(i) {
                // A provided type parameter.
                self.to_ty(ast_ty)
            } else if let (false, Some(default)) = (infer_types, def.default) {
                // No type parameter provided, but a default exists.
                default.subst_spanned(self.tcx, substs, Some(span))
            } else {
                // No type parameters were provided, we can infer all.
                // This can also be reached in some error cases:
                // We prefer to use inference variables instead of
                // TyError to let type inference recover somewhat.
                self.type_var_for_def(span, def, substs)
            }
        });

        // The things we are substituting into the type should not contain
        // escaping late-bound regions, and nor should the base type scheme.
        let ty = self.tcx.item_type(def.def_id());
        assert!(!substs.has_escaping_regions());
        assert!(!ty.has_escaping_regions());

        // Add all the obligations that are required, substituting and
        // normalized appropriately.
        let bounds = self.instantiate_bounds(span, def.def_id(), &substs);
        self.add_obligations_for_parameters(
            traits::ObligationCause::new(span, self.body_id, traits::ItemObligation(def.def_id())),
            &bounds);

        // Substitute the values for the type parameters into the type of
        // the referenced item.
        let ty_substituted = self.instantiate_type_scheme(span, &substs, &ty);

        if let Some((ty::ImplContainer(impl_def_id), self_ty)) = ufcs_associated {
            // In the case of `Foo<T>::method` and `<Foo<T>>::method`, if `method`
            // is inherent, there is no `Self` parameter, instead, the impl needs
            // type parameters, which we can infer by unifying the provided `Self`
            // with the substituted impl type.
            let ty = self.tcx.item_type(impl_def_id);

            let impl_ty = self.instantiate_type_scheme(span, &substs, &ty);
            match self.sub_types(false, &self.misc(span), self_ty, impl_ty) {
                Ok(ok) => self.register_infer_ok_obligations(ok),
                Err(_) => {
                    span_bug!(span,
                        "instantiate_value_path: (UFCS) {:?} was a subtype of {:?} but now is not?",
                        self_ty,
                        impl_ty);
                }
            }
        }

        debug!("instantiate_value_path: type of {:?} is {:?}",
               node_id,
               ty_substituted);
        self.write_substs(node_id, ty::ItemSubsts {
            substs: substs
        });
        ty_substituted
    }

    /// Report errors if the provided parameters are too few or too many.
    fn check_path_parameter_count(&self,
                                  span: Span,
                                  segment: &mut Option<(&hir::PathSegment, &ty::Generics)>) {
        let (lifetimes, types, infer_types, bindings) = {
            match segment.map(|(s, _)| &s.parameters) {
                Some(&hir::AngleBracketedParameters(ref data)) => {
                    (&data.lifetimes[..], &data.types[..], data.infer_types, &data.bindings[..])
                }
                Some(&hir::ParenthesizedParameters(_)) => {
                    span_bug!(span, "parenthesized parameters cannot appear in ExprPath");
                }
                None => (&[][..], &[][..], true, &[][..])
            }
        };

        let count = |n| {
            format!("{} parameter{}", n, if n == 1 { "" } else { "s" })
        };

        // Check provided lifetime parameters.
        let lifetime_defs = segment.map_or(&[][..], |(_, generics)| &generics.regions);
        if lifetimes.len() > lifetime_defs.len() {
            struct_span_err!(self.tcx.sess, span, E0088,
                             "too many lifetime parameters provided: \
                              expected {}, found {}",
                              count(lifetime_defs.len()),
                              count(lifetimes.len()))
                .span_label(span, &format!("unexpected lifetime parameter{}",
                                           match lifetimes.len() { 1 => "", _ => "s" }))
                .emit();
        } else if lifetimes.len() > 0 && lifetimes.len() < lifetime_defs.len() {
            struct_span_err!(self.tcx.sess, span, E0090,
                             "too few lifetime parameters provided: \
                             expected {}, found {}",
                             count(lifetime_defs.len()),
                             count(lifetimes.len()))
                .span_label(span, &format!("too few lifetime parameters"))
                .emit();
        }

        // The case where there is not enough lifetime parameters is not checked,
        // because this is not possible - a function never takes lifetime parameters.
        // See discussion for Pull Request 36208.

        // Check provided type parameters.
        let type_defs = segment.map_or(&[][..], |(_, generics)| {
            if generics.parent.is_none() {
                &generics.types[generics.has_self as usize..]
            } else {
                &generics.types
            }
        });
        let required_len = type_defs.iter()
                                    .take_while(|d| d.default.is_none())
                                    .count();
        if types.len() > type_defs.len() {
            let span = types[type_defs.len()].span;
            struct_span_err!(self.tcx.sess, span, E0087,
                             "too many type parameters provided: \
                              expected at most {}, found {}",
                             count(type_defs.len()),
                             count(types.len()))
                .span_label(span, &format!("too many type parameters")).emit();

            // To prevent derived errors to accumulate due to extra
            // type parameters, we force instantiate_value_path to
            // use inference variables instead of the provided types.
            *segment = None;
        } else if !infer_types && types.len() < required_len {
            let adjust = |len| if len > 1 { "parameters" } else { "parameter" };
            let required_param_str = adjust(required_len);
            let actual_param_str = adjust(types.len());
            struct_span_err!(self.tcx.sess, span, E0089,
                             "too few type parameters provided: \
                              expected {} {}, found {} {}",
                             count(required_len),
                             required_param_str,
                             count(types.len()),
                             actual_param_str)
                .span_label(span, &format!("expected {} type {}", required_len, required_param_str))
                .emit();
        }

        if !bindings.is_empty() {
            span_err!(self.tcx.sess, bindings[0].span, E0182,
                      "unexpected binding of associated item in expression path \
                       (only allowed in type paths)");
        }
    }

    fn structurally_resolve_type_or_else<F>(&self, sp: Span, ty: Ty<'tcx>, f: F)
                                            -> Ty<'tcx>
        where F: Fn() -> Ty<'tcx>
    {
        let mut ty = self.resolve_type_vars_with_obligations(ty);

        if ty.is_ty_var() {
            let alternative = f();

            // If not, error.
            if alternative.is_ty_var() || alternative.references_error() {
                if !self.is_tainted_by_errors() {
                    self.type_error_message(sp, |_actual| {
                        "the type of this value must be known in this context".to_string()
                    }, ty);
                }
                self.demand_suptype(sp, self.tcx.types.err, ty);
                ty = self.tcx.types.err;
            } else {
                self.demand_suptype(sp, alternative, ty);
                ty = alternative;
            }
        }

        ty
    }

    // Resolves `typ` by a single level if `typ` is a type variable.  If no
    // resolution is possible, then an error is reported.
    pub fn structurally_resolved_type(&self, sp: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.structurally_resolve_type_or_else(sp, ty, || {
            self.tcx.types.err
        })
    }

    fn with_loop_ctxt<F: FnOnce()>(&self, id: ast::NodeId, ctxt: LoopCtxt<'gcx, 'tcx>, f: F)
                                   -> LoopCtxt<'gcx, 'tcx> {
        let index;
        {
            let mut enclosing_loops = self.enclosing_loops.borrow_mut();
            index = enclosing_loops.stack.len();
            enclosing_loops.by_id.insert(id, index);
            enclosing_loops.stack.push(ctxt);
        }
        f();
        {
            let mut enclosing_loops = self.enclosing_loops.borrow_mut();
            debug_assert!(enclosing_loops.stack.len() == index + 1);
            enclosing_loops.by_id.remove(&id).expect("missing loop context");
            (enclosing_loops.stack.pop().expect("missing loop context"))
        }
    }
}

pub fn check_bounds_are_used<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                       generics: &hir::Generics,
                                       ty: Ty<'tcx>) {
    debug!("check_bounds_are_used(n_tps={}, ty={:?})",
           generics.ty_params.len(),  ty);

    // make a vector of booleans initially false, set to true when used
    if generics.ty_params.is_empty() { return; }
    let mut tps_used = vec![false; generics.ty_params.len()];

    for leaf_ty in ty.walk() {
        if let ty::TyParam(ParamTy {idx, ..}) = leaf_ty.sty {
            debug!("Found use of ty param num {}", idx);
            tps_used[idx as usize - generics.lifetimes.len()] = true;
        }
    }

    for (&used, param) in tps_used.iter().zip(&generics.ty_params) {
        if !used {
            struct_span_err!(ccx.tcx.sess, param.span, E0091,
                "type parameter `{}` is unused",
                param.name)
                .span_label(param.span, &format!("unused type parameter"))
                .emit();
        }
    }
}
