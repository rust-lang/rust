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

use astconv::{AstConv, ast_region_to_region, PathParamMode};
use dep_graph::DepNode;
use fmt_macros::{Parser, Piece, Position};
use middle::cstore::LOCAL_CRATE;
use hir::def::{self, Def};
use hir::def_id::DefId;
use hir::pat_util;
use rustc::infer::{self, InferCtxt, InferOk, TypeOrigin, TypeTrace, type_variable};
use rustc::ty::subst::{self, Subst, Substs, VecPerParamSpace, ParamSpace};
use rustc::traits::{self, ProjectionMode};
use rustc::ty::{GenericPredicates, TypeScheme};
use rustc::ty::{ParamTy, ParameterEnvironment};
use rustc::ty::{LvaluePreference, NoPreference, PreferMutLvalue};
use rustc::ty::{self, ToPolyTraitRef, Ty, TyCtxt, Visibility};
use rustc::ty::{MethodCall, MethodCallee};
use rustc::ty::adjustment;
use rustc::ty::fold::TypeFoldable;
use rustc::ty::util::{Representability, IntTypeExt};
use require_c_abi_if_variadic;
use rscope::{ElisionFailureInfo, RegionScope};
use session::{Session, CompileResult};
use CrateCtxt;
use TypeAndSubsts;
use lint;
use util::common::{block_query, ErrorReported, indenter, loop_query};
use util::nodemap::{DefIdMap, FnvHashMap, NodeMap};

use std::cell::{Cell, Ref, RefCell};
use std::collections::{HashSet};
use std::mem::replace;
use std::ops::Deref;
use syntax::abi::Abi;
use syntax::ast;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::{self, Spanned};
use syntax::parse::token::{self, InternedString, keywords};
use syntax::ptr::P;
use syntax::util::lev_distance::find_best_match_for_name;
use syntax_pos::{self, Span};
use errors::DiagnosticBuilder;

use rustc::hir::intravisit::{self, Visitor};
use rustc::hir::{self, PatKind};
use rustc::hir::print as pprust;
use rustc_back::slice;
use rustc_const_eval::eval_repeat_count;

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
            ty::TySlice(_) | ty::TyStr | ty::TyTrait(..) => {
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
                    hir::DefaultBlock | hir::PushUnstableBlock | hir:: PopUnstableBlock =>
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

    ret_ty: ty::FnOutput<'tcx>,

    ps: RefCell<UnsafetyState>,

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
    pub fn inherited(&'a self, param_env: Option<ty::ParameterEnvironment<'gcx>>)
                     -> InheritedBuilder<'a, 'gcx, 'tcx> {
        InheritedBuilder {
            ccx: self,
            infcx: self.tcx.infer_ctxt(Some(ty::Tables::empty()),
                                       param_env,
                                       ProjectionMode::AnyFinal)
        }
    }
}

impl<'a, 'gcx, 'tcx> InheritedBuilder<'a, 'gcx, 'tcx> {
    fn enter<F, R>(&'tcx mut self, f: F) -> R
        where F: for<'b> FnOnce(Inherited<'b, 'gcx, 'tcx>) -> R
    {
        let ccx = self.ccx;
        self.infcx.enter(|infcx| {
            f(Inherited {
                ccx: ccx,
                infcx: infcx,
                fulfillment_cx: RefCell::new(traits::FulfillmentContext::new()),
                locals: RefCell::new(NodeMap()),
                deferred_call_resolutions: RefCell::new(DefIdMap()),
                deferred_cast_checks: RefCell::new(Vec::new()),
            })
        })
    }
}

impl<'a, 'gcx, 'tcx> Inherited<'a, 'gcx, 'tcx> {
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
    fn visit_item(&mut self, i: &'tcx hir::Item) {
        check_item_type(self.ccx, i);
        intravisit::walk_item(self, i);
    }

    fn visit_ty(&mut self, t: &'tcx hir::Ty) {
        match t.node {
            hir::TyFixedLengthVec(_, ref expr) => {
                check_const_in_type(self.ccx, &expr, self.ccx.tcx.types.usize);
            }
            _ => {}
        }

        intravisit::walk_ty(self, t);
    }
}

impl<'a, 'tcx> Visitor<'tcx> for CheckItemBodiesVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &'tcx hir::Item) {
        check_item_body(self.ccx, i);
    }
}

pub fn check_wf_new(ccx: &CrateCtxt) -> CompileResult {
    ccx.tcx.sess.track_errors(|| {
        let mut visit = wfcheck::CheckTypeWellFormedVisitor::new(ccx);
        ccx.tcx.visit_all_items_in_krate(DepNode::WfCheck, &mut visit);
    })
}

pub fn check_item_types(ccx: &CrateCtxt) -> CompileResult {
    ccx.tcx.sess.track_errors(|| {
        let mut visit = CheckItemTypesVisitor { ccx: ccx };
        ccx.tcx.visit_all_items_in_krate(DepNode::TypeckItemType, &mut visit);
    })
}

pub fn check_item_bodies(ccx: &CrateCtxt) -> CompileResult {
    ccx.tcx.sess.track_errors(|| {
        let mut visit = CheckItemBodiesVisitor { ccx: ccx };
        ccx.tcx.visit_all_items_in_krate(DepNode::TypeckItemBody, &mut visit);
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
                           body: &'tcx hir::Block,
                           fn_id: ast::NodeId,
                           fn_span: Span,
                           raw_fty: Ty<'tcx>,
                           param_env: ty::ParameterEnvironment<'tcx>)
{
    let fn_ty = match raw_fty.sty {
        ty::TyFnDef(_, _, f) => f,
        _ => span_bug!(body.span, "check_bare_fn: function type expected")
    };

    ccx.inherited(Some(param_env)).enter(|inh| {
        // Compute the fty from point of view of inside fn.
        let fn_scope = inh.tcx.region_maps.call_site_extent(fn_id, body.id);
        let fn_sig =
            fn_ty.sig.subst(inh.tcx, &inh.parameter_environment.free_substs);
        let fn_sig =
            inh.tcx.liberate_late_bound_regions(fn_scope, &fn_sig);
        let fn_sig =
            inh.normalize_associated_types_in(body.span, body.id, &fn_sig);

        let fcx = check_fn(&inh, fn_ty.unsafety, fn_id, &fn_sig, decl, fn_id, body);

        fcx.select_all_obligations_and_apply_defaults();
        fcx.closure_analyze_fn(body);
        fcx.select_obligations_where_possible();
        fcx.check_casts();
        fcx.select_all_obligations_or_error(); // Casts can introduce new obligations.

        fcx.regionck_fn(fn_id, fn_span, decl, body);
        fcx.resolve_type_vars_in_fn(decl, body);
    });
}

struct GatherLocalsVisitor<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>
}

impl<'a, 'gcx, 'tcx> GatherLocalsVisitor<'a, 'gcx, 'tcx> {
    fn assign(&mut self, _span: Span, nid: ast::NodeId, ty_opt: Option<Ty<'tcx>>) -> Ty<'tcx> {
        match ty_opt {
            None => {
                // infer the variable's type
                let var_ty = self.fcx.next_ty_var();
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
        if let PatKind::Binding(_, ref path1, _) = p.node {
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

    fn visit_block(&mut self, b: &'gcx hir::Block) {
        // non-obvious: the `blk` variable maps to region lb, so
        // we have to keep this up-to-date.  This
        // is... unfortunate.  It'd be nice to not need this.
        intravisit::walk_block(self, b);
    }

    // Since an expr occurs as part of the type fixed size arrays we
    // need to record the type for that node
    fn visit_ty(&mut self, t: &'gcx hir::Ty) {
        match t.node {
            hir::TyFixedLengthVec(ref ty, ref count_expr) => {
                self.visit_ty(&ty);
                self.fcx.check_expr_with_hint(&count_expr, self.fcx.tcx.types.usize);
            }
            hir::TyBareFn(ref function_declaration) => {
                intravisit::walk_fn_decl_nopat(self, &function_declaration.decl);
                walk_list!(self, visit_lifetime_def, &function_declaration.lifetimes);
            }
            _ => intravisit::walk_ty(self, t)
        }
    }

    // Don't descend into the bodies of nested closures
    fn visit_fn(&mut self, _: intravisit::FnKind<'gcx>, _: &'gcx hir::FnDecl,
                _: &'gcx hir::Block, _: Span, _: ast::NodeId) { }
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
                            body: &'gcx hir::Block)
                            -> FnCtxt<'a, 'gcx, 'tcx>
{
    let arg_tys = &fn_sig.inputs;
    let ret_ty = fn_sig.output;

    debug!("check_fn(arg_tys={:?}, ret_ty={:?}, fn_id={})",
           arg_tys,
           ret_ty,
           fn_id);

    // Create the function context.  This is either derived from scratch or,
    // in the case of function expressions, based on the outer context.
    let fcx = FnCtxt::new(inherited, ret_ty, body.id);
    *fcx.ps.borrow_mut() = UnsafetyState::function(unsafety, unsafety_id);

    if let ty::FnConverging(ret_ty) = ret_ty {
        fcx.require_type_is_sized(ret_ty, decl.output.span(), traits::ReturnType);
    }

    debug!("fn-sig-map: fn_id={} fn_sig={:?}", fn_id, fn_sig);

    inherited.tables.borrow_mut().liberated_fn_sigs.insert(fn_id, fn_sig.clone());

    {
        let mut visit = GatherLocalsVisitor { fcx: &fcx, };

        // Add formal parameters.
        for (arg_ty, input) in arg_tys.iter().zip(&decl.inputs) {
            // The type of the argument must be well-formed.
            //
            // NB -- this is now checked in wfcheck, but that
            // currently only results in warnings, so we issue an
            // old-style WF obligation here so that we still get the
            // errors that we used to get.
            fcx.register_old_wf_obligation(arg_ty, input.ty.span, traits::MiscObligation);

            // Create type variables for each argument.
            pat_util::pat_bindings(&input.pat, |_bm, pat_id, sp, _path| {
                let var_ty = visit.assign(sp, pat_id, None);
                fcx.require_type_is_sized(var_ty, sp, traits::VariableType(pat_id));
            });

            // Check the pattern.
            fcx.check_pat(&input.pat, *arg_ty);
        }

        visit.visit_block(body);
    }

    fcx.check_block_with_expected(body, match ret_ty {
        ty::FnConverging(result_type) => ExpectHasType(result_type),
        ty::FnDiverging => NoExpectation
    });

    for (input, arg) in decl.inputs.iter().zip(arg_tys) {
        fcx.write_ty(input.id, arg);
    }

    fcx
}

pub fn check_struct(ccx: &CrateCtxt, id: ast::NodeId, span: Span) {
    let tcx = ccx.tcx;

    check_representable(tcx, span, id, "struct");

    if tcx.lookup_simd(ccx.tcx.map.local_def_id(id)) {
        check_simd(tcx, span, id);
    }
}

pub fn check_item_type<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>, it: &'tcx hir::Item) {
    debug!("check_item_type(it.id={}, it.name={})",
           it.id,
           ccx.tcx.item_path_str(ccx.tcx.map.local_def_id(it.id)));
    let _indenter = indenter();
    match it.node {
      // Consts can play a role in type-checking, so they are included here.
      hir::ItemStatic(_, _, ref e) |
      hir::ItemConst(_, ref e) => check_const(ccx, it.span, &e, it.id),
      hir::ItemEnum(ref enum_definition, _) => {
        check_enum_variants(ccx,
                            it.span,
                            &enum_definition.variants,
                            it.id);
      }
      hir::ItemFn(..) => {} // entirely within check_item_body
      hir::ItemImpl(_, _, _, _, _, ref impl_items) => {
          debug!("ItemImpl {} with id {}", it.name, it.id);
          let impl_def_id = ccx.tcx.map.local_def_id(it.id);
          match ccx.tcx.impl_trait_ref(impl_def_id) {
              Some(impl_trait_ref) => {
                  let trait_def_id = impl_trait_ref.def_id;

                  check_impl_items_against_trait(ccx,
                                                 it.span,
                                                 impl_def_id,
                                                 &impl_trait_ref,
                                                 impl_items);
                  check_on_unimplemented(
                      ccx,
                      &ccx.tcx.lookup_trait_def(trait_def_id).generics,
                      it,
                      ccx.tcx.item_name(trait_def_id));
              }
              None => { }
          }
      }
      hir::ItemTrait(..) => {
        let def_id = ccx.tcx.map.local_def_id(it.id);
        let generics = &ccx.tcx.lookup_trait_def(def_id).generics;
        check_on_unimplemented(ccx, generics, it, it.name);
      }
      hir::ItemStruct(..) => {
        check_struct(ccx, it.id, it.span);
      }
      hir::ItemTy(_, ref generics) => {
        let pty_ty = ccx.tcx.node_id_to_type(it.id);
        check_bounds_are_used(ccx, &generics.ty_params, pty_ty);
      }
      hir::ItemForeignMod(ref m) => {
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
                let pty = ccx.tcx.lookup_item_type(ccx.tcx.map.local_def_id(item.id));
                if !pty.generics.types.is_empty() {
                    let mut err = struct_span_err!(ccx.tcx.sess, item.span, E0044,
                        "foreign items may not have type parameters");
                    span_help!(&mut err, item.span,
                        "consider using specialization instead of \
                        type parameters");
                    err.emit();
                }

                if let hir::ForeignItemFn(ref fn_decl, _) = item.node {
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
      hir::ItemFn(ref decl, _, _, _, _, ref body) => {
        let fn_pty = ccx.tcx.lookup_item_type(ccx.tcx.map.local_def_id(it.id));
        let param_env = ParameterEnvironment::for_item(ccx.tcx, it.id);
        check_bare_fn(ccx, &decl, &body, it.id, it.span, fn_pty.ty, param_env);
      }
      hir::ItemImpl(_, _, _, _, _, ref impl_items) => {
        debug!("ItemImpl {} with id {}", it.name, it.id);

        let impl_pty = ccx.tcx.lookup_item_type(ccx.tcx.map.local_def_id(it.id));

        for impl_item in impl_items {
            match impl_item.node {
                hir::ImplItemKind::Const(_, ref expr) => {
                    check_const(ccx, impl_item.span, &expr, impl_item.id)
                }
                hir::ImplItemKind::Method(ref sig, ref body) => {
                    check_method_body(ccx, &impl_pty.generics, sig, body,
                                      impl_item.id, impl_item.span);
                }
                hir::ImplItemKind::Type(_) => {
                    // Nothing to do here.
                }
            }
        }
      }
      hir::ItemTrait(_, _, _, ref trait_items) => {
        let trait_def = ccx.tcx.lookup_trait_def(ccx.tcx.map.local_def_id(it.id));
        for trait_item in trait_items {
            match trait_item.node {
                hir::ConstTraitItem(_, Some(ref expr)) => {
                    check_const(ccx, trait_item.span, &expr, trait_item.id)
                }
                hir::MethodTraitItem(ref sig, Some(ref body)) => {
                    check_trait_fn_not_const(ccx, trait_item.span, sig.constness);

                    check_method_body(ccx, &trait_def.generics, sig, body,
                                      trait_item.id, trait_item.span);
                }
                hir::MethodTraitItem(ref sig, None) => {
                    check_trait_fn_not_const(ccx, trait_item.span, sig.constness);
                }
                hir::ConstTraitItem(_, None) |
                hir::TypeTraitItem(..) => {
                    // Nothing to do.
                }
            }
        }
      }
      _ => {/* nothing to do */ }
    }
}

fn check_trait_fn_not_const<'a,'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                     span: Span,
                                     constness: hir::Constness)
{
    match constness {
        hir::Constness::NotConst => {
            // good
        }
        hir::Constness::Const => {
            span_err!(ccx.tcx.sess, span, E0379, "trait fns cannot be declared const");
        }
    }
}

fn check_on_unimplemented<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                    generics: &ty::Generics,
                                    item: &hir::Item,
                                    name: ast::Name) {
    if let Some(ref attr) = item.attrs.iter().find(|a| {
        a.check_name("rustc_on_unimplemented")
    }) {
        if let Some(ref istring) = attr.value_str() {
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
                            t.name.as_str() == s
                        }) {
                            Some(_) => (),
                            None => {
                                span_err!(ccx.tcx.sess, attr.span, E0230,
                                                 "there is no type parameter \
                                                          {} on trait {}",
                                                           s, name);
                            }
                        },
                        // `{:1}` and `{}` are not to be used
                        Position::ArgumentIs(_) | Position::ArgumentNext => {
                            span_err!(ccx.tcx.sess, attr.span, E0231,
                                                  "only named substitution \
                                                   parameters are allowed");
                        }
                    }
                }
            }
        } else {
            span_err!(ccx.tcx.sess, attr.span, E0232,
                                  "this attribute must have a value, \
                                   eg `#[rustc_on_unimplemented = \"foo\"]`")
        }
    }
}

/// Type checks a method body.
///
/// # Parameters
///
/// * `item_generics`: generics defined on the impl/trait that contains
///   the method
/// * `self_bound`: bound for the `Self` type parameter, if any
/// * `method`: the method definition
fn check_method_body<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                               item_generics: &ty::Generics<'tcx>,
                               sig: &'tcx hir::MethodSig,
                               body: &'tcx hir::Block,
                               id: ast::NodeId, span: Span) {
    debug!("check_method_body(item_generics={:?}, id={})",
            item_generics, id);
    let param_env = ParameterEnvironment::for_item(ccx.tcx, id);

    let fty = ccx.tcx.node_id_to_type(id);
    debug!("check_method_body: fty={:?}", fty);

    check_bare_fn(ccx, &sig.decl, body, id, span, fty, param_env);
}

fn report_forbidden_specialization<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                             impl_item: &hir::ImplItem,
                                             parent_impl: DefId)
{
    let mut err = struct_span_err!(
        tcx.sess, impl_item.span, E0520,
        "item `{}` is provided by an `impl` that specializes \
         another, but the item in the parent `impl` is not \
         marked `default` and so it cannot be specialized.",
        impl_item.name);

    match tcx.span_of_impl(parent_impl) {
        Ok(span) => {
            err.span_note(span, "parent implementation is here:");
        }
        Err(cname) => {
            err.note(&format!("parent implementation is in crate `{}`", cname));
        }
    }

    err.emit();
}

fn check_specialization_validity<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                           trait_def: &ty::TraitDef<'tcx>,
                                           impl_id: DefId,
                                           impl_item: &hir::ImplItem)
{
    let ancestors = trait_def.ancestors(impl_id);

    let parent = match impl_item.node {
        hir::ImplItemKind::Const(..) => {
            ancestors.const_defs(tcx, impl_item.name).skip(1).next()
                .map(|node_item| node_item.map(|parent| parent.defaultness))
        }
        hir::ImplItemKind::Method(..) => {
            ancestors.fn_defs(tcx, impl_item.name).skip(1).next()
                .map(|node_item| node_item.map(|parent| parent.defaultness))

        }
        hir::ImplItemKind::Type(_) => {
            ancestors.type_defs(tcx, impl_item.name).skip(1).next()
                .map(|node_item| node_item.map(|parent| parent.defaultness))
        }
    };

    if let Some(parent) = parent {
        if parent.item.is_final() {
            report_forbidden_specialization(tcx, impl_item, parent.node.def_id());
        }
    }

}

fn check_impl_items_against_trait<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                            impl_span: Span,
                                            impl_id: DefId,
                                            impl_trait_ref: &ty::TraitRef<'tcx>,
                                            impl_items: &[hir::ImplItem]) {
    // If the trait reference itself is erroneous (so the compilation is going
    // to fail), skip checking the items here -- the `impl_item` table in `tcx`
    // isn't populated for such impls.
    if impl_trait_ref.references_error() { return; }

    // Locate trait definition and items
    let tcx = ccx.tcx;
    let trait_def = tcx.lookup_trait_def(impl_trait_ref.def_id);
    let trait_items = tcx.trait_items(impl_trait_ref.def_id);
    let mut overridden_associated_type = None;

    // Check existing impl methods to see if they are both present in trait
    // and compatible with trait signature
    for impl_item in impl_items {
        let ty_impl_item = ccx.tcx.impl_or_trait_item(ccx.tcx.map.local_def_id(impl_item.id));
        let ty_trait_item = trait_items.iter()
            .find(|ac| ac.name() == ty_impl_item.name());

        // Check that impl definition matches trait definition
        if let Some(ty_trait_item) = ty_trait_item {
            match impl_item.node {
                hir::ImplItemKind::Const(..) => {
                    let impl_const = match ty_impl_item {
                        ty::ConstTraitItem(ref cti) => cti,
                        _ => span_bug!(impl_item.span, "non-const impl-item for const")
                    };

                    // Find associated const definition.
                    if let &ty::ConstTraitItem(ref trait_const) = ty_trait_item {
                        compare_const_impl(ccx,
                                           &impl_const,
                                           impl_item.span,
                                           trait_const,
                                           &impl_trait_ref);
                    } else {
                        span_err!(tcx.sess, impl_item.span, E0323,
                                  "item `{}` is an associated const, \
                                  which doesn't match its trait `{:?}`",
                                  impl_const.name,
                                  impl_trait_ref)
                    }
                }
                hir::ImplItemKind::Method(ref sig, ref body) => {
                    check_trait_fn_not_const(ccx, impl_item.span, sig.constness);

                    let impl_method = match ty_impl_item {
                        ty::MethodTraitItem(ref mti) => mti,
                        _ => span_bug!(impl_item.span, "non-method impl-item for method")
                    };

                    if let &ty::MethodTraitItem(ref trait_method) = ty_trait_item {
                        compare_impl_method(ccx,
                                            &impl_method,
                                            impl_item.span,
                                            body.id,
                                            &trait_method,
                                            &impl_trait_ref);
                    } else {
                        span_err!(tcx.sess, impl_item.span, E0324,
                                  "item `{}` is an associated method, \
                                  which doesn't match its trait `{:?}`",
                                  impl_method.name,
                                  impl_trait_ref)
                    }
                }
                hir::ImplItemKind::Type(_) => {
                    let impl_type = match ty_impl_item {
                        ty::TypeTraitItem(ref tti) => tti,
                        _ => span_bug!(impl_item.span, "non-type impl-item for type")
                    };

                    if let &ty::TypeTraitItem(ref at) = ty_trait_item {
                        if let Some(_) = at.ty {
                            overridden_associated_type = Some(impl_item);
                        }
                    } else {
                        span_err!(tcx.sess, impl_item.span, E0325,
                                  "item `{}` is an associated type, \
                                  which doesn't match its trait `{:?}`",
                                  impl_type.name,
                                  impl_trait_ref)
                    }
                }
            }
        }

        check_specialization_validity(tcx, trait_def, impl_id, impl_item);
    }

    // Check for missing items from trait
    let provided_methods = tcx.provided_trait_methods(impl_trait_ref.def_id);
    let mut missing_items = Vec::new();
    let mut invalidated_items = Vec::new();
    let associated_type_overridden = overridden_associated_type.is_some();
    for trait_item in trait_items.iter() {
        let is_implemented;
        let is_provided;

        match *trait_item {
            ty::ConstTraitItem(ref associated_const) => {
                is_provided = associated_const.has_value;
                is_implemented = impl_items.iter().any(|ii| {
                    match ii.node {
                        hir::ImplItemKind::Const(..) => {
                            ii.name == associated_const.name
                        }
                        _ => false,
                    }
                });
            }
            ty::MethodTraitItem(ref trait_method) => {
                is_provided = provided_methods.iter().any(|m| m.name == trait_method.name);
                is_implemented = trait_def.ancestors(impl_id)
                    .fn_defs(tcx, trait_method.name)
                    .next()
                    .map(|node_item| !node_item.node.is_from_trait())
                    .unwrap_or(false);
            }
            ty::TypeTraitItem(ref trait_assoc_ty) => {
                is_provided = trait_assoc_ty.ty.is_some();
                is_implemented = trait_def.ancestors(impl_id)
                    .type_defs(tcx, trait_assoc_ty.name)
                    .next()
                    .map(|node_item| !node_item.node.is_from_trait())
                    .unwrap_or(false);
            }
        }

        if !is_implemented {
            if !is_provided {
                missing_items.push(trait_item.name());
            } else if associated_type_overridden {
                invalidated_items.push(trait_item.name());
            }
        }
    }

    if !missing_items.is_empty() {
        span_err!(tcx.sess, impl_span, E0046,
            "not all trait items implemented, missing: `{}`",
            missing_items.iter()
                  .map(|name| name.to_string())
                  .collect::<Vec<_>>().join("`, `"))
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

/// Checks a constant appearing in a type. At the moment this is just the
/// length expression in a fixed-length vector, but someday it might be
/// extended to type-level numeric literals.
fn check_const_in_type<'a,'tcx>(ccx: &'a CrateCtxt<'a,'tcx>,
                                expr: &'tcx hir::Expr,
                                expected_type: Ty<'tcx>) {
    ccx.inherited(None).enter(|inh| {
        let fcx = FnCtxt::new(&inh, ty::FnConverging(expected_type), expr.id);
        fcx.check_const_with_ty(expr.span, expr, expected_type);
    });
}

fn check_const<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                        sp: Span,
                        e: &'tcx hir::Expr,
                        id: ast::NodeId) {
    let param_env = ParameterEnvironment::for_item(ccx.tcx, id);
    ccx.inherited(Some(param_env)).enter(|inh| {
        let rty = ccx.tcx.node_id_to_type(id);
        let fcx = FnCtxt::new(&inh, ty::FnConverging(rty), e.id);
        let declty = fcx.tcx.lookup_item_type(ccx.tcx.map.local_def_id(id)).ty;
        fcx.check_const_with_ty(sp, e, declty);
    });
}

/// Checks whether a type can be represented in memory. In particular, it
/// identifies types that contain themselves without indirection through a
/// pointer, which would mean their size is unbounded.
pub fn check_representable<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     sp: Span,
                                     item_id: ast::NodeId,
                                     _designation: &str) -> bool {
    let rty = tcx.node_id_to_type(item_id);

    // Check that it is possible to represent this type. This call identifies
    // (1) types that contain themselves and (2) types that contain a different
    // recursive type. It is only necessary to throw an error on those that
    // contain themselves. For case 2, there must be an inner type that will be
    // caught by case 1.
    match rty.is_representable(tcx, sp) {
        Representability::SelfRecursive => {
            let item_def_id = tcx.map.local_def_id(item_id);
            tcx.recursive_type_with_infinite_size_error(item_def_id).emit();
            return false
        }
        Representability::Representable | Representability::ContainsRecursive => (),
    }
    return true
}

pub fn check_simd<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, sp: Span, id: ast::NodeId) {
    let t = tcx.node_id_to_type(id);
    match t.sty {
        ty::TyStruct(def, substs) => {
            let fields = &def.struct_variant().fields;
            if fields.is_empty() {
                span_err!(tcx.sess, sp, E0075, "SIMD vector cannot be empty");
                return;
            }
            let e = fields[0].ty(tcx, substs);
            if !fields.iter().all(|f| f.ty(tcx, substs) == e) {
                span_err!(tcx.sess, sp, E0076, "SIMD vector should be homogeneous");
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
        span_err!(ccx.tcx.sess, sp, E0084,
            "unsupported representation for zero-variant enum");
    }

    ccx.inherited(None).enter(|inh| {
        let rty = ccx.tcx.node_id_to_type(id);
        let fcx = FnCtxt::new(&inh, ty::FnConverging(rty), id);

        let repr_type_ty = ccx.tcx.enum_repr_type(Some(&hint)).to_ty(ccx.tcx);
        for v in vs {
            if let Some(ref e) = v.node.disr_expr {
                fcx.check_const_with_ty(e.span, e, repr_type_ty);
            }
        }

        let def_id = ccx.tcx.map.local_def_id(id);

        let variants = &ccx.tcx.lookup_adt_def(def_id).variants;
        let mut disr_vals: Vec<ty::Disr> = Vec::new();
        for (v, variant) in vs.iter().zip(variants.iter()) {
            let current_disr_val = variant.disr_val;

            // Check for duplicate discriminant values
            if let Some(i) = disr_vals.iter().position(|&x| x == current_disr_val) {
                let mut err = struct_span_err!(ccx.tcx.sess, v.span, E0081,
                    "discriminant value `{}` already exists", disr_vals[i]);
                let variant_i_node_id = ccx.tcx.map.as_local_node_id(variants[i].did).unwrap();
                span_note!(&mut err, ccx.tcx.map.span(variant_i_node_id),
                    "conflicting discriminant here");
                err.emit();
            }
            disr_vals.push(current_disr_val);
        }
    });

    check_representable(ccx.tcx, sp, id, "enum");
}

impl<'a, 'gcx, 'tcx> AstConv<'gcx, 'tcx> for FnCtxt<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn ast_ty_to_ty_cache(&self) -> &RefCell<NodeMap<Ty<'tcx>>> {
        &self.ast_ty_to_ty_cache
    }

    fn get_item_type_scheme(&self, _: Span, id: DefId)
                            -> Result<ty::TypeScheme<'tcx>, ErrorReported>
    {
        Ok(self.tcx().lookup_item_type(id))
    }

    fn get_trait_def(&self, _: Span, id: DefId)
                     -> Result<&'tcx ty::TraitDef<'tcx>, ErrorReported>
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
                                              if data.0.self_ty().is_param(def.space, def.index) {
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

    fn trait_defines_associated_type_named(&self,
                                           trait_def_id: DefId,
                                           assoc_name: ast::Name)
                                           -> bool
    {
        let trait_def = self.tcx().lookup_trait_def(trait_def_id);
        trait_def.associated_type_names.contains(&assoc_name)
    }

    fn ty_infer(&self,
                ty_param_def: Option<ty::TypeParameterDef<'tcx>>,
                substs: Option<&mut subst::Substs<'tcx>>,
                space: Option<subst::ParamSpace>,
                span: Span) -> Ty<'tcx> {
        // Grab the default doing subsitution
        let default = ty_param_def.and_then(|def| {
            def.default.map(|ty| type_variable::Default {
                ty: ty.subst_spanned(self.tcx(), substs.as_ref().unwrap(), Some(span)),
                origin_span: span,
                def_id: def.default_def_id
            })
        });

        let ty_var = self.next_ty_var_with_default(default);

        // Finally we add the type variable to the substs
        match substs {
            None => ty_var,
            Some(substs) => { substs.types.push(space.unwrap(), ty_var); ty_var }
        }
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
        self.next_region_var(infer::MiscVariable(span))
    }

    fn anon_regions(&self, span: Span, count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<ElisionFailureInfo>>> {
        Ok((0..count).map(|_| {
            self.next_region_var(infer::MiscVariable(span))
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
               rty: ty::FnOutput<'tcx>,
               body_id: ast::NodeId)
               -> FnCtxt<'a, 'gcx, 'tcx> {
        FnCtxt {
            ast_ty_to_ty_cache: RefCell::new(NodeMap()),
            body_id: body_id,
            writeback_errors: Cell::new(false),
            err_count_on_creation: inh.tcx.sess.err_count(),
            ret_ty: rty,
            ps: RefCell::new(UnsafetyState::function(hir::Unsafety::Normal, 0)),
            inh: inh,
        }
    }

    pub fn param_env(&self) -> &ty::ParameterEnvironment<'tcx> {
        &self.parameter_environment
    }

    pub fn sess(&self) -> &Session {
        &self.tcx.sess
    }

    pub fn err_count_since_creation(&self) -> usize {
        self.tcx.sess.err_count() - self.err_count_on_creation
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
                span_err!(self.tcx.sess, span, E0513,
                          "no type for local variable {}",
                          nid);
                self.tcx.types.err
            }
        }
    }

    #[inline]
    pub fn write_ty(&self, node_id: ast::NodeId, ty: Ty<'tcx>) {
        debug!("write_ty({}, {:?}) in fcx {}",
               node_id, ty, self.tag());
        self.tables.borrow_mut().node_types.insert(node_id, ty);
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
                                      derefs: usize) {
        self.write_adjustment(
            node_id,
            adjustment::AdjustDerefRef(adjustment::AutoDerefRef {
                autoderefs: derefs,
                autoref: None,
                unsize: None
            })
        );
    }

    pub fn write_adjustment(&self,
                            node_id: ast::NodeId,
                            adj: adjustment::AutoAdjustment<'tcx>) {
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
    fn instantiate_bounds(&self,
                          span: Span,
                          substs: &Substs<'tcx>,
                          bounds: &ty::GenericPredicates<'tcx>)
                          -> ty::InstantiatedPredicates<'tcx>
    {
        ty::InstantiatedPredicates {
            predicates: self.instantiate_type_scheme(span, substs, &bounds.predicates)
        }
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

    /// Instantiates the type in `did` with the generics in `path` and returns
    /// it (registering the necessary trait obligations along the way).
    ///
    /// Note that this function is only intended to be used with type-paths,
    /// not with value-paths.
    pub fn instantiate_type(&self,
                            did: DefId,
                            path: &hir::Path)
                            -> Ty<'tcx>
    {
        debug!("instantiate_type(did={:?}, path={:?})", did, path);
        let type_scheme =
            self.tcx.lookup_item_type(did);
        let type_predicates =
            self.tcx.lookup_predicates(did);
        let substs = AstConv::ast_path_substs_for_ty(self, self,
                                                     path.span,
                                                     PathParamMode::Optional,
                                                     &type_scheme.generics,
                                                     path.segments.last().unwrap());
        debug!("instantiate_type: ty={:?} substs={:?}", &type_scheme.ty, &substs);
        let bounds =
            self.instantiate_bounds(path.span, &substs, &type_predicates);
        self.add_obligations_for_parameters(
            traits::ObligationCause::new(
                path.span,
                self.body_id,
                traits::ItemObligation(did)),
            &bounds);

        self.instantiate_type_scheme(path.span, &substs, &type_scheme.ty)
    }

    /// Return the dict-like variant corresponding to a given `Def`.
    pub fn def_struct_variant(&self,
                              def: Def,
                              _span: Span)
                              -> Option<(ty::AdtDef<'tcx>, ty::VariantDef<'tcx>)>
    {
        let (adt, variant) = match def {
            Def::Variant(enum_id, variant_id) => {
                let adt = self.tcx.lookup_adt_def(enum_id);
                (adt, adt.variant_with_id(variant_id))
            }
            Def::Struct(did) | Def::TyAlias(did) => {
                let typ = self.tcx.lookup_item_type(did);
                if let ty::TyStruct(adt, _) = typ.ty.sty {
                    (adt, adt.struct_variant())
                } else {
                    return None;
                }
            }
            _ => return None
        };

        let var_kind = variant.kind();
        if var_kind == ty::VariantKind::Struct {
            Some((adt, variant))
        } else if var_kind == ty::VariantKind::Unit {
             Some((adt, variant))
         } else {
             None
         }
    }

    pub fn write_nil(&self, node_id: ast::NodeId) {
        self.write_ty(node_id, self.tcx.mk_nil());
    }
    pub fn write_error(&self, node_id: ast::NodeId) {
        self.write_ty(node_id, self.tcx.types.err);
    }

    pub fn require_type_meets(&self,
                              ty: Ty<'tcx>,
                              span: Span,
                              code: traits::ObligationCauseCode<'tcx>,
                              bound: ty::BuiltinBound)
    {
        self.register_builtin_bound(
            ty,
            bound,
            traits::ObligationCause::new(span, self.body_id, code));
    }

    pub fn require_type_is_sized(&self,
                                 ty: Ty<'tcx>,
                                 span: Span,
                                 code: traits::ObligationCauseCode<'tcx>)
    {
        self.require_type_meets(ty, span, code, ty::BoundSized);
    }

    pub fn require_expr_have_sized_type(&self,
                                        expr: &hir::Expr,
                                        code: traits::ObligationCauseCode<'tcx>)
    {
        self.require_type_is_sized(self.expr_ty(expr), expr.span, code);
    }

    pub fn register_builtin_bound(&self,
                                  ty: Ty<'tcx>,
                                  builtin_bound: ty::BuiltinBound,
                                  cause: traits::ObligationCause<'tcx>)
    {
        self.fulfillment_cx.borrow_mut()
            .register_builtin_bound(self, ty, builtin_bound, cause);
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

    pub fn to_ty(&self, ast_t: &hir::Ty) -> Ty<'tcx> {
        let t = AstConv::ast_ty_to_ty(self, self, ast_t);
        self.register_wf_obligation(t, ast_t.span, traits::MiscObligation);
        t
    }

    pub fn expr_ty(&self, ex: &hir::Expr) -> Ty<'tcx> {
        match self.tables.borrow().node_types.get(&ex.id) {
            Some(&t) => t,
            None => {
                bug!("no type for expr in fcx {}", self.tag());
            }
        }
    }

    /// Apply `adjustment` to the type of `expr`
    pub fn adjust_expr_ty(&self,
                          expr: &hir::Expr,
                          adjustment: Option<&adjustment::AutoAdjustment<'tcx>>)
                          -> Ty<'tcx>
    {
        let raw_ty = self.expr_ty(expr);
        let raw_ty = self.shallow_resolve(raw_ty);
        let resolve_ty = |ty: Ty<'tcx>| self.resolve_type_vars_if_possible(&ty);
        raw_ty.adjust(self.tcx, expr.span, expr.id, adjustment, |method_call| {
            self.tables.borrow().method_map.get(&method_call)
                                        .map(|method| resolve_ty(method.ty))
        })
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

    pub fn item_substs(&self) -> Ref<NodeMap<ty::ItemSubsts<'tcx>>> {
        // NOTE: @jroesch this is hack that appears to be fixed on nightly, will monitor if
        // it changes when we upgrade the snapshot compiler
        fn project_item_susbts<'a, 'tcx>(tables: &'a ty::Tables<'tcx>)
                                        -> &'a NodeMap<ty::ItemSubsts<'tcx>> {
            &tables.item_substs
        }

        Ref::map(self.tables.borrow(), project_item_susbts)
    }

    pub fn opt_node_ty_substs<F>(&self,
                                 id: ast::NodeId,
                                 f: F) where
        F: FnOnce(&ty::ItemSubsts<'tcx>),
    {
        match self.tables.borrow().item_substs.get(&id) {
            Some(s) => { f(s) }
            None => { }
        }
    }

    /// Registers an obligation for checking later, during regionck, that the type `ty` must
    /// outlive the region `r`.
    pub fn register_region_obligation(&self,
                                      ty: Ty<'tcx>,
                                      region: ty::Region,
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
        self.register_region_obligation(ty, ty::ReEmpty, cause);
    }

    /// Registers obligations that all types appearing in `substs` are well-formed.
    pub fn add_wf_bounds(&self, substs: &Substs<'tcx>, expr: &hir::Expr)
    {
        for &ty in &substs.types {
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
                    field: ty::FieldDef<'tcx>,
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
                debug!("default_type_parameters: defaulting `{:?}` to `()` because it diverges",
                       resolved);
                self.demand_eqtype(syntax_pos::DUMMY_SP, *ty, self.tcx.mk_nil());
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
            let default_map: FnvHashMap<_, _> =
                unsolved_variables
                    .iter()
                    .filter_map(|t| self.default(t).map(|d| (t, d)))
                    .collect();

            let mut unbound_tyvars = HashSet::new();

            debug!("select_all_obligations_and_apply_defaults: defaults={:?}", default_map);

            // We loop over the unsolved variables, resolving them and if they are
            // and unconstrainted numeric type we add them to the set of unbound
            // variables. We do this so we only apply literal fallback to type
            // variables without defaults.
            for ty in &unsolved_variables {
                let resolved = self.resolve_type_vars_if_possible(ty);
                if self.type_var_diverges(resolved) {
                    self.demand_eqtype(syntax_pos::DUMMY_SP, *ty, self.tcx.mk_nil());
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
                        self.demand_eqtype(syntax_pos::DUMMY_SP, *ty, self.tcx.mk_nil());
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
                                            TypeOrigin::Misc(default.origin_span),
                                            ty, default.ty) {
                                        Ok(InferOk { obligations, .. }) => {
                                            // FIXME(#32730) propagate obligations
                                            assert!(obligations.is_empty())
                                        },
                                        Err(_) => {
                                            conflicts.push((*ty, default));
                                        }
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
                                ty: self.next_ty_var(),
                                origin_span: syntax_pos::DUMMY_SP,
                                def_id: self.tcx.map.local_def_id(0) // what do I put here?
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
                                unbound_vars: &HashSet<Ty<'tcx>>,
                                default_map: &FnvHashMap<&Ty<'tcx>, type_variable::Default<'tcx>>,
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
                self.demand_eqtype(syntax_pos::DUMMY_SP, *ty, self.tcx.mk_nil());
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
                                    TypeOrigin::Misc(default.origin_span),
                                    ty, default.ty) {
                                // FIXME(#32730) propagate obligations
                                Ok(InferOk { obligations, .. }) => assert!(obligations.is_empty()),
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
        match fulfillment_cx.select_all_or_error(self) {
            Ok(()) => { }
            Err(errors) => { self.report_fulfillment_errors(&errors); }
        }

        if let Err(ref errors) = fulfillment_cx.select_rfc1592_obligations(self) {
            self.report_fulfillment_errors_as_warnings(errors, self.body_id);
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
        let ret_ty = self.tcx.no_late_bound_regions(&ret_ty).unwrap().unwrap();

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

        let input_ty = self.next_ty_var();

        // First, try built-in indexing.
        match (adjusted_ty.builtin_index(), &index_ty.sty) {
            (Some(ty), &ty::TyUint(ast::UintTy::Us)) | (Some(ty), &ty::TyInfer(ty::IntVar(_))) => {
                debug!("try_index_step: success, using built-in indexing");
                // If we had `[T; N]`, we should've caught it before unsizing to `[T]`.
                assert!(!unsize);
                self.write_autoderef_adjustment(base_expr.id, autoderefs);
                return Some((tcx.types.usize, ty));
            }
            _ => {}
        }

        // Try `IndexMut` first, if preferred.
        let method = match (lvalue_pref, tcx.lang_items.index_mut_trait()) {
            (PreferMutLvalue, Some(trait_did)) => {
                self.lookup_method_in_trait_adjusted(expr.span,
                                                     Some(&base_expr),
                                                     token::intern("index_mut"),
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
                                                     token::intern("index"),
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
                                   args_no_rcvr: &'gcx [P<hir::Expr>],
                                   tuple_arguments: TupleArgumentsFlag,
                                   expected: Expectation<'tcx>)
                                   -> ty::FnOutput<'tcx> {
        if method_fn_ty.references_error() {
            let err_inputs = self.err_args(args_no_rcvr.len());

            let err_inputs = match tuple_arguments {
                DontTupleArguments => err_inputs,
                TupleArguments => vec![self.tcx.mk_tup(err_inputs)],
            };

            self.check_argument_types(sp, &err_inputs[..], &[], args_no_rcvr,
                                      false, tuple_arguments);
            ty::FnConverging(self.tcx.types.err)
        } else {
            match method_fn_ty.sty {
                ty::TyFnDef(_, _, ref fty) => {
                    // HACK(eddyb) ignore self in the definition (see above).
                    let expected_arg_tys = self.expected_types_for_fn_args(sp, expected,
                                                                           fty.sig.0.output,
                                                                           &fty.sig.0.inputs[1..]);
                    self.check_argument_types(sp, &fty.sig.0.inputs[1..], &expected_arg_tys[..],
                                              args_no_rcvr, fty.sig.0.variadic, tuple_arguments);
                    fty.sig.0.output
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
                            args: &'gcx [P<hir::Expr>],
                            variadic: bool,
                            tuple_arguments: TupleArgumentsFlag) {
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

        fn parameter_count_error<'tcx>(sess: &Session, sp: Span, fn_inputs: &[Ty<'tcx>],
                                       expected_count: usize, arg_count: usize, error_code: &str,
                                       variadic: bool) {
            let mut err = sess.struct_span_err_with_code(sp,
                &format!("this function takes {}{} parameter{} but {} parameter{} supplied",
                    if variadic {"at least "} else {""},
                    expected_count,
                    if expected_count == 1 {""} else {"s"},
                    arg_count,
                    if arg_count == 1 {" was"} else {"s were"}),
                error_code);
            let input_types = fn_inputs.iter().map(|i| format!("{:?}", i)).collect::<Vec<String>>();
            if input_types.len() > 0 {
                err.note(&format!("the following parameter type{} expected: {}",
                        if expected_count == 1 {" was"} else {"s were"},
                        input_types.join(", ")));
            }
            err.emit();
        }

        let formal_tys = if tuple_arguments == TupleArguments {
            let tuple_type = self.structurally_resolved_type(sp, fn_inputs[0]);
            match tuple_type.sty {
                ty::TyTuple(arg_types) if arg_types.len() != args.len() => {
                    parameter_count_error(tcx.sess, sp, fn_inputs, arg_types.len(), args.len(),
                                          "E0057", false);
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
                parameter_count_error(tcx.sess, sp, fn_inputs, expected_arg_count,
                                      supplied_arg_count, "E0060", true);
                expected_arg_tys = &[];
                self.err_args(supplied_arg_count)
            }
        } else {
            parameter_count_error(tcx.sess, sp, fn_inputs, expected_arg_count, supplied_arg_count,
                                  "E0061", false);
            expected_arg_tys = &[];
            self.err_args(supplied_arg_count)
        };

        debug!("check_argument_types: formal_tys={:?}",
               formal_tys.iter().map(|t| self.ty_to_string(*t)).collect::<Vec<String>>());

        // Check the arguments.
        // We do this in a pretty awful way: first we typecheck any arguments
        // that are not anonymous functions, then we typecheck the anonymous
        // functions. This is so that we have more information about the types
        // of arguments when we typecheck the functions. This isn't really the
        // right way to do this.
        let xs = [false, true];
        let mut any_diverges = false; // has any of the arguments diverged?
        let mut warned = false; // have we already warned about unreachable code?
        for check_blocks in &xs {
            let check_blocks = *check_blocks;
            debug!("check_blocks={}", check_blocks);

            // More awful hacks: before we check argument types, try to do
            // an "opportunistic" vtable resolution of any trait bounds on
            // the call. This helps coercions.
            if check_blocks {
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
                if any_diverges && !warned {
                    self.tcx
                        .sess
                        .add_lint(lint::builtin::UNREACHABLE_CODE,
                                  arg.id,
                                  arg.span,
                                  "unreachable expression".to_string());
                    warned = true;
                }
                let is_block = match arg.node {
                    hir::ExprClosure(..) => true,
                    _ => false
                };

                if is_block == check_blocks {
                    debug!("checking the argument");
                    let formal_ty = formal_tys[i];

                    // The special-cased logic below has three functions:
                    // 1. Provide as good of an expected type as possible.
                    let expected = expected_arg_tys.get(i).map(|&ty| {
                        Expectation::rvalue_hint(self, ty)
                    });

                    self.check_expr_with_expectation(&arg,
                        expected.unwrap_or(ExpectHasType(formal_ty)));
                    // 2. Coerce to the most detailed type that could be coerced
                    //    to, which is `expected_ty` if `rvalue_hint` returns an
                    //    `ExpectHasType(expected_ty)`, or the `formal_ty` otherwise.
                    let coerce_ty = expected.and_then(|e| e.only_has_type(self));
                    self.demand_coerce(&arg, coerce_ty.unwrap_or(formal_ty));

                    // 3. Relate the expected type and the formal one,
                    //    if the expected type was used for the coercion.
                    coerce_ty.map(|ty| self.demand_suptype(arg.span, formal_ty, ty));
                }

                if let Some(&arg_ty) = self.tables.borrow().node_types.get(&arg.id) {
                    any_diverges = any_diverges || self.type_var_diverges(arg_ty);
                }
            }
            if any_diverges && !warned {
                let parent = self.tcx.map.get_parent_node(args[0].id);
                self.tcx
                    .sess
                    .add_lint(lint::builtin::UNREACHABLE_CODE,
                              parent,
                              sp,
                              "unreachable call".to_string());
                warned = true;
            }

        }

        // We also need to make sure we at least write the ty of the other
        // arguments which we skipped above.
        if variadic {
            for arg in args.iter().skip(expected_arg_count) {
                self.check_expr(&arg);

                // There are a few types which get autopromoted when passed via varargs
                // in C but we just error out instead and require explicit casts.
                let arg_ty = self.structurally_resolved_type(arg.span,
                                                             self.expr_ty(&arg));
                match arg_ty.sty {
                    ty::TyFloat(ast::FloatTy::F32) => {
                        self.type_error_message(arg.span, |t| {
                            format!("can't pass an `{}` to variadic \
                                     function, cast to `c_double`", t)
                        }, arg_ty, None);
                    }
                    ty::TyInt(ast::IntTy::I8) | ty::TyInt(ast::IntTy::I16) | ty::TyBool => {
                        self.type_error_message(arg.span, |t| {
                            format!("can't pass `{}` to variadic \
                                     function, cast to `c_int`",
                                           t)
                        }, arg_ty, None);
                    }
                    ty::TyUint(ast::UintTy::U8) | ty::TyUint(ast::UintTy::U16) => {
                        self.type_error_message(arg.span, |t| {
                            format!("can't pass `{}` to variadic \
                                     function, cast to `c_uint`",
                                           t)
                        }, arg_ty, None);
                    }
                    ty::TyFnDef(_, _, f) => {
                        let ptr_ty = self.tcx.mk_fn_ptr(f);
                        let ptr_ty = self.resolve_type_vars_if_possible(&ptr_ty);
                        self.type_error_message(arg.span,
                                                |t| {
                            format!("can't pass `{}` to variadic \
                                     function, cast to `{}`", t, ptr_ty)
                        }, arg_ty, None);
                    }
                    _ => {}
                }
            }
        }
    }

    fn err_args(&self, len: usize) -> Vec<Ty<'tcx>> {
        (0..len).map(|_| self.tcx.types.err).collect()
    }

    fn write_call(&self,
                  call_expr: &hir::Expr,
                  output: ty::FnOutput<'tcx>) {
        self.write_ty(call_expr.id, match output {
            ty::FnConverging(output_ty) => output_ty,
            ty::FnDiverging => self.next_diverging_ty_var()
        });
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
        self.check_expr_with_hint(expr, expected);
        self.demand_eqtype(expr.span, expected, self.expr_ty(expr));
    }

    pub fn check_expr_has_type(&self,
                               expr: &'gcx hir::Expr,
                               expected: Ty<'tcx>) {
        self.check_expr_with_hint(expr, expected);
        self.demand_suptype(expr.span, expected, self.expr_ty(expr));
    }

    fn check_expr_coercable_to_type(&self,
                                    expr: &'gcx hir::Expr,
                                    expected: Ty<'tcx>) {
        self.check_expr_with_hint(expr, expected);
        self.demand_coerce(expr, expected);
    }

    fn check_expr_with_hint(&self, expr: &'gcx hir::Expr,
                            expected: Ty<'tcx>) {
        self.check_expr_with_expectation(expr, ExpectHasType(expected))
    }

    fn check_expr_with_expectation(&self,
                                   expr: &'gcx hir::Expr,
                                   expected: Expectation<'tcx>) {
        self.check_expr_with_expectation_and_lvalue_pref(expr, expected, NoPreference)
    }

    fn check_expr(&self, expr: &'gcx hir::Expr)  {
        self.check_expr_with_expectation(expr, NoExpectation)
    }

    fn check_expr_with_lvalue_pref(&self, expr: &'gcx hir::Expr,
                                   lvalue_pref: LvaluePreference)  {
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
        let tcx = self.tcx;

        let ity = tcx.lookup_item_type(did);
        let (tps, rps, raw_ty) =
            (ity.generics.types.get_slice(subst::TypeSpace),
             ity.generics.regions.get_slice(subst::TypeSpace),
             ity.ty);

        debug!("impl_self_ty: tps={:?} rps={:?} raw_ty={:?}", tps, rps, raw_ty);

        let rps = self.region_vars_for_defs(span, rps);
        let mut substs = subst::Substs::new(
            VecPerParamSpace::empty(),
            VecPerParamSpace::new(rps, Vec::new(), Vec::new()));
        self.type_vars_for_defs(span, ParamSpace::TypeSpace, &mut substs, tps);
        let substd_ty = self.instantiate_type_scheme(span, &substs, &raw_ty);

        TypeAndSubsts { substs: substs, ty: substd_ty }
    }

    /// Unifies the return type with the expected type early, for more coercions
    /// and forward type information on the argument expressions.
    fn expected_types_for_fn_args(&self,
                                  call_span: Span,
                                  expected_ret: Expectation<'tcx>,
                                  formal_ret: ty::FnOutput<'tcx>,
                                  formal_args: &[Ty<'tcx>])
                                  -> Vec<Ty<'tcx>> {
        let expected_args = expected_ret.only_has_type(self).and_then(|ret_ty| {
            if let ty::FnConverging(formal_ret_ty) = formal_ret {
                self.commit_regions_if_ok(|| {
                    // Attempt to apply a subtyping relationship between the formal
                    // return type (likely containing type variables if the function
                    // is polymorphic) and the expected return type.
                    // No argument expectations are produced if unification fails.
                    let origin = TypeOrigin::Misc(call_span);
                    let ures = self.sub_types(false, origin, formal_ret_ty, ret_ty);
                    // FIXME(#15760) can't use try! here, FromError doesn't default
                    // to identity so the resulting type is not constrained.
                    match ures {
                        // FIXME(#32730) propagate obligations
                        Ok(InferOk { obligations, .. }) => assert!(obligations.is_empty()),
                        Err(e) => return Err(e),
                    }

                    // Record all the argument types, with the substitutions
                    // produced from the above subtyping unification.
                    Ok(formal_args.iter().map(|ty| {
                        self.resolve_type_vars_if_possible(ty)
                    }).collect())
                }).ok()
            } else {
                None
            }
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
                         args: &'gcx [P<hir::Expr>],
                         tps: &[P<hir::Ty>],
                         expected: Expectation<'tcx>,
                         lvalue_pref: LvaluePreference) {
        let rcvr = &args[0];
        self.check_expr_with_lvalue_pref(&rcvr, lvalue_pref);

        // no need to check for bot/err -- callee does that
        let expr_t = self.structurally_resolved_type(expr.span, self.expr_ty(&rcvr));

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
                    self.report_method_error(method_name.span, expr_t,
                                             method_name.node, Some(rcvr), error);
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

        self.write_call(expr, ret_ty);
    }

    // A generic function for checking the then and else in an if
    // or if-else.
    fn check_then_else(&self,
                       cond_expr: &'gcx hir::Expr,
                       then_blk: &'gcx hir::Block,
                       opt_else_expr: Option<&'gcx hir::Expr>,
                       id: ast::NodeId,
                       sp: Span,
                       expected: Expectation<'tcx>) {
        self.check_expr_has_type(cond_expr, self.tcx.types.bool);

        let expected = expected.adjust_for_branches(self);
        self.check_block_with_expected(then_blk, expected);
        let then_ty = self.node_ty(then_blk.id);

        let unit = self.tcx.mk_nil();
        let (origin, expected, found, result) =
        if let Some(else_expr) = opt_else_expr {
            self.check_expr_with_expectation(else_expr, expected);
            let else_ty = self.expr_ty(else_expr);
            let origin = TypeOrigin::IfExpression(sp);

            // Only try to coerce-unify if we have a then expression
            // to assign coercions to, otherwise it's () or diverging.
            let result = if let Some(ref then) = then_blk.expr {
                let res = self.try_find_coercion_lub(origin, || Some(&**then),
                                                     then_ty, else_expr);

                // In case we did perform an adjustment, we have to update
                // the type of the block, because old trans still uses it.
                let adj = self.tables.borrow().adjustments.get(&then.id).cloned();
                if res.is_ok() && adj.is_some() {
                    self.write_ty(then_blk.id, self.adjust_expr_ty(then, adj.as_ref()));
                }

                res
            } else {
                self.commit_if_ok(|_| {
                    let trace = TypeTrace::types(origin, true, then_ty, else_ty);
                    self.lub(true, trace, &then_ty, &else_ty)
                        .map(|InferOk { value, obligations }| {
                            // FIXME(#32730) propagate obligations
                            assert!(obligations.is_empty());
                            value
                        })
                })
            };
            (origin, then_ty, else_ty, result)
        } else {
            let origin = TypeOrigin::IfExpressionWithNoElse(sp);
            (origin, unit, then_ty,
             self.eq_types(true, origin, unit, then_ty)
                 .map(|InferOk { obligations, .. }| {
                     // FIXME(#32730) propagate obligations
                     assert!(obligations.is_empty());
                     unit
                 }))
        };

        let if_ty = match result {
            Ok(ty) => {
                if self.expr_ty(cond_expr).references_error() {
                    self.tcx.types.err
                } else {
                    ty
                }
            }
            Err(e) => {
                self.report_mismatched_types(origin, expected, found, e);
                self.tcx.types.err
            }
        };

        self.write_ty(id, if_ty);
    }

    // Check field access expressions
    fn check_field(&self,
                   expr: &'gcx hir::Expr,
                   lvalue_pref: LvaluePreference,
                   base: &'gcx hir::Expr,
                   field: &Spanned<ast::Name>) {
        self.check_expr_with_lvalue_pref(base, lvalue_pref);
        let expr_t = self.structurally_resolved_type(expr.span,
                                                     self.expr_ty(base));
        let mut private_candidate = None;
        let mut autoderef = self.autoderef(expr.span, expr_t);
        while let Some((base_t, autoderefs)) = autoderef.next() {
            if let ty::TyStruct(base_def, substs) = base_t.sty {
                debug!("struct named {:?}",  base_t);
                if let Some(field) = base_def.struct_variant().find_field_named(field.node) {
                    let field_ty = self.field_ty(expr.span, field, substs);
                    if field.vis.is_accessible_from(self.body_id, &self.tcx().map) {
                        autoderef.finalize(lvalue_pref, Some(base));
                        self.write_ty(expr.id, field_ty);
                        self.write_autoderef_adjustment(base.id, autoderefs);
                        return;
                    }
                    private_candidate = Some((base_def.did, field_ty));
                }
            }
        }
        autoderef.unambiguous_final_ty();

        if let Some((did, field_ty)) = private_candidate {
            let struct_path = self.tcx().item_path_str(did);
            self.write_ty(expr.id, field_ty);
            let msg = format!("field `{}` of struct `{}` is private", field.node, struct_path);
            let mut err = self.tcx().sess.struct_span_err(expr.span, &msg);
            // Also check if an accessible method exists, which is often what is meant.
            if self.method_exists(field.span, field.node, expr_t, expr.id, false) {
                err.note(&format!("a method `{}` also exists, perhaps you wish to call it",
                                  field.node));
            }
            err.emit();
        } else if field.node == keywords::Invalid.name() {
            self.write_error(expr.id);
        } else if self.method_exists(field.span, field.node, expr_t, expr.id, true) {
            self.type_error_struct(field.span, |actual| {
                format!("attempted to take value of method `{}` on type \
                         `{}`", field.node, actual)
            }, expr_t, None)
                .help(
                       "maybe a `()` to call it is missing? \
                       If not, try an anonymous function")
                .emit();
            self.write_error(expr.id);
        } else {
            let mut err = self.type_error_struct(expr.span, |actual| {
                format!("attempted access of field `{}` on type `{}`, \
                         but no field with that name was found",
                        field.node, actual)
            }, expr_t, None);
            if let ty::TyStruct(def, _) = expr_t.sty {
                Self::suggest_field_names(&mut err, def.struct_variant(), field, vec![]);
            }
            err.emit();
            self.write_error(expr.id);
        }
    }

    // displays hints about the closest matches in field names
    fn suggest_field_names(err: &mut DiagnosticBuilder,
                           variant: ty::VariantDef<'tcx>,
                           field: &Spanned<ast::Name>,
                           skip : Vec<InternedString>) {
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
        if let Some(name) = find_best_match_for_name(names, &name, Some(name.len())) {
            err.span_help(field.span,
                          &format!("did you mean `{}`?", name));
        }
    }

    // Check tuple index expressions
    fn check_tup_field(&self,
                       expr: &'gcx hir::Expr,
                       lvalue_pref: LvaluePreference,
                       base: &'gcx hir::Expr,
                       idx: codemap::Spanned<usize>) {
        self.check_expr_with_lvalue_pref(base, lvalue_pref);
        let expr_t = self.structurally_resolved_type(expr.span,
                                                     self.expr_ty(base));
        let mut private_candidate = None;
        let mut tuple_like = false;
        let mut autoderef = self.autoderef(expr.span, expr_t);
        while let Some((base_t, autoderefs)) = autoderef.next() {
            let field = match base_t.sty {
                ty::TyStruct(base_def, substs) => {
                    tuple_like = base_def.struct_variant().is_tuple_struct();
                    if !tuple_like { continue }

                    debug!("tuple struct named {:?}",  base_t);
                    base_def.struct_variant().fields.get(idx.node).and_then(|field| {
                        let field_ty = self.field_ty(expr.span, field, substs);
                        private_candidate = Some((base_def.did, field_ty));
                        if field.vis.is_accessible_from(self.body_id, &self.tcx().map) {
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
                self.write_ty(expr.id, field_ty);
                self.write_autoderef_adjustment(base.id, autoderefs);
                return;
            }
        }
        autoderef.unambiguous_final_ty();

        if let Some((did, field_ty)) = private_candidate {
            let struct_path = self.tcx().item_path_str(did);
            let msg = format!("field `{}` of struct `{}` is private", idx.node, struct_path);
            self.tcx().sess.span_err(expr.span, &msg);
            self.write_ty(expr.id, field_ty);
            return;
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
            expr_t, None);

        self.write_error(expr.id);
    }

    fn report_unknown_field(&self,
                            ty: Ty<'tcx>,
                            variant: ty::VariantDef<'tcx>,
                            field: &hir::Field,
                            skip_fields: &[hir::Field]) {
        let mut err = self.type_error_struct(
            field.name.span,
            |actual| if let ty::TyEnum(..) = ty.sty {
                format!("struct variant `{}::{}` has no field named `{}`",
                        actual, variant.name.as_str(), field.name.node)
            } else {
                format!("structure `{}` has no field named `{}`",
                        actual, field.name.node)
            },
            ty,
            None);
        // prevent all specified fields from being suggested
        let skip_fields = skip_fields.iter().map(|ref x| x.name.node.as_str());
        Self::suggest_field_names(&mut err, variant, &field.name, skip_fields.collect());
        err.emit();
    }

    fn check_expr_struct_fields(&self,
                                adt_ty: Ty<'tcx>,
                                span: Span,
                                variant: ty::VariantDef<'tcx>,
                                ast_fields: &'gcx [hir::Field],
                                check_completeness: bool) {
        let tcx = self.tcx;
        let substs = match adt_ty.sty {
            ty::TyStruct(_, substs) | ty::TyEnum(_, substs) => substs,
            _ => span_bug!(span, "non-ADT passed to check_expr_struct_fields")
        };

        let mut remaining_fields = FnvHashMap();
        for field in &variant.fields {
            remaining_fields.insert(field.name, field);
        }

        let mut error_happened = false;

        // Typecheck each field.
        for field in ast_fields {
            let expected_field_type;

            if let Some(v_field) = remaining_fields.remove(&field.name.node) {
                expected_field_type = self.field_ty(field.span, v_field, substs);
            } else {
                error_happened = true;
                expected_field_type = tcx.types.err;
                if let Some(_) = variant.find_field_named(field.name.node) {
                    span_err!(self.tcx.sess, field.name.span, E0062,
                        "field `{}` specified more than once",
                        field.name.node);
                } else {
                    self.report_unknown_field(adt_ty, variant, field, ast_fields);
                }
            }

            // Make sure to give a type to the field even if there's
            // an error, so we can continue typechecking
            self.check_expr_coercable_to_type(&field.expr, expected_field_type);
        }

            // Make sure the programmer specified all the fields.
        if check_completeness &&
            !error_happened &&
            !remaining_fields.is_empty()
        {
            span_err!(tcx.sess, span, E0063,
                      "missing field{} {} in initializer of `{}`",
                      if remaining_fields.len() == 1 {""} else {"s"},
                      remaining_fields.keys()
                                      .map(|n| format!("`{}`", n))
                                      .collect::<Vec<_>>()
                                      .join(", "),
                      adt_ty);
        }

    }

    fn check_struct_fields_on_error(&self,
                                    id: ast::NodeId,
                                    fields: &'gcx [hir::Field],
                                    base_expr: &'gcx Option<P<hir::Expr>>) {
        // Make sure to still write the types
        // otherwise we might ICE
        self.write_error(id);
        for field in fields {
            self.check_expr(&field.expr);
        }
        match *base_expr {
            Some(ref base) => self.check_expr(&base),
            None => {}
        }
    }

    fn check_expr_struct(&self,
                         expr: &hir::Expr,
                         path: &hir::Path,
                         fields: &'gcx [hir::Field],
                         base_expr: &'gcx Option<P<hir::Expr>>)
    {
        let tcx = self.tcx;

        // Find the relevant variant
        let def = tcx.expect_def(expr.id);
        if def == Def::Err {
            self.set_tainted_by_errors();
            self.check_struct_fields_on_error(expr.id, fields, base_expr);
            return;
        }
        let variant = match self.def_struct_variant(def, path.span) {
            Some((_, variant)) => variant,
            None => {
                span_err!(self.tcx.sess, path.span, E0071,
                          "`{}` does not name a structure",
                          pprust::path_to_string(path));
                self.check_struct_fields_on_error(expr.id, fields, base_expr);
                return;
            }
        };

        let expr_ty = self.instantiate_type(def.def_id(), path);
        self.write_ty(expr.id, expr_ty);

        self.check_expr_struct_fields(expr_ty, path.span, variant, fields,
                                      base_expr.is_none());
        if let &Some(ref base_expr) = base_expr {
            self.check_expr_has_type(base_expr, expr_ty);
            match expr_ty.sty {
                ty::TyStruct(adt, substs) => {
                    self.tables.borrow_mut().fru_field_types.insert(
                        expr.id,
                        adt.struct_variant().fields.iter().map(|f| {
                            self.normalize_associated_types_in(
                                expr.span, &f.ty(tcx, substs)
                            )
                        }).collect()
                    );
                }
                _ => {
                    span_err!(tcx.sess, base_expr.span, E0436,
                              "functional record update syntax requires a struct");
                }
            }
        }
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
                                                   lvalue_pref: LvaluePreference) {
        debug!(">> typechecking: expr={:?} expected={:?}",
               expr, expected);

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
            self.check_expr_with_expectation(subexpr, expected_inner);
            let referent_ty = self.expr_ty(&subexpr);
            self.write_ty(id, tcx.mk_box(referent_ty));
          }

          hir::ExprLit(ref lit) => {
            let typ = self.check_lit(&lit, expected);
            self.write_ty(id, typ);
          }
          hir::ExprBinary(op, ref lhs, ref rhs) => {
            self.check_binop(expr, op, lhs, rhs);
          }
          hir::ExprAssignOp(op, ref lhs, ref rhs) => {
            self.check_binop_assign(expr, op, lhs, rhs);
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
            self.check_expr_with_expectation_and_lvalue_pref(&oprnd,
                                                             expected_inner,
                                                             lvalue_pref);
            let mut oprnd_t = self.expr_ty(&oprnd);

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
                            }, oprnd_t, None);
                            oprnd_t = tcx.types.err;
                        }
                    }
                    hir::UnNot => {
                        oprnd_t = self.structurally_resolved_type(oprnd.span,
                                                                  oprnd_t);
                        if !(oprnd_t.is_integral() || oprnd_t.sty == ty::TyBool) {
                            oprnd_t = self.check_user_unop("!", "not",
                                                           tcx.lang_items.not_trait(),
                                                           expr, &oprnd, oprnd_t, unop);
                        }
                    }
                    hir::UnNeg => {
                        oprnd_t = self.structurally_resolved_type(oprnd.span,
                                                                  oprnd_t);
                        if !(oprnd_t.is_integral() || oprnd_t.is_fp()) {
                            oprnd_t = self.check_user_unop("-", "neg",
                                                           tcx.lang_items.neg_trait(),
                                                           expr, &oprnd, oprnd_t, unop);
                        }
                    }
                }
            }
            self.write_ty(id, oprnd_t);
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
            self.check_expr_with_expectation_and_lvalue_pref(&oprnd, hint, lvalue_pref);

            let tm = ty::TypeAndMut { ty: self.expr_ty(&oprnd), mutbl: mutbl };
            let oprnd_t = if tm.ty.references_error() {
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
                tcx.mk_ref(tcx.mk_region(region), tm)
            };
            self.write_ty(id, oprnd_t);
          }
          hir::ExprPath(ref maybe_qself, ref path) => {
              let opt_self_ty = maybe_qself.as_ref().map(|qself| {
                  self.to_ty(&qself.ty)
              });

              let path_res = tcx.expect_resolution(id);
              if let Some((opt_ty, segments, def)) =
                      self.resolve_ty_and_def_ufcs(path_res, opt_self_ty, path,
                                                   expr.span, expr.id) {
                  if def != Def::Err {
                      let (scheme, predicates) = self.type_scheme_and_predicates_for_def(expr.span,
                                                                                         def);
                      self.instantiate_path(segments, scheme, &predicates,
                                            opt_ty, def, expr.span, id);
                  } else {
                      self.set_tainted_by_errors();
                      self.write_ty(id, self.tcx.types.err);
                  }
              }

              // We always require that the type provided as the value for
              // a type parameter outlives the moment of instantiation.
              self.opt_node_ty_substs(expr.id, |item_substs| {
                  self.add_wf_bounds(&item_substs.substs, expr);
              });
          }
          hir::ExprInlineAsm(_, ref outputs, ref inputs) => {
              for output in outputs {
                  self.check_expr(output);
              }
              for input in inputs {
                  self.check_expr(input);
              }
              self.write_nil(id);
          }
          hir::ExprBreak(_) => { self.write_ty(id, self.next_diverging_ty_var()); }
          hir::ExprAgain(_) => { self.write_ty(id, self.next_diverging_ty_var()); }
          hir::ExprRet(ref expr_opt) => {
            match self.ret_ty {
                ty::FnConverging(result_type) => {
                    if let Some(ref e) = *expr_opt {
                        self.check_expr_coercable_to_type(&e, result_type);
                    } else {
                        let eq_result = self.eq_types(false,
                                                      TypeOrigin::Misc(expr.span),
                                                      result_type,
                                                      tcx.mk_nil())
                            // FIXME(#32730) propagate obligations
                            .map(|InferOk { obligations, .. }| assert!(obligations.is_empty()));
                        if eq_result.is_err() {
                            span_err!(tcx.sess, expr.span, E0069,
                                      "`return;` in a function whose return type is not `()`");
                        }
                    }
                }
                ty::FnDiverging => {
                    if let Some(ref e) = *expr_opt {
                        self.check_expr(&e);
                    }
                    span_err!(tcx.sess, expr.span, E0166,
                        "`return` in a function declared as diverging");
                }
            }
            self.write_ty(id, self.next_diverging_ty_var());
          }
          hir::ExprAssign(ref lhs, ref rhs) => {
            self.check_expr_with_lvalue_pref(&lhs, PreferMutLvalue);

            let tcx = self.tcx;
            if !tcx.expr_is_lval(&lhs) {
                span_err!(tcx.sess, expr.span, E0070,
                    "invalid left-hand side expression");
            }

            let lhs_ty = self.expr_ty(&lhs);
            self.check_expr_coercable_to_type(&rhs, lhs_ty);
            let rhs_ty = self.expr_ty(&rhs);

            self.require_expr_have_sized_type(&lhs, traits::AssignmentLhsSized);

            if lhs_ty.references_error() || rhs_ty.references_error() {
                self.write_error(id);
            } else {
                self.write_nil(id);
            }
          }
          hir::ExprIf(ref cond, ref then_blk, ref opt_else_expr) => {
            self.check_then_else(&cond, &then_blk, opt_else_expr.as_ref().map(|e| &**e),
                                 id, expr.span, expected);
          }
          hir::ExprWhile(ref cond, ref body, _) => {
            self.check_expr_has_type(&cond, tcx.types.bool);
            self.check_block_no_value(&body);
            let cond_ty = self.expr_ty(&cond);
            let body_ty = self.node_ty(body.id);
            if cond_ty.references_error() || body_ty.references_error() {
                self.write_error(id);
            }
            else {
                self.write_nil(id);
            }
          }
          hir::ExprLoop(ref body, _) => {
            self.check_block_no_value(&body);
            if !may_break(tcx, expr.id, &body) {
                self.write_ty(id, self.next_diverging_ty_var());
            } else {
                self.write_nil(id);
            }
          }
          hir::ExprMatch(ref discrim, ref arms, match_src) => {
            self.check_match(expr, &discrim, arms, expected, match_src);
          }
          hir::ExprClosure(capture, ref decl, ref body, _) => {
              self.check_expr_closure(expr, capture, &decl, &body, expected);
          }
          hir::ExprBlock(ref b) => {
            self.check_block_with_expected(&b, expected);
            self.write_ty(id, self.node_ty(b.id));
          }
          hir::ExprCall(ref callee, ref args) => {
              self.check_call(expr, &callee, &args[..], expected);

              // we must check that return type of called functions is WF:
              let ret_ty = self.expr_ty(expr);
              self.register_wf_obligation(ret_ty, expr.span, traits::MiscObligation);
          }
          hir::ExprMethodCall(name, ref tps, ref args) => {
              self.check_method_call(expr, name, &args[..], &tps[..], expected, lvalue_pref);
              let arg_tys = args.iter().map(|a| self.expr_ty(&a));
              let args_err = arg_tys.fold(false, |rest_err, a| rest_err || a.references_error());
              if args_err {
                  self.write_error(id);
              }
          }
          hir::ExprCast(ref e, ref t) => {
            if let hir::TyFixedLengthVec(_, ref count_expr) = t.node {
                self.check_expr_with_hint(&count_expr, tcx.types.usize);
            }

            // Find the type of `e`. Supply hints based on the type we are casting to,
            // if appropriate.
            let t_cast = self.to_ty(t);
            let t_cast = self.resolve_type_vars_if_possible(&t_cast);
            self.check_expr_with_expectation(e, ExpectCastableToType(t_cast));
            let t_expr = self.expr_ty(e);
            let t_cast = self.resolve_type_vars_if_possible(&t_cast);

            // Eagerly check for some obvious errors.
            if t_expr.references_error() || t_cast.references_error() {
                self.write_error(id);
            } else {
                // Write a type for the whole expression, assuming everything is going
                // to work out Ok.
                self.write_ty(id, t_cast);

                // Defer other checks until we're done type checking.
                let mut deferred_cast_checks = self.deferred_cast_checks.borrow_mut();
                match cast::CastCheck::new(self, e, t_expr, t_cast, t.span, expr.span) {
                    Ok(cast_check) => {
                        deferred_cast_checks.push(cast_check);
                    }
                    Err(ErrorReported) => {
                        self.write_error(id);
                    }
                }
            }
          }
          hir::ExprType(ref e, ref t) => {
            let typ = self.to_ty(&t);
            self.check_expr_eq_type(&e, typ);
            self.write_ty(id, typ);
          }
          hir::ExprVec(ref args) => {
            let uty = expected.to_option(self).and_then(|uty| {
                match uty.sty {
                    ty::TyArray(ty, _) | ty::TySlice(ty) => Some(ty),
                    _ => None
                }
            });

            let mut unified = self.next_ty_var();
            let coerce_to = uty.unwrap_or(unified);

            for (i, e) in args.iter().enumerate() {
                self.check_expr_with_hint(e, coerce_to);
                let e_ty = self.expr_ty(e);
                let origin = TypeOrigin::Misc(e.span);

                // Special-case the first element, as it has no "previous expressions".
                let result = if i == 0 {
                    self.try_coerce(e, coerce_to)
                } else {
                    let prev_elems = || args[..i].iter().map(|e| &**e);
                    self.try_find_coercion_lub(origin, prev_elems, unified, e)
                };

                match result {
                    Ok(ty) => unified = ty,
                    Err(e) => {
                        self.report_mismatched_types(origin, unified, e_ty, e);
                    }
                }
            }
            self.write_ty(id, tcx.mk_array(unified, args.len()));
          }
          hir::ExprRepeat(ref element, ref count_expr) => {
            self.check_expr_has_type(&count_expr, tcx.types.usize);
            let count = eval_repeat_count(self.tcx.global_tcx(), &count_expr);

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
                    let t: Ty = self.next_ty_var();
                    self.check_expr_has_type(&element, t);
                    (self.expr_ty(&element), t)
                }
            };

            if count > 1 {
                // For [foo, ..n] where n > 1, `foo` must have
                // Copy type:
                self.require_type_meets(t, expr.span, traits::RepeatVec, ty::BoundCopy);
            }

            if element_ty.references_error() {
                self.write_error(id);
            } else {
                let t = tcx.mk_array(t, count);
                self.write_ty(id, t);
            }
          }
          hir::ExprTup(ref elts) => {
            let flds = expected.only_has_type(self).and_then(|ty| {
                match ty.sty {
                    ty::TyTuple(ref flds) => Some(&flds[..]),
                    _ => None
                }
            });
            let mut err_field = false;

            let elt_ts = elts.iter().enumerate().map(|(i, e)| {
                let t = match flds {
                    Some(ref fs) if i < fs.len() => {
                        let ety = fs[i];
                        self.check_expr_coercable_to_type(&e, ety);
                        ety
                    }
                    _ => {
                        self.check_expr_with_expectation(&e, NoExpectation);
                        self.expr_ty(&e)
                    }
                };
                err_field = err_field || t.references_error();
                t
            }).collect();
            if err_field {
                self.write_error(id);
            } else {
                let typ = tcx.mk_tup(elt_ts);
                self.write_ty(id, typ);
            }
          }
          hir::ExprStruct(ref path, ref fields, ref base_expr) => {
            self.check_expr_struct(expr, path, fields, base_expr);

            self.require_expr_have_sized_type(expr, traits::StructInitializerSized);
          }
          hir::ExprField(ref base, ref field) => {
            self.check_field(expr, lvalue_pref, &base, field);
          }
          hir::ExprTupField(ref base, idx) => {
            self.check_tup_field(expr, lvalue_pref, &base, idx);
          }
          hir::ExprIndex(ref base, ref idx) => {
              self.check_expr_with_lvalue_pref(&base, lvalue_pref);
              self.check_expr(&idx);

              let base_t = self.expr_ty(&base);
              let idx_t = self.expr_ty(&idx);

              if base_t.references_error() {
                  self.write_ty(id, base_t);
              } else if idx_t.references_error() {
                  self.write_ty(id, idx_t);
              } else {
                  let base_t = self.structurally_resolved_type(expr.span, base_t);
                  match self.lookup_indexing(expr, base, base_t, idx_t, lvalue_pref) {
                      Some((index_ty, element_ty)) => {
                          let idx_expr_ty = self.expr_ty(idx);
                          self.demand_eqtype(expr.span, index_ty, idx_expr_ty);
                          self.write_ty(id, element_ty);
                      }
                      None => {
                          self.check_expr_has_type(&idx, self.tcx.types.err);
                          let mut err = self.type_error_struct(
                              expr.span,
                              |actual| {
                                  format!("cannot index a value of type `{}`",
                                          actual)
                              },
                              base_t,
                              None);
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
                          self.write_ty(id, self.tcx().types.err);
                      }
                  }
              }
           }
        }

        debug!("type of expr({}) {} is...", expr.id,
               pprust::expr_to_string(expr));
        debug!("... {:?}, expected is {:?}",
               self.expr_ty(expr),
               expected);
    }

    pub fn resolve_ty_and_def_ufcs<'b>(&self,
                                       path_res: def::PathResolution,
                                       opt_self_ty: Option<Ty<'tcx>>,
                                       path: &'b hir::Path,
                                       span: Span,
                                       node_id: ast::NodeId)
                                       -> Option<(Option<Ty<'tcx>>, &'b [hir::PathSegment], Def)>
    {

        // If fully resolved already, we don't have to do anything.
        if path_res.depth == 0 {
            Some((opt_self_ty, &path.segments, path_res.base_def))
        } else {
            let def = path_res.base_def;
            let ty_segments = path.segments.split_last().unwrap().1;
            let base_ty_end = path.segments.len() - path_res.depth;
            let (ty, _def) = AstConv::finish_resolving_def_to_ty(self, self, span,
                                                                 PathParamMode::Optional,
                                                                 def,
                                                                 opt_self_ty,
                                                                 node_id,
                                                                 &ty_segments[..base_ty_end],
                                                                 &ty_segments[base_ty_end..]);
            let item_segment = path.segments.last().unwrap();
            let item_name = item_segment.name;
            let def = match self.resolve_ufcs(span, item_name, ty, node_id) {
                Ok(def) => Some(def),
                Err(error) => {
                    let def = match error {
                        method::MethodError::PrivateMatch(def) => Some(def),
                        _ => None,
                    };
                    if item_name != keywords::Invalid.name() {
                        self.report_method_error(span, ty, item_name, None, error);
                    }
                    def
                }
            };

            if let Some(def) = def {
                // Write back the new resolution.
                self.tcx().def_map.borrow_mut().insert(node_id, def::PathResolution::new(def));
                Some((Some(ty), slice::ref_slice(item_segment), def))
            } else {
                self.write_error(node_id);
                None
            }
        }
    }

    pub fn check_decl_initializer(&self,
                                  local: &'gcx hir::Local,
                                  init: &'gcx hir::Expr)
    {
        let ref_bindings = self.tcx.pat_contains_ref_binding(&local.pat);

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
            self.check_expr_with_lvalue_pref(init, LvaluePreference::from_mutbl(m));
            let init_ty = self.expr_ty(init);
            self.demand_eqtype(init.span, init_ty, local_ty);
        } else {
            self.check_expr_coercable_to_type(init, local_ty)
        };
    }

    pub fn check_decl_local(&self, local: &'gcx hir::Local)  {
        let t = self.local_ty(local.span, local.id);
        self.write_ty(local.id, t);

        if let Some(ref init) = local.init {
            self.check_decl_initializer(local, &init);
            let init_ty = self.expr_ty(&init);
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

    pub fn check_stmt(&self, stmt: &'gcx hir::Stmt)  {
        let node_id;
        let mut saw_bot = false;
        let mut saw_err = false;
        match stmt.node {
          hir::StmtDecl(ref decl, id) => {
            node_id = id;
            match decl.node {
              hir::DeclLocal(ref l) => {
                  self.check_decl_local(&l);
                  let l_t = self.node_ty(l.id);
                  saw_bot = saw_bot || self.type_var_diverges(l_t);
                  saw_err = saw_err || l_t.references_error();
              }
              hir::DeclItem(_) => {/* ignore for now */ }
            }
          }
          hir::StmtExpr(ref expr, id) => {
            node_id = id;
            // Check with expected type of ()
            self.check_expr_has_type(&expr, self.tcx.mk_nil());
            let expr_ty = self.expr_ty(&expr);
            saw_bot = saw_bot || self.type_var_diverges(expr_ty);
            saw_err = saw_err || expr_ty.references_error();
          }
          hir::StmtSemi(ref expr, id) => {
            node_id = id;
            self.check_expr(&expr);
            let expr_ty = self.expr_ty(&expr);
            saw_bot |= self.type_var_diverges(expr_ty);
            saw_err |= expr_ty.references_error();
          }
        }
        if saw_bot {
            self.write_ty(node_id, self.next_diverging_ty_var());
        }
        else if saw_err {
            self.write_error(node_id);
        }
        else {
            self.write_nil(node_id)
        }
    }

    pub fn check_block_no_value(&self, blk: &'gcx hir::Block)  {
        self.check_block_with_expected(blk, ExpectHasType(self.tcx.mk_nil()));
        let blkty = self.node_ty(blk.id);
        if blkty.references_error() {
            self.write_error(blk.id);
        } else {
            let nilty = self.tcx.mk_nil();
            self.demand_suptype(blk.span, nilty, blkty);
        }
    }

    fn check_block_with_expected(&self,
                                 blk: &'gcx hir::Block,
                                 expected: Expectation<'tcx>) {
        let prev = {
            let mut fcx_ps = self.ps.borrow_mut();
            let unsafety_state = fcx_ps.recurse(blk);
            replace(&mut *fcx_ps, unsafety_state)
        };

        let mut warned = false;
        let mut any_diverges = false;
        let mut any_err = false;
        for s in &blk.stmts {
            self.check_stmt(s);
            let s_id = s.node.id();
            let s_ty = self.node_ty(s_id);
            if any_diverges && !warned && match s.node {
                hir::StmtDecl(ref decl, _) => {
                    match decl.node {
                        hir::DeclLocal(_) => true,
                        _ => false,
                    }
                }
                hir::StmtExpr(_, _) | hir::StmtSemi(_, _) => true,
            } {
                self.tcx
                    .sess
                    .add_lint(lint::builtin::UNREACHABLE_CODE,
                              s_id,
                              s.span,
                              "unreachable statement".to_string());
                warned = true;
            }
            any_diverges = any_diverges || self.type_var_diverges(s_ty);
            any_err = any_err || s_ty.references_error();
        }
        match blk.expr {
            None => if any_err {
                self.write_error(blk.id);
            } else if any_diverges {
                self.write_ty(blk.id, self.next_diverging_ty_var());
            } else {
                self.write_nil(blk.id);
            },
            Some(ref e) => {
                if any_diverges && !warned {
                    self.tcx
                        .sess
                        .add_lint(lint::builtin::UNREACHABLE_CODE,
                                  e.id,
                                  e.span,
                                  "unreachable expression".to_string());
                }
                let ety = match expected {
                    ExpectHasType(ety) => {
                        self.check_expr_coercable_to_type(&e, ety);
                        ety
                    }
                    _ => {
                        self.check_expr_with_expectation(&e, expected);
                        self.expr_ty(&e)
                    }
                };

                if any_err {
                    self.write_error(blk.id);
                } else if any_diverges {
                    self.write_ty(blk.id, self.next_diverging_ty_var());
                } else {
                    self.write_ty(blk.id, ety);
                }
            }
        };

        *self.ps.borrow_mut() = prev;
    }


    fn check_const_with_ty(&self,
                           _: Span,
                           e: &'gcx hir::Expr,
                           declty: Ty<'tcx>) {
        // Gather locals in statics (because of block expressions).
        // This is technically unnecessary because locals in static items are forbidden,
        // but prevents type checking from blowing up before const checking can properly
        // emit an error.
        GatherLocalsVisitor { fcx: self }.visit_expr(e);

        self.check_expr_coercable_to_type(e, declty);

        self.select_all_obligations_and_apply_defaults();
        self.closure_analyze_const(e);
        self.select_obligations_where_possible();
        self.check_casts();
        self.select_all_obligations_or_error();

        self.regionck_expr(e);
        self.resolve_type_vars_in_expr(e);
    }

    // Returns the type parameter count and the type for the given definition.
    fn type_scheme_and_predicates_for_def(&self,
                                          sp: Span,
                                          defn: Def)
                                          -> (TypeScheme<'tcx>, GenericPredicates<'tcx>) {
        match defn {
            Def::Local(_, nid) | Def::Upvar(_, nid, _, _) => {
                let typ = self.local_ty(sp, nid);
                (ty::TypeScheme { generics: ty::Generics::empty(), ty: typ },
                 ty::GenericPredicates::empty())
            }
            Def::Fn(id) | Def::Method(id) |
            Def::Static(id, _) | Def::Variant(_, id) |
            Def::Struct(id) | Def::Const(id) | Def::AssociatedConst(id) => {
                (self.tcx.lookup_item_type(id), self.tcx.lookup_predicates(id))
            }
            Def::Trait(_) |
            Def::Enum(..) |
            Def::TyAlias(..) |
            Def::AssociatedTy(..) |
            Def::PrimTy(_) |
            Def::TyParam(..) |
            Def::Mod(..) |
            Def::ForeignMod(..) |
            Def::Label(..) |
            Def::SelfTy(..) |
            Def::Err => {
                span_bug!(sp, "expected value, found {:?}", defn);
            }
        }
    }

    // Instantiates the given path, which must refer to an item with the given
    // number of type parameters and type.
    pub fn instantiate_path(&self,
                            segments: &[hir::PathSegment],
                            type_scheme: TypeScheme<'tcx>,
                            type_predicates: &ty::GenericPredicates<'tcx>,
                            opt_self_ty: Option<Ty<'tcx>>,
                            def: Def,
                            span: Span,
                            node_id: ast::NodeId) {
        debug!("instantiate_path(path={:?}, def={:?}, node_id={}, type_scheme={:?})",
               segments,
               def,
               node_id,
               type_scheme);

        // We need to extract the type parameters supplied by the user in
        // the path `path`. Due to the current setup, this is a bit of a
        // tricky-process; the problem is that resolve only tells us the
        // end-point of the path resolution, and not the intermediate steps.
        // Luckily, we can (at least for now) deduce the intermediate steps
        // just from the end-point.
        //
        // There are basically four cases to consider:
        //
        // 1. Reference to a *type*, such as a struct or enum:
        //
        //        mod a { struct Foo<T> { ... } }
        //
        //    Because we don't allow types to be declared within one
        //    another, a path that leads to a type will always look like
        //    `a::b::Foo<T>` where `a` and `b` are modules. This implies
        //    that only the final segment can have type parameters, and
        //    they are located in the TypeSpace.
        //
        //    *Note:* Generally speaking, references to types don't
        //    actually pass through this function, but rather the
        //    `ast_ty_to_ty` function in `astconv`. However, in the case
        //    of struct patterns (and maybe literals) we do invoke
        //    `instantiate_path` to get the general type of an instance of
        //    a struct. (In these cases, there are actually no type
        //    parameters permitted at present, but perhaps we will allow
        //    them in the future.)
        //
        // 1b. Reference to an enum variant or tuple-like struct:
        //
        //        struct foo<T>(...)
        //        enum E<T> { foo(...) }
        //
        //    In these cases, the parameters are declared in the type
        //    space.
        //
        // 2. Reference to a *fn item*:
        //
        //        fn foo<T>() { }
        //
        //    In this case, the path will again always have the form
        //    `a::b::foo::<T>` where only the final segment should have
        //    type parameters. However, in this case, those parameters are
        //    declared on a value, and hence are in the `FnSpace`.
        //
        // 3. Reference to a *method*:
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
        // 4. Reference to an *associated const*:
        //
        // impl<A> AnotherStruct<A> {
        // const FOO: B = BAR;
        // }
        //
        // The path in this case will look like
        // `a::b::AnotherStruct::<A>::FOO`, so the penultimate segment
        // only will have parameters in TypeSpace.
        //
        // The first step then is to categorize the segments appropriately.

        assert!(!segments.is_empty());

        let mut ufcs_associated = None;
        let mut segment_spaces: Vec<_>;
        match def {
            // Case 1 and 1b. Reference to a *type* or *enum variant*.
            Def::SelfTy(..) |
            Def::Struct(..) |
            Def::Variant(..) |
            Def::Enum(..) |
            Def::TyAlias(..) |
            Def::AssociatedTy(..) |
            Def::Trait(..) |
            Def::PrimTy(..) |
            Def::TyParam(..) => {
                // Everything but the final segment should have no
                // parameters at all.
                segment_spaces = vec![None; segments.len() - 1];
                segment_spaces.push(Some(subst::TypeSpace));
            }

            // Case 2. Reference to a top-level value.
            Def::Fn(..) |
            Def::Const(..) |
            Def::Static(..) => {
                segment_spaces = vec![None; segments.len() - 1];
                segment_spaces.push(Some(subst::FnSpace));
            }

            // Case 3. Reference to a method.
            Def::Method(def_id) => {
                let container = self.tcx.impl_or_trait_item(def_id).container();
                match container {
                    ty::TraitContainer(trait_did) => {
                        callee::check_legal_trait_for_method_call(self.ccx, span, trait_did)
                    }
                    ty::ImplContainer(_) => {}
                }

                if segments.len() >= 2 {
                    segment_spaces = vec![None; segments.len() - 2];
                    segment_spaces.push(Some(subst::TypeSpace));
                    segment_spaces.push(Some(subst::FnSpace));
                } else {
                    // `<T>::method` will end up here, and so can `T::method`.
                    let self_ty = opt_self_ty.expect("UFCS sugared method missing Self");
                    segment_spaces = vec![Some(subst::FnSpace)];
                    ufcs_associated = Some((container, self_ty));
                }
            }

            Def::AssociatedConst(def_id) => {
                let container = self.tcx.impl_or_trait_item(def_id).container();
                match container {
                    ty::TraitContainer(trait_did) => {
                        callee::check_legal_trait_for_method_call(self.ccx, span, trait_did)
                    }
                    ty::ImplContainer(_) => {}
                }

                if segments.len() >= 2 {
                    segment_spaces = vec![None; segments.len() - 2];
                    segment_spaces.push(Some(subst::TypeSpace));
                    segment_spaces.push(None);
                } else {
                    // `<T>::CONST` will end up here, and so can `T::CONST`.
                    let self_ty = opt_self_ty.expect("UFCS sugared const missing Self");
                    segment_spaces = vec![None];
                    ufcs_associated = Some((container, self_ty));
                }
            }

            // Other cases. Various nonsense that really shouldn't show up
            // here. If they do, an error will have been reported
            // elsewhere. (I hope)
            Def::Mod(..) |
            Def::ForeignMod(..) |
            Def::Local(..) |
            Def::Label(..) |
            Def::Upvar(..) => {
                segment_spaces = vec![None; segments.len()];
            }

            Def::Err => {
                self.set_tainted_by_errors();
                segment_spaces = vec![None; segments.len()];
            }
        }
        assert_eq!(segment_spaces.len(), segments.len());

        // In `<T as Trait<A, B>>::method`, `A` and `B` are mandatory, but
        // `opt_self_ty` can also be Some for `Foo::method`, where Foo's
        // type parameters are not mandatory.
        let require_type_space = opt_self_ty.is_some() && ufcs_associated.is_none();

        debug!("segment_spaces={:?}", segment_spaces);

        // Next, examine the definition, and determine how many type
        // parameters we expect from each space.
        let type_defs = &type_scheme.generics.types;
        let region_defs = &type_scheme.generics.regions;

        // Now that we have categorized what space the parameters for each
        // segment belong to, let's sort out the parameters that the user
        // provided (if any) into their appropriate spaces. We'll also report
        // errors if type parameters are provided in an inappropriate place.
        let mut substs = Substs::empty();
        for (&opt_space, segment) in segment_spaces.iter().zip(segments) {
            if let Some(space) = opt_space {
                self.push_explicit_parameters_from_segment_to_substs(space,
                                                                     span,
                                                                     type_defs,
                                                                     region_defs,
                                                                     segment,
                                                                     &mut substs);
            } else {
                self.tcx.prohibit_type_params(slice::ref_slice(segment));
            }
        }
        if let Some(self_ty) = opt_self_ty {
            if type_defs.len(subst::SelfSpace) == 1 {
                substs.types.push(subst::SelfSpace, self_ty);
            }
        }

        // Now we have to compare the types that the user *actually*
        // provided against the types that were *expected*. If the user
        // did not provide any types, then we want to substitute inference
        // variables. If the user provided some types, we may still need
        // to add defaults. If the user provided *too many* types, that's
        // a problem.
        for &space in &[subst::SelfSpace, subst::TypeSpace, subst::FnSpace] {
            self.adjust_type_parameters(span, space, type_defs,
                                        require_type_space, &mut substs);
            assert_eq!(substs.types.len(space), type_defs.len(space));

            self.adjust_region_parameters(span, space, region_defs, &mut substs);
            assert_eq!(substs.regions.len(space), region_defs.len(space));
        }

        // The things we are substituting into the type should not contain
        // escaping late-bound regions, and nor should the base type scheme.
        let substs = self.tcx.mk_substs(substs);
        assert!(!substs.has_regions_escaping_depth(0));
        assert!(!type_scheme.has_escaping_regions());

        // Add all the obligations that are required, substituting and
        // normalized appropriately.
        let bounds = self.instantiate_bounds(span, &substs, &type_predicates);
        self.add_obligations_for_parameters(
            traits::ObligationCause::new(span, self.body_id, traits::ItemObligation(def.def_id())),
            &bounds);

        // Substitute the values for the type parameters into the type of
        // the referenced item.
        let ty_substituted = self.instantiate_type_scheme(span, &substs, &type_scheme.ty);


        if let Some((ty::ImplContainer(impl_def_id), self_ty)) = ufcs_associated {
            // In the case of `Foo<T>::method` and `<Foo<T>>::method`, if `method`
            // is inherent, there is no `Self` parameter, instead, the impl needs
            // type parameters, which we can infer by unifying the provided `Self`
            // with the substituted impl type.
            let impl_scheme = self.tcx.lookup_item_type(impl_def_id);
            assert_eq!(substs.types.len(subst::TypeSpace),
                       impl_scheme.generics.types.len(subst::TypeSpace));
            assert_eq!(substs.regions.len(subst::TypeSpace),
                       impl_scheme.generics.regions.len(subst::TypeSpace));

            let impl_ty = self.instantiate_type_scheme(span, &substs, &impl_scheme.ty);
            match self.sub_types(false, TypeOrigin::Misc(span), self_ty, impl_ty) {
                Ok(InferOk { obligations, .. }) => {
                    // FIXME(#32730) propagate obligations
                    assert!(obligations.is_empty());
                }
                Err(_) => {
                    span_bug!(span,
                        "instantiate_path: (UFCS) {:?} was a subtype of {:?} but now is not?",
                        self_ty,
                        impl_ty);
                }
            }
        }

        debug!("instantiate_path: type of {:?} is {:?}",
               node_id,
               ty_substituted);
        self.write_ty(node_id, ty_substituted);
        self.write_substs(node_id, ty::ItemSubsts {
            substs: substs
        });
    }

    /// Finds the parameters that the user provided and adds them to `substs`. If too many
    /// parameters are provided, then reports an error and clears the output vector.
    ///
    /// We clear the output vector because that will cause the `adjust_XXX_parameters()` later to
    /// use inference variables. This seems less likely to lead to derived errors.
    ///
    /// Note that we *do not* check for *too few* parameters here. Due to the presence of defaults
    /// etc that is more complicated. I wanted however to do the reporting of *too many* parameters
    /// here because we can easily use the precise span of the N+1'th parameter.
    fn push_explicit_parameters_from_segment_to_substs(&self,
        space: subst::ParamSpace,
        span: Span,
        type_defs: &VecPerParamSpace<ty::TypeParameterDef<'tcx>>,
        region_defs: &VecPerParamSpace<ty::RegionParameterDef>,
        segment: &hir::PathSegment,
        substs: &mut Substs<'tcx>)
    {
        match segment.parameters {
            hir::AngleBracketedParameters(ref data) => {
                self.push_explicit_angle_bracketed_parameters_from_segment_to_substs(
                    space, type_defs, region_defs, data, substs);
            }

            hir::ParenthesizedParameters(ref data) => {
                span_err!(self.tcx.sess, span, E0238,
                    "parenthesized parameters may only be used with a trait");
                self.push_explicit_parenthesized_parameters_from_segment_to_substs(
                    space, span, type_defs, data, substs);
            }
        }
    }

    fn push_explicit_angle_bracketed_parameters_from_segment_to_substs(&self,
        space: subst::ParamSpace,
        type_defs: &VecPerParamSpace<ty::TypeParameterDef<'tcx>>,
        region_defs: &VecPerParamSpace<ty::RegionParameterDef>,
        data: &hir::AngleBracketedParameterData,
        substs: &mut Substs<'tcx>)
    {
        {
            let type_count = type_defs.len(space);
            assert_eq!(substs.types.len(space), 0);
            for (i, typ) in data.types.iter().enumerate() {
                let t = self.to_ty(&typ);
                if i < type_count {
                    substs.types.push(space, t);
                } else if i == type_count {
                    span_err!(self.tcx.sess, typ.span, E0087,
                        "too many type parameters provided: \
                         expected at most {} parameter{}, \
                         found {} parameter{}",
                         type_count,
                         if type_count == 1 {""} else {"s"},
                         data.types.len(),
                         if data.types.len() == 1 {""} else {"s"});
                    substs.types.truncate(space, 0);
                    break;
                }
            }
        }

        if !data.bindings.is_empty() {
            span_err!(self.tcx.sess, data.bindings[0].span, E0182,
                      "unexpected binding of associated item in expression path \
                       (only allowed in type paths)");
        }

        {
            let region_count = region_defs.len(space);
            assert_eq!(substs.regions.len(space), 0);
            for (i, lifetime) in data.lifetimes.iter().enumerate() {
                let r = ast_region_to_region(self.tcx, lifetime);
                if i < region_count {
                    substs.regions.push(space, r);
                } else if i == region_count {
                    span_err!(self.tcx.sess, lifetime.span, E0088,
                        "too many lifetime parameters provided: \
                         expected {} parameter{}, found {} parameter{}",
                        region_count,
                        if region_count == 1 {""} else {"s"},
                        data.lifetimes.len(),
                        if data.lifetimes.len() == 1 {""} else {"s"});
                    substs.regions.truncate(space, 0);
                    break;
                }
            }
        }
    }

    /// As with
    /// `push_explicit_angle_bracketed_parameters_from_segment_to_substs`,
    /// but intended for `Foo(A,B) -> C` form. This expands to
    /// roughly the same thing as `Foo<(A,B),C>`. One important
    /// difference has to do with the treatment of anonymous
    /// regions, which are translated into bound regions (NYI).
    fn push_explicit_parenthesized_parameters_from_segment_to_substs(&self,
        space: subst::ParamSpace,
        span: Span,
        type_defs: &VecPerParamSpace<ty::TypeParameterDef<'tcx>>,
        data: &hir::ParenthesizedParameterData,
        substs: &mut Substs<'tcx>)
    {
        let type_count = type_defs.len(space);
        if type_count < 2 {
            span_err!(self.tcx.sess, span, E0167,
                      "parenthesized form always supplies 2 type parameters, \
                      but only {} parameter(s) were expected",
                      type_count);
        }

        let input_tys: Vec<Ty> =
            data.inputs.iter().map(|ty| self.to_ty(&ty)).collect();

        let tuple_ty = self.tcx.mk_tup(input_tys);

        if type_count >= 1 {
            substs.types.push(space, tuple_ty);
        }

        let output_ty: Option<Ty> =
            data.output.as_ref().map(|ty| self.to_ty(&ty));

        let output_ty =
            output_ty.unwrap_or(self.tcx.mk_nil());

        if type_count >= 2 {
            substs.types.push(space, output_ty);
        }
    }

    fn adjust_type_parameters(&self,
        span: Span,
        space: ParamSpace,
        defs: &VecPerParamSpace<ty::TypeParameterDef<'tcx>>,
        require_type_space: bool,
        substs: &mut Substs<'tcx>)
    {
        let provided_len = substs.types.len(space);
        let desired = defs.get_slice(space);
        let required_len = desired.iter()
                              .take_while(|d| d.default.is_none())
                              .count();

        debug!("adjust_type_parameters(space={:?}, \
               provided_len={}, \
               desired_len={}, \
               required_len={})",
               space,
               provided_len,
               desired.len(),
               required_len);

        // Enforced by `push_explicit_parameters_from_segment_to_substs()`.
        assert!(provided_len <= desired.len());

        // Nothing specified at all: supply inference variables for
        // everything.
        if provided_len == 0 && !(require_type_space && space == subst::TypeSpace) {
            substs.types.replace(space, Vec::new());
            self.type_vars_for_defs(span, space, substs, &desired[..]);
            return;
        }

        // Too few parameters specified: report an error and use Err
        // for everything.
        if provided_len < required_len {
            let qualifier =
                if desired.len() != required_len { "at least " } else { "" };
            span_err!(self.tcx.sess, span, E0089,
                "too few type parameters provided: expected {}{} parameter{}, \
                 found {} parameter{}",
                qualifier, required_len,
                if required_len == 1 {""} else {"s"},
                provided_len,
                if provided_len == 1 {""} else {"s"});
            substs.types.replace(space, vec![self.tcx.types.err; desired.len()]);
            return;
        }

        // Otherwise, add in any optional parameters that the user
        // omitted. The case of *too many* parameters is handled
        // already by
        // push_explicit_parameters_from_segment_to_substs(). Note
        // that the *default* type are expressed in terms of all prior
        // parameters, so we have to substitute as we go with the
        // partial substitution that we have built up.
        for i in provided_len..desired.len() {
            let default = desired[i].default.unwrap();
            let default = default.subst_spanned(self.tcx, substs, Some(span));
            substs.types.push(space, default);
        }
        assert_eq!(substs.types.len(space), desired.len());

        debug!("Final substs: {:?}", substs);
    }

    fn adjust_region_parameters(&self,
        span: Span,
        space: ParamSpace,
        defs: &VecPerParamSpace<ty::RegionParameterDef>,
        substs: &mut Substs)
    {
        let provided_len = substs.regions.len(space);
        let desired = defs.get_slice(space);

        // Enforced by `push_explicit_parameters_from_segment_to_substs()`.
        assert!(provided_len <= desired.len());

        // If nothing was provided, just use inference variables.
        if provided_len == 0 {
            substs.regions.replace(
                space,
                self.region_vars_for_defs(span, desired));
            return;
        }

        // If just the right number were provided, everybody is happy.
        if provided_len == desired.len() {
            return;
        }

        // Otherwise, too few were provided. Report an error and then
        // use inference variables.
        span_err!(self.tcx.sess, span, E0090,
            "too few lifetime parameters provided: expected {} parameter{}, \
             found {} parameter{}",
            desired.len(),
            if desired.len() == 1 {""} else {"s"},
            provided_len,
            if provided_len == 1 {""} else {"s"});

        substs.regions.replace(
            space,
            self.region_vars_for_defs(span, desired));
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
                    }, ty, None);
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
}

// Returns true if b contains a break that can exit from b
pub fn may_break(tcx: TyCtxt, id: ast::NodeId, b: &hir::Block) -> bool {
    // First: is there an unlabeled break immediately
    // inside the loop?
    (loop_query(&b, |e| {
        match *e {
            hir::ExprBreak(None) => true,
            _ => false
        }
    })) ||
    // Second: is there a labeled break with label
    // <id> nested anywhere inside the loop?
    (block_query(b, |e| {
        if let hir::ExprBreak(Some(_)) = e.node {
            tcx.expect_def(e.id) == Def::Label(id)
        } else {
            false
        }
    }))
}

pub fn check_bounds_are_used<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                       tps: &[hir::TyParam],
                                       ty: Ty<'tcx>) {
    debug!("check_bounds_are_used(n_tps={}, ty={:?})",
           tps.len(),  ty);

    // make a vector of booleans initially false, set to true when used
    if tps.is_empty() { return; }
    let mut tps_used = vec![false; tps.len()];

    for leaf_ty in ty.walk() {
        if let ty::TyParam(ParamTy {idx, ..}) = leaf_ty.sty {
            debug!("Found use of ty param num {}", idx);
            tps_used[idx as usize] = true;
        }
    }

    for (i, b) in tps_used.iter().enumerate() {
        if !*b {
            span_err!(ccx.tcx.sess, tps[i].span, E0091,
                "type parameter `{}` is unused",
                tps[i].name);
        }
    }
}
