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

use astconv::{self, ast_region_to_region, ast_ty_to_ty, AstConv, PathParamMode};
use check::_match::pat_ctxt;
use fmt_macros::{Parser, Piece, Position};
use middle::astconv_util::prohibit_type_params;
use middle::cstore::LOCAL_CRATE;
use middle::def;
use middle::def_id::DefId;
use middle::infer;
use middle::infer::{TypeOrigin, type_variable};
use middle::pat_util::{self, pat_id_map};
use middle::privacy::{AllPublic, LastMod};
use middle::subst::{self, Subst, Substs, VecPerParamSpace, ParamSpace, TypeSpace};
use middle::traits::{self, report_fulfillment_errors};
use middle::ty::{FnSig, GenericPredicates, TypeScheme};
use middle::ty::{Disr, ParamTy, ParameterEnvironment};
use middle::ty::{LvaluePreference, NoPreference, PreferMutLvalue};
use middle::ty::{self, HasTypeFlags, RegionEscape, ToPolyTraitRef, Ty};
use middle::ty::{MethodCall, MethodCallee};
use middle::ty::adjustment;
use middle::ty::error::TypeError;
use middle::ty::fold::{TypeFolder, TypeFoldable};
use middle::ty::util::Representability;
use require_c_abi_if_variadic;
use rscope::{ElisionFailureInfo, RegionScope};
use session::Session;
use {CrateCtxt, lookup_full_def};
use TypeAndSubsts;
use lint;
use util::common::{block_query, ErrorReported, indenter, loop_query};
use util::nodemap::{DefIdMap, FnvHashMap, NodeMap};

use std::cell::{Cell, Ref, RefCell};
use std::collections::{HashSet};
use std::mem::replace;
use syntax::abi;
use syntax::ast;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::{self, Span, Spanned};
use syntax::owned_slice::OwnedSlice;
use syntax::parse::token::{self, InternedString};
use syntax::ptr::P;
use syntax::util::lev_distance::lev_distance;

use rustc_front::intravisit::{self, Visitor};
use rustc_front::hir;
use rustc_front::hir::Visibility;
use rustc_front::hir::{Item, ItemImpl};
use rustc_front::print::pprust;
use rustc_back::slice;

mod assoc;
pub mod dropck;
pub mod _match;
pub mod writeback;
pub mod regionck;
pub mod coercion;
pub mod demand;
pub mod method;
mod upvar;
mod wf;
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
pub struct Inherited<'a, 'tcx: 'a> {
    infcx: infer::InferCtxt<'a, 'tcx>,
    locals: RefCell<NodeMap<Ty<'tcx>>>,

    tables: &'a RefCell<ty::Tables<'tcx>>,

    // When we process a call like `c()` where `c` is a closure type,
    // we may not have decided yet whether `c` is a `Fn`, `FnMut`, or
    // `FnOnce` closure. In that case, we defer full resolution of the
    // call until upvar inference can kick in and make the
    // decision. We keep these deferred resolutions grouped by the
    // def-id of the closure, so that once we decide, we can easily go
    // back and process them.
    deferred_call_resolutions: RefCell<DefIdMap<Vec<DeferredCallResolutionHandler<'tcx>>>>,

    deferred_cast_checks: RefCell<Vec<cast::CastCheck<'tcx>>>,
}

trait DeferredCallResolution<'tcx> {
    fn resolve<'a>(&mut self, fcx: &FnCtxt<'a,'tcx>);
}

type DeferredCallResolutionHandler<'tcx> = Box<DeferredCallResolution<'tcx>+'tcx>;

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

impl<'tcx> Expectation<'tcx> {
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
    fn adjust_for_branches<'a>(&self, fcx: &FnCtxt<'a, 'tcx>) -> Expectation<'tcx> {
        match *self {
            ExpectHasType(ety) => {
                let ety = fcx.infcx().shallow_resolve(ety);
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
pub struct FnCtxt<'a, 'tcx: 'a> {
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

    inh: &'a Inherited<'a, 'tcx>,

    ccx: &'a CrateCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> Inherited<'a, 'tcx> {
    fn new(tcx: &'a ty::ctxt<'tcx>,
           tables: &'a RefCell<ty::Tables<'tcx>>,
           param_env: ty::ParameterEnvironment<'a, 'tcx>)
           -> Inherited<'a, 'tcx> {

        Inherited {
            infcx: infer::new_infer_ctxt(tcx, tables, Some(param_env), true),
            locals: RefCell::new(NodeMap()),
            tables: tables,
            deferred_call_resolutions: RefCell::new(DefIdMap()),
            deferred_cast_checks: RefCell::new(Vec::new()),
        }
    }

    fn normalize_associated_types_in<T>(&self,
                                        span: Span,
                                        body_id: ast::NodeId,
                                        value: &T)
                                        -> T
        where T : TypeFoldable<'tcx> + HasTypeFlags
    {
        let mut fulfillment_cx = self.infcx.fulfillment_cx.borrow_mut();
        assoc::normalize_associated_types_in(&self.infcx,
                                             &mut fulfillment_cx,
                                             span,
                                             body_id,
                                             value)
    }

}

// Used by check_const and check_enum_variants
pub fn blank_fn_ctxt<'a, 'tcx>(ccx: &'a CrateCtxt<'a, 'tcx>,
                               inh: &'a Inherited<'a, 'tcx>,
                               rty: ty::FnOutput<'tcx>,
                               body_id: ast::NodeId)
                               -> FnCtxt<'a, 'tcx> {
    FnCtxt {
        body_id: body_id,
        writeback_errors: Cell::new(false),
        err_count_on_creation: ccx.tcx.sess.err_count(),
        ret_ty: rty,
        ps: RefCell::new(UnsafetyState::function(hir::Unsafety::Normal, 0)),
        inh: inh,
        ccx: ccx
    }
}

fn static_inherited_fields<'a, 'tcx>(ccx: &'a CrateCtxt<'a, 'tcx>,
                                     tables: &'a RefCell<ty::Tables<'tcx>>)
                                    -> Inherited<'a, 'tcx> {
    // It's kind of a kludge to manufacture a fake function context
    // and statement context, but we might as well do write the code only once
    let param_env = ccx.tcx.empty_parameter_environment();
    Inherited::new(ccx.tcx, &tables, param_env)
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
                check_const_in_type(self.ccx, &**expr, self.ccx.tcx.types.usize);
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

pub fn check_wf_old(ccx: &CrateCtxt) {
    // FIXME(#25759). The new code below is much more reliable but (for now)
    // only generates warnings. So as to ensure that we continue
    // getting errors where we used to get errors, we run the old wf
    // code first and abort if it encounters any errors. If no abort
    // comes, we run the new code and issue warnings.
    let krate = ccx.tcx.map.krate();
    let mut visit = wf::CheckTypeWellFormedVisitor::new(ccx);
    krate.visit_all_items(&mut visit);

    // If types are not well-formed, it leads to all manner of errors
    // downstream, so stop reporting errors at this point.
    ccx.tcx.sess.abort_if_errors();
}

pub fn check_wf_new(ccx: &CrateCtxt) {
    let krate = ccx.tcx.map.krate();
    let mut visit = wfcheck::CheckTypeWellFormedVisitor::new(ccx);
    krate.visit_all_items(&mut visit);

    // If types are not well-formed, it leads to all manner of errors
    // downstream, so stop reporting errors at this point.
    ccx.tcx.sess.abort_if_errors();
}

pub fn check_item_types(ccx: &CrateCtxt) {
    let krate = ccx.tcx.map.krate();
    let mut visit = CheckItemTypesVisitor { ccx: ccx };
    krate.visit_all_items(&mut visit);
    ccx.tcx.sess.abort_if_errors();
}

pub fn check_item_bodies(ccx: &CrateCtxt) {
    let krate = ccx.tcx.map.krate();
    let mut visit = CheckItemBodiesVisitor { ccx: ccx };
    krate.visit_all_items(&mut visit);

    ccx.tcx.sess.abort_if_errors();
}

pub fn check_drop_impls(ccx: &CrateCtxt) {
    let drop_trait = match ccx.tcx.lang_items.drop_trait() {
        Some(id) => ccx.tcx.lookup_trait_def(id), None => { return }
    };
    drop_trait.for_each_impl(ccx.tcx, |drop_impl_did| {
        if drop_impl_did.is_local() {
            match dropck::check_drop_impl(ccx.tcx, drop_impl_did) {
                Ok(()) => {}
                Err(()) => {
                    assert!(ccx.tcx.sess.has_errors());
                }
            }
        }
    });

    ccx.tcx.sess.abort_if_errors();
}

fn check_bare_fn<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                           decl: &'tcx hir::FnDecl,
                           body: &'tcx hir::Block,
                           fn_id: ast::NodeId,
                           fn_span: Span,
                           raw_fty: Ty<'tcx>,
                           param_env: ty::ParameterEnvironment<'a, 'tcx>)
{
    match raw_fty.sty {
        ty::TyBareFn(_, ref fn_ty) => {
            let tables = RefCell::new(ty::Tables::empty());
            let inh = Inherited::new(ccx.tcx, &tables, param_env);

            // Compute the fty from point of view of inside fn.
            let fn_scope = ccx.tcx.region_maps.item_extent(body.id);
            let fn_sig =
                fn_ty.sig.subst(ccx.tcx, &inh.infcx.parameter_environment.free_substs);
            let fn_sig =
                ccx.tcx.liberate_late_bound_regions(fn_scope, &fn_sig);
            let fn_sig =
                inh.normalize_associated_types_in(body.span,
                                                  body.id,
                                                  &fn_sig);

            let fcx = check_fn(ccx, fn_ty.unsafety, fn_id, &fn_sig,
                               decl, fn_id, body, &inh);

            fcx.select_all_obligations_and_apply_defaults();
            upvar::closure_analyze_fn(&fcx, fn_id, decl, body);
            fcx.select_obligations_where_possible();
            fcx.check_casts();
            fcx.select_all_obligations_or_error(); // Casts can introduce new obligations.

            regionck::regionck_fn(&fcx, fn_id, fn_span, decl, body);
            writeback::resolve_type_vars_in_fn(&fcx, decl, body);
        }
        _ => ccx.tcx.sess.impossible_case(body.span,
                                 "check_bare_fn: function type expected")
    }
}

struct GatherLocalsVisitor<'a, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'tcx>
}

impl<'a, 'tcx> GatherLocalsVisitor<'a, 'tcx> {
    fn assign(&mut self, _span: Span, nid: ast::NodeId, ty_opt: Option<Ty<'tcx>>) -> Ty<'tcx> {
        match ty_opt {
            None => {
                // infer the variable's type
                let var_ty = self.fcx.infcx().next_ty_var();
                self.fcx.inh.locals.borrow_mut().insert(nid, var_ty);
                var_ty
            }
            Some(typ) => {
                // take type that the user specified
                self.fcx.inh.locals.borrow_mut().insert(nid, typ);
                typ
            }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for GatherLocalsVisitor<'a, 'tcx> {
    // Add explicitly-declared locals.
    fn visit_local(&mut self, local: &'tcx hir::Local) {
        let o_ty = match local.ty {
            Some(ref ty) => Some(self.fcx.to_ty(&**ty)),
            None => None
        };
        self.assign(local.span, local.id, o_ty);
        debug!("Local variable {:?} is assigned type {}",
               local.pat,
               self.fcx.infcx().ty_to_string(
                   self.fcx.inh.locals.borrow().get(&local.id).unwrap().clone()));
        intravisit::walk_local(self, local);
    }

    // Add pattern bindings.
    fn visit_pat(&mut self, p: &'tcx hir::Pat) {
        if let hir::PatIdent(_, ref path1, _) = p.node {
            if pat_util::pat_is_binding(&self.fcx.ccx.tcx.def_map.borrow(), p) {
                let var_ty = self.assign(p.span, p.id, None);

                self.fcx.require_type_is_sized(var_ty, p.span,
                                               traits::VariableType(p.id));

                debug!("Pattern binding {} is assigned to {} with type {:?}",
                       path1.node,
                       self.fcx.infcx().ty_to_string(
                           self.fcx.inh.locals.borrow().get(&p.id).unwrap().clone()),
                       var_ty);
            }
        }
        intravisit::walk_pat(self, p);
    }

    fn visit_block(&mut self, b: &'tcx hir::Block) {
        // non-obvious: the `blk` variable maps to region lb, so
        // we have to keep this up-to-date.  This
        // is... unfortunate.  It'd be nice to not need this.
        intravisit::walk_block(self, b);
    }

    // Since an expr occurs as part of the type fixed size arrays we
    // need to record the type for that node
    fn visit_ty(&mut self, t: &'tcx hir::Ty) {
        match t.node {
            hir::TyFixedLengthVec(ref ty, ref count_expr) => {
                self.visit_ty(&**ty);
                check_expr_with_hint(self.fcx, &**count_expr, self.fcx.tcx().types.usize);
            }
            hir::TyBareFn(ref function_declaration) => {
                intravisit::walk_fn_decl_nopat(self, &function_declaration.decl);
                walk_list!(self, visit_lifetime_def, &function_declaration.lifetimes);
            }
            _ => intravisit::walk_ty(self, t)
        }
    }

    // Don't descend into the bodies of nested closures
    fn visit_fn(&mut self, _: intravisit::FnKind<'tcx>, _: &'tcx hir::FnDecl,
                _: &'tcx hir::Block, _: Span, _: ast::NodeId) { }
}

/// Helper used by check_bare_fn and check_expr_fn. Does the grungy work of checking a function
/// body and returns the function context used for that purpose, since in the case of a fn item
/// there is still a bit more to do.
///
/// * ...
/// * inherited: other fields inherited from the enclosing fn (if any)
fn check_fn<'a, 'tcx>(ccx: &'a CrateCtxt<'a, 'tcx>,
                      unsafety: hir::Unsafety,
                      unsafety_id: ast::NodeId,
                      fn_sig: &ty::FnSig<'tcx>,
                      decl: &'tcx hir::FnDecl,
                      fn_id: ast::NodeId,
                      body: &'tcx hir::Block,
                      inherited: &'a Inherited<'a, 'tcx>)
                      -> FnCtxt<'a, 'tcx>
{
    let tcx = ccx.tcx;
    let err_count_on_creation = tcx.sess.err_count();

    let arg_tys = &fn_sig.inputs;
    let ret_ty = fn_sig.output;

    debug!("check_fn(arg_tys={:?}, ret_ty={:?}, fn_id={})",
           arg_tys,
           ret_ty,
           fn_id);

    // Create the function context.  This is either derived from scratch or,
    // in the case of function expressions, based on the outer context.
    let fcx = FnCtxt {
        body_id: body.id,
        writeback_errors: Cell::new(false),
        err_count_on_creation: err_count_on_creation,
        ret_ty: ret_ty,
        ps: RefCell::new(UnsafetyState::function(unsafety, unsafety_id)),
        inh: inherited,
        ccx: ccx
    };

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
            pat_util::pat_bindings(
                &tcx.def_map,
                &*input.pat,
                |_bm, pat_id, sp, _path| {
                    let var_ty = visit.assign(sp, pat_id, None);
                    fcx.require_type_is_sized(var_ty, sp,
                                              traits::VariableType(pat_id));
                });

            // Check the pattern.
            let pcx = pat_ctxt {
                fcx: &fcx,
                map: pat_id_map(&tcx.def_map, &*input.pat),
            };
            _match::check_pat(&pcx, &*input.pat, *arg_ty);
        }

        visit.visit_block(body);
    }

    check_block_with_expected(&fcx, body, match ret_ty {
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
      hir::ItemConst(_, ref e) => check_const(ccx, it.span, &**e, it.id),
      hir::ItemEnum(ref enum_definition, _) => {
        check_enum_variants(ccx,
                            it.span,
                            &enum_definition.variants,
                            it.id);
      }
      hir::ItemFn(..) => {} // entirely within check_item_body
      hir::ItemImpl(_, _, _, _, _, ref impl_items) => {
          debug!("ItemImpl {} with id {}", it.name, it.id);
          match ccx.tcx.impl_trait_ref(ccx.tcx.map.local_def_id(it.id)) {
              Some(impl_trait_ref) => {
                check_impl_items_against_trait(ccx,
                                               it.span,
                                               &impl_trait_ref,
                                               impl_items);
              }
              None => { }
          }
      }
      hir::ItemTrait(_, ref generics, _, _) => {
        check_trait_on_unimplemented(ccx, generics, it);
      }
      hir::ItemStruct(..) => {
        check_struct(ccx, it.id, it.span);
      }
      hir::ItemTy(ref t, ref generics) => {
        let pty_ty = ccx.tcx.node_id_to_type(it.id);
        check_bounds_are_used(ccx, t.span, &generics.ty_params, pty_ty);
      }
      hir::ItemForeignMod(ref m) => {
        if m.abi == abi::RustIntrinsic {
            for item in &m.items {
                intrinsic::check_intrinsic_type(ccx, &**item);
            }
        } else if m.abi == abi::PlatformIntrinsic {
            for item in &m.items {
                intrinsic::check_platform_intrinsic_type(ccx, &**item);
            }
        } else {
            for item in &m.items {
                let pty = ccx.tcx.lookup_item_type(ccx.tcx.map.local_def_id(item.id));
                if !pty.generics.types.is_empty() {
                    span_err!(ccx.tcx.sess, item.span, E0044,
                        "foreign items may not have type parameters");
                    span_help!(ccx.tcx.sess, item.span,
                        "consider using specialization instead of \
                        type parameters");
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
        check_bare_fn(ccx, &**decl, &**body, it.id, it.span, fn_pty.ty, param_env);
      }
      hir::ItemImpl(_, _, _, _, _, ref impl_items) => {
        debug!("ItemImpl {} with id {}", it.name, it.id);

        let impl_pty = ccx.tcx.lookup_item_type(ccx.tcx.map.local_def_id(it.id));

        for impl_item in impl_items {
            match impl_item.node {
                hir::ImplItemKind::Const(_, ref expr) => {
                    check_const(ccx, impl_item.span, &*expr, impl_item.id)
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
                    check_const(ccx, trait_item.span, &*expr, trait_item.id)
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

fn check_trait_on_unimplemented<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                               generics: &hir::Generics,
                               item: &hir::Item) {
    if let Some(ref attr) = item.attrs.iter().find(|a| {
        a.check_name("rustc_on_unimplemented")
    }) {
        if let Some(ref istring) = attr.value_str() {
            let parser = Parser::new(&istring);
            let types = &*generics.ty_params;
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
                                                           s, item.name);
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

fn check_impl_items_against_trait<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                            impl_span: Span,
                                            impl_trait_ref: &ty::TraitRef<'tcx>,
                                            impl_items: &[P<hir::ImplItem>]) {
    // Locate trait methods
    let tcx = ccx.tcx;
    let trait_items = tcx.trait_items(impl_trait_ref.def_id);
    let mut overridden_associated_type = None;

    // Check existing impl methods to see if they are both present in trait
    // and compatible with trait signature
    for impl_item in impl_items {
        let ty_impl_item = ccx.tcx.impl_or_trait_item(ccx.tcx.map.local_def_id(impl_item.id));
        let ty_trait_item = trait_items.iter()
            .find(|ac| ac.name() == ty_impl_item.name())
            .unwrap_or_else(|| {
                // This is checked by resolve
                tcx.sess.span_bug(impl_item.span,
                                  &format!("impl-item `{}` is not a member of `{:?}`",
                                           ty_impl_item.name(),
                                           impl_trait_ref));
            });
        match impl_item.node {
            hir::ImplItemKind::Const(..) => {
                let impl_const = match ty_impl_item {
                    ty::ConstTraitItem(ref cti) => cti,
                    _ => tcx.sess.span_bug(impl_item.span, "non-const impl-item for const")
                };

                // Find associated const definition.
                if let &ty::ConstTraitItem(ref trait_const) = ty_trait_item {
                    compare_const_impl(ccx.tcx,
                                       &impl_const,
                                       impl_item.span,
                                       trait_const,
                                       &*impl_trait_ref);
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
                    _ => tcx.sess.span_bug(impl_item.span, "non-method impl-item for method")
                };

                if let &ty::MethodTraitItem(ref trait_method) = ty_trait_item {
                    compare_impl_method(ccx.tcx,
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
                    _ => tcx.sess.span_bug(impl_item.span, "non-type impl-item for type")
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

    // Check for missing items from trait
    let provided_methods = tcx.provided_trait_methods(impl_trait_ref.def_id);
    let mut missing_items = Vec::new();
    let mut invalidated_items = Vec::new();
    let associated_type_overridden = overridden_associated_type.is_some();
    for trait_item in trait_items.iter() {
        match *trait_item {
            ty::ConstTraitItem(ref associated_const) => {
                let is_implemented = impl_items.iter().any(|ii| {
                    match ii.node {
                        hir::ImplItemKind::Const(..) => {
                            ii.name == associated_const.name
                        }
                        _ => false,
                    }
                });
                let is_provided = associated_const.has_value;

                if !is_implemented {
                    if !is_provided {
                        missing_items.push(associated_const.name);
                    } else if associated_type_overridden {
                        invalidated_items.push(associated_const.name);
                    }
                }
            }
            ty::MethodTraitItem(ref trait_method) => {
                let is_implemented =
                    impl_items.iter().any(|ii| {
                        match ii.node {
                            hir::ImplItemKind::Method(..) => {
                                ii.name == trait_method.name
                            }
                            _ => false,
                        }
                    });
                let is_provided =
                    provided_methods.iter().any(|m| m.name == trait_method.name);
                if !is_implemented {
                    if !is_provided {
                        missing_items.push(trait_method.name);
                    } else if associated_type_overridden {
                        invalidated_items.push(trait_method.name);
                    }
                }
            }
            ty::TypeTraitItem(ref associated_type) => {
                let is_implemented = impl_items.iter().any(|ii| {
                    match ii.node {
                        hir::ImplItemKind::Type(_) => {
                            ii.name == associated_type.name
                        }
                        _ => false,
                    }
                });
                let is_provided = associated_type.ty.is_some();
                if !is_implemented {
                    if !is_provided {
                        missing_items.push(associated_type.name);
                    } else if associated_type_overridden {
                        invalidated_items.push(associated_type.name);
                    }
                }
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

fn report_cast_to_unsized_type<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                         span: Span,
                                         t_span: Span,
                                         e_span: Span,
                                         t_cast: Ty<'tcx>,
                                         t_expr: Ty<'tcx>,
                                         id: ast::NodeId) {
    let tstr = fcx.infcx().ty_to_string(t_cast);
    fcx.type_error_message(span, |actual| {
        format!("cast to unsized type: `{}` as `{}`", actual, tstr)
    }, t_expr, None);
    match t_expr.sty {
        ty::TyRef(_, ty::TypeAndMut { mutbl: mt, .. }) => {
            let mtstr = match mt {
                hir::MutMutable => "mut ",
                hir::MutImmutable => ""
            };
            if t_cast.is_trait() {
                match fcx.tcx().sess.codemap().span_to_snippet(t_span) {
                    Ok(s) => {
                        fcx.tcx().sess.span_suggestion(t_span,
                                                       "try casting to a reference instead:",
                                                       format!("&{}{}", mtstr, s));
                    },
                    Err(_) =>
                        span_help!(fcx.tcx().sess, t_span,
                                   "did you mean `&{}{}`?", mtstr, tstr),
                }
            } else {
                span_help!(fcx.tcx().sess, span,
                           "consider using an implicit coercion to `&{}{}` instead",
                           mtstr, tstr);
            }
        }
        ty::TyBox(..) => {
            match fcx.tcx().sess.codemap().span_to_snippet(t_span) {
                Ok(s) => {
                    fcx.tcx().sess.span_suggestion(t_span,
                                                   "try casting to a `Box` instead:",
                                                   format!("Box<{}>", s));
                },
                Err(_) =>
                    span_help!(fcx.tcx().sess, t_span, "did you mean `Box<{}>`?", tstr),
            }
        }
        _ => {
            span_help!(fcx.tcx().sess, e_span,
                       "consider using a box or reference as appropriate");
        }
    }
    fcx.write_error(id);
}


impl<'a, 'tcx> AstConv<'tcx> for FnCtxt<'a, 'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> { self.ccx.tcx }

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
        Some(&self.inh.infcx.parameter_environment.free_substs)
    }

    fn get_type_parameter_bounds(&self,
                                 _: Span,
                                 node_id: ast::NodeId)
                                 -> Result<Vec<ty::PolyTraitRef<'tcx>>, ErrorReported>
    {
        let def = self.tcx().type_parameter_def(node_id);
        let r = self.inh.infcx.parameter_environment
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
        let trait_def = self.ccx.tcx.lookup_trait_def(trait_def_id);
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

        let ty_var = self.infcx().next_ty_var_with_default(default);

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
            self.infcx().replace_late_bound_regions_with_fresh_var(
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
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> { self.ccx.tcx }

    pub fn infcx(&self) -> &infer::InferCtxt<'a,'tcx> {
        &self.inh.infcx
    }

    pub fn param_env(&self) -> &ty::ParameterEnvironment<'a,'tcx> {
        &self.inh.infcx.parameter_environment
    }

    pub fn sess(&self) -> &Session {
        &self.tcx().sess
    }

    pub fn err_count_since_creation(&self) -> usize {
        self.ccx.tcx.sess.err_count() - self.err_count_on_creation
    }

    /// Resolves type variables in `ty` if possible. Unlike the infcx
    /// version, this version will also select obligations if it seems
    /// useful, in an effort to get more type information.
    fn resolve_type_vars_if_possible(&self, mut ty: Ty<'tcx>) -> Ty<'tcx> {
        debug!("resolve_type_vars_if_possible(ty={:?})", ty);

        // No TyInfer()? Nothing needs doing.
        if !ty.has_infer_types() {
            debug!("resolve_type_vars_if_possible: ty={:?}", ty);
            return ty;
        }

        // If `ty` is a type variable, see whether we already know what it is.
        ty = self.infcx().resolve_type_vars_if_possible(&ty);
        if !ty.has_infer_types() {
            debug!("resolve_type_vars_if_possible: ty={:?}", ty);
            return ty;
        }

        // If not, try resolving any new fcx obligations that have cropped up.
        self.select_new_obligations();
        ty = self.infcx().resolve_type_vars_if_possible(&ty);
        if !ty.has_infer_types() {
            debug!("resolve_type_vars_if_possible: ty={:?}", ty);
            return ty;
        }

        // If not, try resolving *all* pending obligations as much as
        // possible. This can help substantially when there are
        // indirect dependencies that don't seem worth tracking
        // precisely.
        self.select_obligations_where_possible();
        ty = self.infcx().resolve_type_vars_if_possible(&ty);

        debug!("resolve_type_vars_if_possible: ty={:?}", ty);
        ty
    }

    fn record_deferred_call_resolution(&self,
                                       closure_def_id: DefId,
                                       r: DeferredCallResolutionHandler<'tcx>) {
        let mut deferred_call_resolutions = self.inh.deferred_call_resolutions.borrow_mut();
        deferred_call_resolutions.entry(closure_def_id).or_insert(vec![]).push(r);
    }

    fn remove_deferred_call_resolutions(&self,
                                        closure_def_id: DefId)
                                        -> Vec<DeferredCallResolutionHandler<'tcx>>
    {
        let mut deferred_call_resolutions = self.inh.deferred_call_resolutions.borrow_mut();
        deferred_call_resolutions.remove(&closure_def_id).unwrap_or(Vec::new())
    }

    pub fn tag(&self) -> String {
        let self_ptr: *const FnCtxt = self;
        format!("{:?}", self_ptr)
    }

    pub fn local_ty(&self, span: Span, nid: ast::NodeId) -> Ty<'tcx> {
        match self.inh.locals.borrow().get(&nid) {
            Some(&t) => t,
            None => {
                span_err!(self.tcx().sess, span, E0513,
                          "no type for local variable {}",
                          nid);
                self.tcx().types.err
            }
        }
    }

    #[inline]
    pub fn write_ty(&self, node_id: ast::NodeId, ty: Ty<'tcx>) {
        debug!("write_ty({}, {:?}) in fcx {}",
               node_id, ty, self.tag());
        self.inh.tables.borrow_mut().node_types.insert(node_id, ty);
    }

    pub fn write_substs(&self, node_id: ast::NodeId, substs: ty::ItemSubsts<'tcx>) {
        if !substs.substs.is_noop() {
            debug!("write_substs({}, {:?}) in fcx {}",
                   node_id,
                   substs,
                   self.tag());

            self.inh.tables.borrow_mut().item_substs.insert(node_id, substs);
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

        self.inh.tables.borrow_mut().adjustments.insert(node_id, adj);
    }

    /// Basically whenever we are converting from a type scheme into
    /// the fn body space, we always want to normalize associated
    /// types as well. This function combines the two.
    fn instantiate_type_scheme<T>(&self,
                                  span: Span,
                                  substs: &Substs<'tcx>,
                                  value: &T)
                                  -> T
        where T : TypeFoldable<'tcx> + HasTypeFlags
    {
        let value = value.subst(self.tcx(), substs);
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
        where T : TypeFoldable<'tcx> + HasTypeFlags
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
        self.inh
            .infcx
            .fulfillment_cx
            .borrow_mut()
            .normalize_projection_type(self.infcx(),
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
            self.tcx().lookup_item_type(did);
        let type_predicates =
            self.tcx().lookup_predicates(did);
        let substs = astconv::ast_path_substs_for_ty(self, self,
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
                              def: def::Def,
                              span: Span)
                              -> Option<(ty::AdtDef<'tcx>, ty::VariantDef<'tcx>)>
    {
        let (adt, variant) = match def {
            def::DefVariant(enum_id, variant_id, true) => {
                let adt = self.tcx().lookup_adt_def(enum_id);
                (adt, adt.variant_with_id(variant_id))
            }
            def::DefTy(did, _) | def::DefStruct(did) => {
                let typ = self.tcx().lookup_item_type(did);
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
            if !self.tcx().sess.features.borrow().braced_empty_structs {
                self.tcx().sess.span_err(span, "empty structs and enum variants \
                                                with braces are unstable");
                fileline_help!(self.tcx().sess, span, "add #![feature(braced_empty_structs)] to \
                                                       the crate features to enable");
            }

             Some((adt, variant))
         } else {
             None
         }
    }

    pub fn write_nil(&self, node_id: ast::NodeId) {
        self.write_ty(node_id, self.tcx().mk_nil());
    }
    pub fn write_error(&self, node_id: ast::NodeId) {
        self.write_ty(node_id, self.tcx().types.err);
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

    pub fn type_is_known_to_be_sized(&self,
                                     ty: Ty<'tcx>,
                                     span: Span)
                                     -> bool
    {
        traits::type_known_to_meet_builtin_bound(self.infcx(),
                                                 ty,
                                                 ty::BoundSized,
                                                 span)
    }

    pub fn register_builtin_bound(&self,
                                  ty: Ty<'tcx>,
                                  builtin_bound: ty::BuiltinBound,
                                  cause: traits::ObligationCause<'tcx>)
    {
        self.inh.infcx.fulfillment_cx.borrow_mut()
            .register_builtin_bound(self.infcx(), ty, builtin_bound, cause);
    }

    pub fn register_predicate(&self,
                              obligation: traits::PredicateObligation<'tcx>)
    {
        debug!("register_predicate({:?})",
               obligation);
        self.inh.infcx.fulfillment_cx
            .borrow_mut()
            .register_predicate_obligation(self.infcx(), obligation);
    }

    pub fn to_ty(&self, ast_t: &hir::Ty) -> Ty<'tcx> {
        let t = ast_ty_to_ty(self, self, ast_t);
        self.register_wf_obligation(t, ast_t.span, traits::MiscObligation);
        t
    }

    pub fn expr_ty(&self, ex: &hir::Expr) -> Ty<'tcx> {
        match self.inh.tables.borrow().node_types.get(&ex.id) {
            Some(&t) => t,
            None => {
                self.tcx().sess.bug(&format!("no type for expr in fcx {}",
                                            self.tag()));
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
        let raw_ty = self.infcx().shallow_resolve(raw_ty);
        let resolve_ty = |ty: Ty<'tcx>| self.infcx().resolve_type_vars_if_possible(&ty);
        raw_ty.adjust(self.tcx(), expr.span, expr.id, adjustment, |method_call| {
            self.inh.tables.borrow().method_map.get(&method_call)
                                        .map(|method| resolve_ty(method.ty))
        })
    }

    pub fn node_ty(&self, id: ast::NodeId) -> Ty<'tcx> {
        match self.inh.tables.borrow().node_types.get(&id) {
            Some(&t) => t,
            None if self.err_count_since_creation() != 0 => self.tcx().types.err,
            None => {
                self.tcx().sess.bug(
                    &format!("no type for node {}: {} in fcx {}",
                            id, self.tcx().map.node_to_string(id),
                            self.tag()));
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

        Ref::map(self.inh.tables.borrow(), project_item_susbts)
    }

    pub fn opt_node_ty_substs<F>(&self,
                                 id: ast::NodeId,
                                 f: F) where
        F: FnOnce(&ty::ItemSubsts<'tcx>),
    {
        match self.inh.tables.borrow().item_substs.get(&id) {
            Some(s) => { f(s) }
            None => { }
        }
    }

    pub fn mk_subty(&self,
                    a_is_expected: bool,
                    origin: TypeOrigin,
                    sub: Ty<'tcx>,
                    sup: Ty<'tcx>)
                    -> Result<(), TypeError<'tcx>> {
        infer::mk_subty(self.infcx(), a_is_expected, origin, sub, sup)
    }

    pub fn mk_eqty(&self,
                   a_is_expected: bool,
                   origin: TypeOrigin,
                   sub: Ty<'tcx>,
                   sup: Ty<'tcx>)
                   -> Result<(), TypeError<'tcx>> {
        infer::mk_eqty(self.infcx(), a_is_expected, origin, sub, sup)
    }

    pub fn mk_subr(&self,
                   origin: infer::SubregionOrigin<'tcx>,
                   sub: ty::Region,
                   sup: ty::Region) {
        infer::mk_subr(self.infcx(), origin, sub, sup)
    }

    pub fn type_error_message<M>(&self,
                                 sp: Span,
                                 mk_msg: M,
                                 actual_ty: Ty<'tcx>,
                                 err: Option<&TypeError<'tcx>>) where
        M: FnOnce(String) -> String,
    {
        self.infcx().type_error_message(sp, mk_msg, actual_ty, err);
    }

    pub fn report_mismatched_types(&self,
                                   sp: Span,
                                   e: Ty<'tcx>,
                                   a: Ty<'tcx>,
                                   err: &TypeError<'tcx>) {
        self.infcx().report_mismatched_types(sp, e, a, err)
    }

    /// Registers an obligation for checking later, during regionck, that the type `ty` must
    /// outlive the region `r`.
    pub fn register_region_obligation(&self,
                                      ty: Ty<'tcx>,
                                      region: ty::Region,
                                      cause: traits::ObligationCause<'tcx>)
    {
        let mut fulfillment_cx = self.inh.infcx.fulfillment_cx.borrow_mut();
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
    pub fn field_ty(&self,
                    span: Span,
                    field: ty::FieldDef<'tcx>,
                    substs: &Substs<'tcx>)
                    -> Ty<'tcx>
    {
        self.normalize_associated_types_in(span,
                                           &field.ty(self.tcx(), substs))
    }

    // Only for fields! Returns <none> for methods>
    // Indifferent to privacy flags
    fn check_casts(&self) {
        let mut deferred_cast_checks = self.inh.deferred_cast_checks.borrow_mut();
        for cast in deferred_cast_checks.drain(..) {
            cast.check(self);
        }
    }

    /// Apply "fallbacks" to some types
    /// ! gets replaced with (), unconstrained ints with i32, and unconstrained floats with f64.
    fn default_type_parameters(&self) {
        use middle::ty::error::UnconstrainedNumeric::Neither;
        use middle::ty::error::UnconstrainedNumeric::{UnconstrainedInt, UnconstrainedFloat};
        for ty in &self.infcx().unsolved_variables() {
            let resolved = self.infcx().resolve_type_vars_if_possible(ty);
            if self.infcx().type_var_diverges(resolved) {
                demand::eqtype(self, codemap::DUMMY_SP, *ty, self.tcx().mk_nil());
            } else {
                match self.infcx().type_is_unconstrained_numeric(resolved) {
                    UnconstrainedInt => {
                        demand::eqtype(self, codemap::DUMMY_SP, *ty, self.tcx().types.i32)
                    },
                    UnconstrainedFloat => {
                        demand::eqtype(self, codemap::DUMMY_SP, *ty, self.tcx().types.f64)
                    }
                    Neither => { }
                }
            }
        }
    }

    fn select_all_obligations_and_apply_defaults(&self) {
        if self.tcx().sess.features.borrow().default_type_parameter_fallback {
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
        use middle::ty::error::UnconstrainedNumeric::Neither;
        use middle::ty::error::UnconstrainedNumeric::{UnconstrainedInt, UnconstrainedFloat};

        // For the time being this errs on the side of being memory wasteful but provides better
        // error reporting.
        // let type_variables = self.infcx().type_variables.clone();

        // There is a possibility that this algorithm will have to run an arbitrary number of times
        // to terminate so we bound it by the compiler's recursion limit.
        for _ in 0..self.tcx().sess.recursion_limit.get() {
            // First we try to solve all obligations, it is possible that the last iteration
            // has made it possible to make more progress.
            self.select_obligations_where_possible();

            let mut conflicts = Vec::new();

            // Collect all unsolved type, integral and floating point variables.
            let unsolved_variables = self.inh.infcx.unsolved_variables();

            // We must collect the defaults *before* we do any unification. Because we have
            // directly attached defaults to the type variables any unification that occurs
            // will erase defaults causing conflicting defaults to be completely ignored.
            let default_map: FnvHashMap<_, _> =
                unsolved_variables
                    .iter()
                    .filter_map(|t| self.infcx().default(t).map(|d| (t, d)))
                    .collect();

            let mut unbound_tyvars = HashSet::new();

            debug!("select_all_obligations_and_apply_defaults: defaults={:?}", default_map);

            // We loop over the unsolved variables, resolving them and if they are
            // and unconstrainted numberic type we add them to the set of unbound
            // variables. We do this so we only apply literal fallback to type
            // variables without defaults.
            for ty in &unsolved_variables {
                let resolved = self.infcx().resolve_type_vars_if_possible(ty);
                if self.infcx().type_var_diverges(resolved) {
                    demand::eqtype(self, codemap::DUMMY_SP, *ty, self.tcx().mk_nil());
                } else {
                    match self.infcx().type_is_unconstrained_numeric(resolved) {
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
                    let resolved = self.infcx().resolve_type_vars_if_possible(ty);

                    debug!("select_all_obligations_and_apply_defaults: ty: {:?} with default: {:?}",
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


            let _ = self.infcx().commit_if_ok(|_: &infer::CombinedSnapshot| {
                for ty in &unbound_tyvars {
                    if self.infcx().type_var_diverges(ty) {
                        demand::eqtype(self, codemap::DUMMY_SP, *ty, self.tcx().mk_nil());
                    } else {
                        match self.infcx().type_is_unconstrained_numeric(ty) {
                            UnconstrainedInt => {
                                demand::eqtype(self, codemap::DUMMY_SP, *ty, self.tcx().types.i32)
                            },
                            UnconstrainedFloat => {
                                demand::eqtype(self, codemap::DUMMY_SP, *ty, self.tcx().types.f64)
                            }
                            Neither => {
                                if let Some(default) = default_map.get(ty) {
                                    let default = default.clone();
                                    match infer::mk_eqty(self.infcx(), false,
                                                         TypeOrigin::Misc(default.origin_span),
                                                         ty, default.ty) {
                                        Ok(()) => {}
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
                                ty: self.infcx().next_ty_var(),
                                origin_span: codemap::DUMMY_SP,
                                def_id: self.tcx().map.local_def_id(0) // what do I put here?
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


                    self.infcx().report_conflicting_default_types(
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
        use middle::ty::error::UnconstrainedNumeric::Neither;
        use middle::ty::error::UnconstrainedNumeric::{UnconstrainedInt, UnconstrainedFloat};

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
            if self.infcx().type_var_diverges(ty) {
                demand::eqtype(self, codemap::DUMMY_SP, *ty, self.tcx().mk_nil());
            } else {
                match self.infcx().type_is_unconstrained_numeric(ty) {
                    UnconstrainedInt => {
                        demand::eqtype(self, codemap::DUMMY_SP, *ty, self.tcx().types.i32)
                    },
                    UnconstrainedFloat => {
                        demand::eqtype(self, codemap::DUMMY_SP, *ty, self.tcx().types.f64)
                    },
                    Neither => {
                        if let Some(default) = default_map.get(ty) {
                            let default = default.clone();
                            match infer::mk_eqty(self.infcx(), false,
                                                 TypeOrigin::Misc(default.origin_span),
                                                 ty, default.ty) {
                                Ok(()) => {}
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
        assert!(self.inh.deferred_call_resolutions.borrow().is_empty());

        self.select_all_obligations_and_apply_defaults();

        let mut fulfillment_cx = self.inh.infcx.fulfillment_cx.borrow_mut();
        match fulfillment_cx.select_all_or_error(self.infcx()) {
            Ok(()) => { }
            Err(errors) => { report_fulfillment_errors(self.infcx(), &errors); }
        }
    }

    /// Select as many obligations as we can at present.
    fn select_obligations_where_possible(&self) {
        match
            self.inh.infcx.fulfillment_cx
            .borrow_mut()
            .select_where_possible(self.infcx())
        {
            Ok(()) => { }
            Err(errors) => { report_fulfillment_errors(self.infcx(), &errors); }
        }
    }

    /// Try to select any fcx obligation that we haven't tried yet, in an effort
    /// to improve inference. You could just call
    /// `select_obligations_where_possible` except that it leads to repeated
    /// work.
    fn select_new_obligations(&self) {
        match
            self.inh.infcx.fulfillment_cx
            .borrow_mut()
            .select_new_obligations(self.infcx())
        {
            Ok(()) => { }
            Err(errors) => { report_fulfillment_errors(self.infcx(), &errors); }
        }
    }

}

impl<'a, 'tcx> RegionScope for FnCtxt<'a, 'tcx> {
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
        self.infcx().next_region_var(infer::MiscVariable(span))
    }

    fn anon_regions(&self, span: Span, count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<ElisionFailureInfo>>> {
        Ok((0..count).map(|_| {
            self.infcx().next_region_var(infer::MiscVariable(span))
        }).collect())
    }
}

/// Whether `autoderef` requires types to resolve.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnresolvedTypeAction {
    /// Produce an error and return `TyError` whenever a type cannot
    /// be resolved (i.e. it is `TyInfer`).
    Error,
    /// Go on without emitting any errors, and return the unresolved
    /// type. Useful for probing, e.g. in coercions.
    Ignore
}

/// Executes an autoderef loop for the type `t`. At each step, invokes `should_stop` to decide
/// whether to terminate the loop. Returns the final type and number of derefs that it performed.
///
/// Note: this method does not modify the adjustments table. The caller is responsible for
/// inserting an AutoAdjustment record into the `fcx` using one of the suitable methods.
pub fn autoderef<'a, 'tcx, T, F>(fcx: &FnCtxt<'a, 'tcx>,
                                 sp: Span,
                                 base_ty: Ty<'tcx>,
                                 opt_expr: Option<&hir::Expr>,
                                 unresolved_type_action: UnresolvedTypeAction,
                                 mut lvalue_pref: LvaluePreference,
                                 mut should_stop: F)
                                 -> (Ty<'tcx>, usize, Option<T>)
    where F: FnMut(Ty<'tcx>, usize) -> Option<T>,
{
    debug!("autoderef(base_ty={:?}, opt_expr={:?}, lvalue_pref={:?})",
           base_ty,
           opt_expr,
           lvalue_pref);

    let mut t = base_ty;
    for autoderefs in 0..fcx.tcx().sess.recursion_limit.get() {
        let resolved_t = match unresolved_type_action {
            UnresolvedTypeAction::Error => {
                structurally_resolved_type(fcx, sp, t)
            }
            UnresolvedTypeAction::Ignore => {
                // We can continue even when the type cannot be resolved
                // (i.e. it is an inference variable) because `Ty::builtin_deref`
                // and `try_overloaded_deref` both simply return `None`
                // in such a case without producing spurious errors.
                fcx.resolve_type_vars_if_possible(t)
            }
        };
        if resolved_t.references_error() {
            return (resolved_t, autoderefs, None);
        }

        match should_stop(resolved_t, autoderefs) {
            Some(x) => return (resolved_t, autoderefs, Some(x)),
            None => {}
        }

        // Otherwise, deref if type is derefable:
        let mt = match resolved_t.builtin_deref(false, lvalue_pref) {
            Some(mt) => Some(mt),
            None => {
                let method_call =
                    opt_expr.map(|expr| MethodCall::autoderef(expr.id, autoderefs as u32));

                // Super subtle: it might seem as though we should
                // pass `opt_expr` to `try_overloaded_deref`, so that
                // the (implicit) autoref of using an overloaded deref
                // would get added to the adjustment table. However we
                // do not do that, because it's kind of a
                // "meta-adjustment" -- instead, we just leave it
                // unrecorded and know that there "will be" an
                // autoref. regionck and other bits of the code base,
                // when they encounter an overloaded autoderef, have
                // to do some reconstructive surgery. This is a pretty
                // complex mess that is begging for a proper MIR.
                try_overloaded_deref(fcx, sp, method_call, None, resolved_t, lvalue_pref)
            }
        };
        match mt {
            Some(mt) => {
                t = mt.ty;
                if mt.mutbl == hir::MutImmutable {
                    lvalue_pref = NoPreference;
                }
            }
            None => return (resolved_t, autoderefs, None)
        }
    }

    // We've reached the recursion limit, error gracefully.
    span_err!(fcx.tcx().sess, sp, E0055,
        "reached the recursion limit while auto-dereferencing {:?}",
        base_ty);
    (fcx.tcx().types.err, 0, None)
}

fn try_overloaded_deref<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                  span: Span,
                                  method_call: Option<MethodCall>,
                                  base_expr: Option<&hir::Expr>,
                                  base_ty: Ty<'tcx>,
                                  lvalue_pref: LvaluePreference)
                                  -> Option<ty::TypeAndMut<'tcx>>
{
    // Try DerefMut first, if preferred.
    let method = match (lvalue_pref, fcx.tcx().lang_items.deref_mut_trait()) {
        (PreferMutLvalue, Some(trait_did)) => {
            method::lookup_in_trait(fcx, span, base_expr,
                                    token::intern("deref_mut"), trait_did,
                                    base_ty, None)
        }
        _ => None
    };

    // Otherwise, fall back to Deref.
    let method = match (method, fcx.tcx().lang_items.deref_trait()) {
        (None, Some(trait_did)) => {
            method::lookup_in_trait(fcx, span, base_expr,
                                    token::intern("deref"), trait_did,
                                    base_ty, None)
        }
        (method, _) => method
    };

    make_overloaded_lvalue_return_type(fcx, method_call, method)
}

/// For the overloaded lvalue expressions (`*x`, `x[3]`), the trait returns a type of `&T`, but the
/// actual type we assign to the *expression* is `T`. So this function just peels off the return
/// type by one layer to yield `T`. It also inserts the `method-callee` into the method map.
fn make_overloaded_lvalue_return_type<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                                method_call: Option<MethodCall>,
                                                method: Option<MethodCallee<'tcx>>)
                                                -> Option<ty::TypeAndMut<'tcx>>
{
    match method {
        Some(method) => {
            // extract method return type, which will be &T;
            // all LB regions should have been instantiated during method lookup
            let ret_ty = method.ty.fn_ret();
            let ret_ty = fcx.tcx().no_late_bound_regions(&ret_ty).unwrap().unwrap();

            if let Some(method_call) = method_call {
                fcx.inh.tables.borrow_mut().method_map.insert(method_call, method);
            }

            // method returns &T, but the type as visible to user is T, so deref
            ret_ty.builtin_deref(true, NoPreference)
        }
        None => None,
    }
}

fn lookup_indexing<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                             expr: &hir::Expr,
                             base_expr: &'tcx hir::Expr,
                             base_ty: Ty<'tcx>,
                             idx_ty: Ty<'tcx>,
                             lvalue_pref: LvaluePreference)
                             -> Option<(/*index type*/ Ty<'tcx>, /*element type*/ Ty<'tcx>)>
{
    // FIXME(#18741) -- this is almost but not quite the same as the
    // autoderef that normal method probing does. They could likely be
    // consolidated.

    let (ty, autoderefs, final_mt) = autoderef(fcx,
                                               base_expr.span,
                                               base_ty,
                                               Some(base_expr),
                                               UnresolvedTypeAction::Error,
                                               lvalue_pref,
                                               |adj_ty, idx| {
        try_index_step(fcx, MethodCall::expr(expr.id), expr, base_expr,
                       adj_ty, idx, false, lvalue_pref, idx_ty)
    });

    if final_mt.is_some() {
        return final_mt;
    }

    // After we have fully autoderef'd, if the resulting type is [T; n], then
    // do a final unsized coercion to yield [T].
    if let ty::TyArray(element_ty, _) = ty.sty {
        let adjusted_ty = fcx.tcx().mk_slice(element_ty);
        try_index_step(fcx, MethodCall::expr(expr.id), expr, base_expr,
                       adjusted_ty, autoderefs, true, lvalue_pref, idx_ty)
    } else {
        None
    }
}

/// To type-check `base_expr[index_expr]`, we progressively autoderef (and otherwise adjust)
/// `base_expr`, looking for a type which either supports builtin indexing or overloaded indexing.
/// This loop implements one step in that search; the autoderef loop is implemented by
/// `lookup_indexing`.
fn try_index_step<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                            method_call: MethodCall,
                            expr: &hir::Expr,
                            base_expr: &'tcx hir::Expr,
                            adjusted_ty: Ty<'tcx>,
                            autoderefs: usize,
                            unsize: bool,
                            lvalue_pref: LvaluePreference,
                            index_ty: Ty<'tcx>)
                            -> Option<(/*index type*/ Ty<'tcx>, /*element type*/ Ty<'tcx>)>
{
    let tcx = fcx.tcx();
    debug!("try_index_step(expr={:?}, base_expr.id={:?}, adjusted_ty={:?}, \
                           autoderefs={}, unsize={}, index_ty={:?})",
           expr,
           base_expr,
           adjusted_ty,
           autoderefs,
           unsize,
           index_ty);

    let input_ty = fcx.infcx().next_ty_var();

    // First, try built-in indexing.
    match (adjusted_ty.builtin_index(), &index_ty.sty) {
        (Some(ty), &ty::TyUint(ast::TyUs)) | (Some(ty), &ty::TyInfer(ty::IntVar(_))) => {
            debug!("try_index_step: success, using built-in indexing");
            // If we had `[T; N]`, we should've caught it before unsizing to `[T]`.
            assert!(!unsize);
            fcx.write_autoderef_adjustment(base_expr.id, autoderefs);
            return Some((tcx.types.usize, ty));
        }
        _ => {}
    }

    // Try `IndexMut` first, if preferred.
    let method = match (lvalue_pref, tcx.lang_items.index_mut_trait()) {
        (PreferMutLvalue, Some(trait_did)) => {
            method::lookup_in_trait_adjusted(fcx,
                                             expr.span,
                                             Some(&*base_expr),
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
            method::lookup_in_trait_adjusted(fcx,
                                             expr.span,
                                             Some(&*base_expr),
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
    method.and_then(|method| {
        debug!("try_index_step: success, using overloaded indexing");
        make_overloaded_lvalue_return_type(fcx, Some(method_call), Some(method)).
            map(|ret| (input_ty, ret.ty))
    })
}

fn check_method_argument_types<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                         sp: Span,
                                         method_fn_ty: Ty<'tcx>,
                                         callee_expr: &'tcx hir::Expr,
                                         args_no_rcvr: &'tcx [P<hir::Expr>],
                                         tuple_arguments: TupleArgumentsFlag,
                                         expected: Expectation<'tcx>)
                                         -> ty::FnOutput<'tcx> {
    if method_fn_ty.references_error() {
        let err_inputs = err_args(fcx.tcx(), args_no_rcvr.len());

        let err_inputs = match tuple_arguments {
            DontTupleArguments => err_inputs,
            TupleArguments => vec![fcx.tcx().mk_tup(err_inputs)],
        };

        check_argument_types(fcx,
                             sp,
                             &err_inputs[..],
                             &[],
                             args_no_rcvr,
                             false,
                             tuple_arguments);
        ty::FnConverging(fcx.tcx().types.err)
    } else {
        match method_fn_ty.sty {
            ty::TyBareFn(_, ref fty) => {
                // HACK(eddyb) ignore self in the definition (see above).
                let expected_arg_tys = expected_types_for_fn_args(fcx,
                                                                  sp,
                                                                  expected,
                                                                  fty.sig.0.output,
                                                                  &fty.sig.0.inputs[1..]);
                check_argument_types(fcx,
                                     sp,
                                     &fty.sig.0.inputs[1..],
                                     &expected_arg_tys[..],
                                     args_no_rcvr,
                                     fty.sig.0.variadic,
                                     tuple_arguments);
                fty.sig.0.output
            }
            _ => {
                fcx.tcx().sess.span_bug(callee_expr.span,
                                        "method without bare fn type");
            }
        }
    }
}

/// Generic function that factors out common logic from function calls, method calls and overloaded
/// operators.
fn check_argument_types<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                  sp: Span,
                                  fn_inputs: &[Ty<'tcx>],
                                  expected_arg_tys: &[Ty<'tcx>],
                                  args: &'tcx [P<hir::Expr>],
                                  variadic: bool,
                                  tuple_arguments: TupleArgumentsFlag) {
    let tcx = fcx.ccx.tcx;

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
        fcx.register_wf_obligation(fn_input_ty, sp, traits::MiscObligation);
    }

    let mut expected_arg_tys = expected_arg_tys;
    let expected_arg_count = fn_inputs.len();
    let formal_tys = if tuple_arguments == TupleArguments {
        let tuple_type = structurally_resolved_type(fcx, sp, fn_inputs[0]);
        match tuple_type.sty {
            ty::TyTuple(ref arg_types) => {
                if arg_types.len() != args.len() {
                    span_err!(tcx.sess, sp, E0057,
                        "this function takes {} parameter{} but {} parameter{} supplied",
                        arg_types.len(),
                        if arg_types.len() == 1 {""} else {"s"},
                        args.len(),
                        if args.len() == 1 {" was"} else {"s were"});
                    expected_arg_tys = &[];
                    err_args(fcx.tcx(), args.len())
                } else {
                    expected_arg_tys = match expected_arg_tys.get(0) {
                        Some(&ty) => match ty.sty {
                            ty::TyTuple(ref tys) => &**tys,
                            _ => &[]
                        },
                        None => &[]
                    };
                    (*arg_types).clone()
                }
            }
            _ => {
                span_err!(tcx.sess, sp, E0059,
                    "cannot use call notation; the first type parameter \
                     for the function trait is neither a tuple nor unit");
                expected_arg_tys = &[];
                err_args(fcx.tcx(), args.len())
            }
        }
    } else if expected_arg_count == supplied_arg_count {
        fn_inputs.to_vec()
    } else if variadic {
        if supplied_arg_count >= expected_arg_count {
            fn_inputs.to_vec()
        } else {
            span_err!(tcx.sess, sp, E0060,
                "this function takes at least {} parameter{} \
                 but {} parameter{} supplied",
                expected_arg_count,
                if expected_arg_count == 1 {""} else {"s"},
                supplied_arg_count,
                if supplied_arg_count == 1 {" was"} else {"s were"});
            expected_arg_tys = &[];
            err_args(fcx.tcx(), supplied_arg_count)
        }
    } else {
        span_err!(tcx.sess, sp, E0061,
            "this function takes {} parameter{} but {} parameter{} supplied",
            expected_arg_count,
            if expected_arg_count == 1 {""} else {"s"},
            supplied_arg_count,
            if supplied_arg_count == 1 {" was"} else {"s were"});
        expected_arg_tys = &[];
        err_args(fcx.tcx(), supplied_arg_count)
    };

    debug!("check_argument_types: formal_tys={:?}",
           formal_tys.iter().map(|t| fcx.infcx().ty_to_string(*t)).collect::<Vec<String>>());

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
            fcx.select_new_obligations();
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
                fcx.ccx
                    .tcx
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
                    Expectation::rvalue_hint(fcx.tcx(), ty)
                });

                check_expr_with_unifier(fcx,
                                        &**arg,
                                        expected.unwrap_or(ExpectHasType(formal_ty)),
                                        NoPreference, || {
                    // 2. Coerce to the most detailed type that could be coerced
                    //    to, which is `expected_ty` if `rvalue_hint` returns an
                    //    `ExprHasType(expected_ty)`, or the `formal_ty` otherwise.
                    let coerce_ty = expected.and_then(|e| e.only_has_type(fcx));
                    demand::coerce(fcx, arg.span, coerce_ty.unwrap_or(formal_ty), &**arg);

                    // 3. Relate the expected type and the formal one,
                    //    if the expected type was used for the coercion.
                    coerce_ty.map(|ty| demand::suptype(fcx, arg.span, formal_ty, ty));
                });
            }

            if let Some(&arg_ty) = fcx.inh.tables.borrow().node_types.get(&arg.id) {
                any_diverges = any_diverges || fcx.infcx().type_var_diverges(arg_ty);
            }
        }
        if any_diverges && !warned {
            let parent = fcx.ccx.tcx.map.get_parent_node(args[0].id);
            fcx.ccx
                .tcx
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
            check_expr(fcx, &**arg);

            // There are a few types which get autopromoted when passed via varargs
            // in C but we just error out instead and require explicit casts.
            let arg_ty = structurally_resolved_type(fcx, arg.span,
                                                    fcx.expr_ty(&**arg));
            match arg_ty.sty {
                ty::TyFloat(ast::TyF32) => {
                    fcx.type_error_message(arg.span,
                                           |t| {
                        format!("can't pass an {} to variadic \
                                 function, cast to c_double", t)
                    }, arg_ty, None);
                }
                ty::TyInt(ast::TyI8) | ty::TyInt(ast::TyI16) | ty::TyBool => {
                    fcx.type_error_message(arg.span, |t| {
                        format!("can't pass {} to variadic \
                                 function, cast to c_int",
                                       t)
                    }, arg_ty, None);
                }
                ty::TyUint(ast::TyU8) | ty::TyUint(ast::TyU16) => {
                    fcx.type_error_message(arg.span, |t| {
                        format!("can't pass {} to variadic \
                                 function, cast to c_uint",
                                       t)
                    }, arg_ty, None);
                }
                _ => {}
            }
        }
    }
}

// FIXME(#17596) Ty<'tcx> is incorrectly invariant w.r.t 'tcx.
fn err_args<'tcx>(tcx: &ty::ctxt<'tcx>, len: usize) -> Vec<Ty<'tcx>> {
    (0..len).map(|_| tcx.types.err).collect()
}

fn write_call<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                        call_expr: &hir::Expr,
                        output: ty::FnOutput<'tcx>) {
    fcx.write_ty(call_expr.id, match output {
        ty::FnConverging(output_ty) => output_ty,
        ty::FnDiverging => fcx.infcx().next_diverging_ty_var()
    });
}

// AST fragment checking
fn check_lit<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                       lit: &ast::Lit,
                       expected: Expectation<'tcx>)
                       -> Ty<'tcx>
{
    let tcx = fcx.ccx.tcx;

    match lit.node {
        ast::LitStr(..) => tcx.mk_static_str(),
        ast::LitByteStr(ref v) => {
            tcx.mk_imm_ref(tcx.mk_region(ty::ReStatic),
                            tcx.mk_array(tcx.types.u8, v.len()))
        }
        ast::LitByte(_) => tcx.types.u8,
        ast::LitChar(_) => tcx.types.char,
        ast::LitInt(_, ast::SignedIntLit(t, _)) => tcx.mk_mach_int(t),
        ast::LitInt(_, ast::UnsignedIntLit(t)) => tcx.mk_mach_uint(t),
        ast::LitInt(_, ast::UnsuffixedIntLit(_)) => {
            let opt_ty = expected.to_option(fcx).and_then(|ty| {
                match ty.sty {
                    ty::TyInt(_) | ty::TyUint(_) => Some(ty),
                    ty::TyChar => Some(tcx.types.u8),
                    ty::TyRawPtr(..) => Some(tcx.types.usize),
                    ty::TyBareFn(..) => Some(tcx.types.usize),
                    _ => None
                }
            });
            opt_ty.unwrap_or_else(
                || tcx.mk_int_var(fcx.infcx().next_int_var_id()))
        }
        ast::LitFloat(_, t) => tcx.mk_mach_float(t),
        ast::LitFloatUnsuffixed(_) => {
            let opt_ty = expected.to_option(fcx).and_then(|ty| {
                match ty.sty {
                    ty::TyFloat(_) => Some(ty),
                    _ => None
                }
            });
            opt_ty.unwrap_or_else(
                || tcx.mk_float_var(fcx.infcx().next_float_var_id()))
        }
        ast::LitBool(_) => tcx.types.bool
    }
}

pub fn check_expr_has_type<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                     expr: &'tcx hir::Expr,
                                     expected: Ty<'tcx>) {
    check_expr_with_unifier(
        fcx, expr, ExpectHasType(expected), NoPreference,
        || demand::suptype(fcx, expr.span, expected, fcx.expr_ty(expr)));
}

fn check_expr_coercable_to_type<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                          expr: &'tcx hir::Expr,
                                          expected: Ty<'tcx>) {
    check_expr_with_unifier(
        fcx, expr, ExpectHasType(expected), NoPreference,
        || demand::coerce(fcx, expr.span, expected, expr));
}

fn check_expr_with_hint<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>, expr: &'tcx hir::Expr,
                                  expected: Ty<'tcx>) {
    check_expr_with_unifier(
        fcx, expr, ExpectHasType(expected), NoPreference,
        || ())
}

fn check_expr_with_expectation<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                         expr: &'tcx hir::Expr,
                                         expected: Expectation<'tcx>) {
    check_expr_with_unifier(
        fcx, expr, expected, NoPreference,
        || ())
}

fn check_expr_with_expectation_and_lvalue_pref<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                                         expr: &'tcx hir::Expr,
                                                         expected: Expectation<'tcx>,
                                                         lvalue_pref: LvaluePreference)
{
    check_expr_with_unifier(fcx, expr, expected, lvalue_pref, || ())
}

fn check_expr<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>, expr: &'tcx hir::Expr)  {
    check_expr_with_unifier(fcx, expr, NoExpectation, NoPreference, || ())
}

fn check_expr_with_lvalue_pref<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>, expr: &'tcx hir::Expr,
                                        lvalue_pref: LvaluePreference)  {
    check_expr_with_unifier(fcx, expr, NoExpectation, lvalue_pref, || ())
}

// determine the `self` type, using fresh variables for all variables
// declared on the impl declaration e.g., `impl<A,B> for Vec<(A,B)>`
// would return ($0, $1) where $0 and $1 are freshly instantiated type
// variables.
pub fn impl_self_ty<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                              span: Span, // (potential) receiver for this impl
                              did: DefId)
                              -> TypeAndSubsts<'tcx> {
    let tcx = fcx.tcx();

    let ity = tcx.lookup_item_type(did);
    let (tps, rps, raw_ty) =
        (ity.generics.types.get_slice(subst::TypeSpace),
         ity.generics.regions.get_slice(subst::TypeSpace),
         ity.ty);

    debug!("impl_self_ty: tps={:?} rps={:?} raw_ty={:?}", tps, rps, raw_ty);

    let rps = fcx.inh.infcx.region_vars_for_defs(span, rps);
    let mut substs = subst::Substs::new(
        VecPerParamSpace::empty(),
        VecPerParamSpace::new(rps, Vec::new(), Vec::new()));
    fcx.inh.infcx.type_vars_for_defs(span, ParamSpace::TypeSpace, &mut substs, tps);
    let substd_ty = fcx.instantiate_type_scheme(span, &substs, &raw_ty);

    TypeAndSubsts { substs: substs, ty: substd_ty }
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

/// Unifies the return type with the expected type early, for more coercions
/// and forward type information on the argument expressions.
fn expected_types_for_fn_args<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                        call_span: Span,
                                        expected_ret: Expectation<'tcx>,
                                        formal_ret: ty::FnOutput<'tcx>,
                                        formal_args: &[Ty<'tcx>])
                                        -> Vec<Ty<'tcx>> {
    let expected_args = expected_ret.only_has_type(fcx).and_then(|ret_ty| {
        if let ty::FnConverging(formal_ret_ty) = formal_ret {
            fcx.infcx().commit_regions_if_ok(|| {
                // Attempt to apply a subtyping relationship between the formal
                // return type (likely containing type variables if the function
                // is polymorphic) and the expected return type.
                // No argument expectations are produced if unification fails.
                let origin = TypeOrigin::Misc(call_span);
                let ures = fcx.infcx().sub_types(false, origin, formal_ret_ty, ret_ty);
                // FIXME(#15760) can't use try! here, FromError doesn't default
                // to identity so the resulting type is not constrained.
                if let Err(e) = ures {
                    return Err(e);
                }

                // Record all the argument types, with the substitutions
                // produced from the above subtyping unification.
                Ok(formal_args.iter().map(|ty| {
                    fcx.infcx().resolve_type_vars_if_possible(ty)
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
fn check_expr_with_unifier<'a, 'tcx, F>(fcx: &FnCtxt<'a, 'tcx>,
                                        expr: &'tcx hir::Expr,
                                        expected: Expectation<'tcx>,
                                        lvalue_pref: LvaluePreference,
                                        unifier: F) where
    F: FnOnce(),
{
    debug!(">> typechecking: expr={:?} expected={:?}",
           expr, expected);

    // Checks a method call.
    fn check_method_call<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                   expr: &'tcx hir::Expr,
                                   method_name: Spanned<ast::Name>,
                                   args: &'tcx [P<hir::Expr>],
                                   tps: &[P<hir::Ty>],
                                   expected: Expectation<'tcx>,
                                   lvalue_pref: LvaluePreference) {
        let rcvr = &*args[0];
        check_expr_with_lvalue_pref(fcx, &*rcvr, lvalue_pref);

        // no need to check for bot/err -- callee does that
        let expr_t = structurally_resolved_type(fcx,
                                                expr.span,
                                                fcx.expr_ty(&*rcvr));

        let tps = tps.iter().map(|ast_ty| fcx.to_ty(&**ast_ty)).collect::<Vec<_>>();
        let fn_ty = match method::lookup(fcx,
                                         method_name.span,
                                         method_name.node,
                                         expr_t,
                                         tps,
                                         expr,
                                         rcvr) {
            Ok(method) => {
                let method_ty = method.ty;
                let method_call = MethodCall::expr(expr.id);
                fcx.inh.tables.borrow_mut().method_map.insert(method_call, method);
                method_ty
            }
            Err(error) => {
                method::report_error(fcx, method_name.span, expr_t,
                                     method_name.node, Some(rcvr), error);
                fcx.write_error(expr.id);
                fcx.tcx().types.err
            }
        };

        // Call the generic checker.
        let ret_ty = check_method_argument_types(fcx,
                                                 method_name.span,
                                                 fn_ty,
                                                 expr,
                                                 &args[1..],
                                                 DontTupleArguments,
                                                 expected);

        write_call(fcx, expr, ret_ty);
    }

    // A generic function for checking the then and else in an if
    // or if-else.
    fn check_then_else<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                 cond_expr: &'tcx hir::Expr,
                                 then_blk: &'tcx hir::Block,
                                 opt_else_expr: Option<&'tcx hir::Expr>,
                                 id: ast::NodeId,
                                 sp: Span,
                                 expected: Expectation<'tcx>) {
        check_expr_has_type(fcx, cond_expr, fcx.tcx().types.bool);

        let expected = expected.adjust_for_branches(fcx);
        check_block_with_expected(fcx, then_blk, expected);
        let then_ty = fcx.node_ty(then_blk.id);

        let branches_ty = match opt_else_expr {
            Some(ref else_expr) => {
                check_expr_with_expectation(fcx, &**else_expr, expected);
                let else_ty = fcx.expr_ty(&**else_expr);
                infer::common_supertype(fcx.infcx(),
                                        TypeOrigin::IfExpression(sp),
                                        true,
                                        then_ty,
                                        else_ty)
            }
            None => {
                infer::common_supertype(fcx.infcx(),
                                        TypeOrigin::IfExpressionWithNoElse(sp),
                                        false,
                                        then_ty,
                                        fcx.tcx().mk_nil())
            }
        };

        let cond_ty = fcx.expr_ty(cond_expr);
        let if_ty = if cond_ty.references_error() {
            fcx.tcx().types.err
        } else {
            branches_ty
        };

        fcx.write_ty(id, if_ty);
    }

    // Check field access expressions
    fn check_field<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                            expr: &'tcx hir::Expr,
                            lvalue_pref: LvaluePreference,
                            base: &'tcx hir::Expr,
                            field: &Spanned<ast::Name>) {
        let tcx = fcx.ccx.tcx;
        check_expr_with_lvalue_pref(fcx, base, lvalue_pref);
        let expr_t = structurally_resolved_type(fcx, expr.span,
                                                fcx.expr_ty(base));
        // FIXME(eddyb) #12808 Integrate privacy into this auto-deref loop.
        let (_, autoderefs, field_ty) = autoderef(fcx,
                                                  expr.span,
                                                  expr_t,
                                                  Some(base),
                                                  UnresolvedTypeAction::Error,
                                                  lvalue_pref,
                                                  |base_t, _| {
                match base_t.sty {
                    ty::TyStruct(base_def, substs) => {
                        debug!("struct named {:?}",  base_t);
                        base_def.struct_variant()
                                .find_field_named(field.node)
                                .map(|f| fcx.field_ty(expr.span, f, substs))
                    }
                    _ => None
                }
            });
        match field_ty {
            Some(field_ty) => {
                fcx.write_ty(expr.id, field_ty);
                fcx.write_autoderef_adjustment(base.id, autoderefs);
                return;
            }
            None => {}
        }

        if method::exists(fcx, field.span, field.node, expr_t, expr.id) {
            fcx.type_error_message(
                field.span,
                |actual| {
                    format!("attempted to take value of method `{}` on type \
                            `{}`", field.node, actual)
                },
                expr_t, None);

            tcx.sess.fileline_help(field.span,
                               "maybe a `()` to call it is missing? \
                               If not, try an anonymous function");
        } else {
            fcx.type_error_message(
                expr.span,
                |actual| {
                    format!("attempted access of field `{}` on \
                            type `{}`, but no field with that \
                            name was found",
                            field.node,
                            actual)
                },
                expr_t, None);
            if let ty::TyStruct(def, _) = expr_t.sty {
                suggest_field_names(def.struct_variant(), field, tcx, vec![]);
            }
        }

        fcx.write_error(expr.id);
    }

    // displays hints about the closest matches in field names
    fn suggest_field_names<'tcx>(variant: ty::VariantDef<'tcx>,
                                 field: &Spanned<ast::Name>,
                                 tcx: &ty::ctxt<'tcx>,
                                 skip : Vec<InternedString>) {
        let name = field.node.as_str();
        // only find fits with at least one matching letter
        let mut best_dist = name.len();
        let mut best = None;
        for elem in &variant.fields {
            let n = elem.name.as_str();
            // ignore already set fields
            if skip.iter().any(|x| *x == n) {
                continue;
            }
            // ignore private fields from non-local crates
            if variant.did.krate != LOCAL_CRATE && elem.vis != Visibility::Public {
                continue;
            }
            let dist = lev_distance(&n, &name);
            if dist < best_dist {
                best = Some(n);
                best_dist = dist;
            }
        }
        if let Some(n) = best {
            tcx.sess.span_help(field.span,
                &format!("did you mean `{}`?", n));
        }
    }

    // Check tuple index expressions
    fn check_tup_field<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                                expr: &'tcx hir::Expr,
                                lvalue_pref: LvaluePreference,
                                base: &'tcx hir::Expr,
                                idx: codemap::Spanned<usize>) {
        check_expr_with_lvalue_pref(fcx, base, lvalue_pref);
        let expr_t = structurally_resolved_type(fcx, expr.span,
                                                fcx.expr_ty(base));
        let mut tuple_like = false;
        // FIXME(eddyb) #12808 Integrate privacy into this auto-deref loop.
        let (_, autoderefs, field_ty) = autoderef(fcx,
                                                  expr.span,
                                                  expr_t,
                                                  Some(base),
                                                  UnresolvedTypeAction::Error,
                                                  lvalue_pref,
                                                  |base_t, _| {
                match base_t.sty {
                    ty::TyStruct(base_def, substs) => {
                        tuple_like = base_def.struct_variant().is_tuple_struct();
                        if tuple_like {
                            debug!("tuple struct named {:?}",  base_t);
                            base_def.struct_variant()
                                    .fields
                                    .get(idx.node)
                                    .map(|f| fcx.field_ty(expr.span, f, substs))
                        } else {
                            None
                        }
                    }
                    ty::TyTuple(ref v) => {
                        tuple_like = true;
                        if idx.node < v.len() { Some(v[idx.node]) } else { None }
                    }
                    _ => None
                }
            });
        match field_ty {
            Some(field_ty) => {
                fcx.write_ty(expr.id, field_ty);
                fcx.write_autoderef_adjustment(base.id, autoderefs);
                return;
            }
            None => {}
        }
        fcx.type_error_message(
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

        fcx.write_error(expr.id);
    }

    fn report_unknown_field<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                      ty: Ty<'tcx>,
                                      variant: ty::VariantDef<'tcx>,
                                      field: &hir::Field,
                                      skip_fields: &[hir::Field]) {
        fcx.type_error_message(
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
        suggest_field_names(variant, &field.name, fcx.tcx(), skip_fields.collect());
    }

    fn check_expr_struct_fields<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                          adt_ty: Ty<'tcx>,
                                          span: Span,
                                          variant: ty::VariantDef<'tcx>,
                                          ast_fields: &'tcx [hir::Field],
                                          check_completeness: bool) {
        let tcx = fcx.ccx.tcx;
        let substs = match adt_ty.sty {
            ty::TyStruct(_, substs) | ty::TyEnum(_, substs) => substs,
            _ => tcx.sess.span_bug(span, "non-ADT passed to check_expr_struct_fields")
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
                expected_field_type = fcx.field_ty(field.span, v_field, substs);
            } else {
                error_happened = true;
                expected_field_type = tcx.types.err;
                if let Some(_) = variant.find_field_named(field.name.node) {
                    span_err!(fcx.tcx().sess, field.name.span, E0062,
                        "field `{}` specified more than once",
                        field.name.node);
                } else {
                    report_unknown_field(fcx, adt_ty, variant, field, ast_fields);
                }
            }

            // Make sure to give a type to the field even if there's
            // an error, so we can continue typechecking
            check_expr_coercable_to_type(fcx, &*field.expr, expected_field_type);
        }

            // Make sure the programmer specified all the fields.
        if check_completeness &&
            !error_happened &&
            !remaining_fields.is_empty()
        {
            span_err!(tcx.sess, span, E0063,
                      "missing field{}: {}",
                      if remaining_fields.len() == 1 {""} else {"s"},
                      remaining_fields.keys()
                                      .map(|n| format!("`{}`", n))
                                      .collect::<Vec<_>>()
                                      .join(", "));
        }

    }

    fn check_struct_fields_on_error<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                                             id: ast::NodeId,
                                             fields: &'tcx [hir::Field],
                                             base_expr: &'tcx Option<P<hir::Expr>>) {
        // Make sure to still write the types
        // otherwise we might ICE
        fcx.write_error(id);
        for field in fields {
            check_expr(fcx, &*field.expr);
        }
        match *base_expr {
            Some(ref base) => check_expr(fcx, &**base),
            None => {}
        }
    }

    fn check_expr_struct<'a, 'tcx>(fcx: &FnCtxt<'a,'tcx>,
                                   expr: &hir::Expr,
                                   path: &hir::Path,
                                   fields: &'tcx [hir::Field],
                                   base_expr: &'tcx Option<P<hir::Expr>>)
    {
        let tcx = fcx.tcx();

        // Find the relevant variant
        let def = lookup_full_def(tcx, path.span, expr.id);
        let (adt, variant) = match fcx.def_struct_variant(def, path.span) {
            Some((adt, variant)) => (adt, variant),
            None => {
                span_err!(fcx.tcx().sess, path.span, E0071,
                          "`{}` does not name a structure",
                          pprust::path_to_string(path));
                check_struct_fields_on_error(fcx, expr.id, fields, base_expr);
                return;
            }
        };

        let expr_ty = fcx.instantiate_type(def.def_id(), path);
        fcx.write_ty(expr.id, expr_ty);

        check_expr_struct_fields(fcx, expr_ty, expr.span, variant, fields,
                                 base_expr.is_none());

        if let &Some(ref base_expr) = base_expr {
            check_expr_has_type(fcx, base_expr, expr_ty);
            if adt.adt_kind() == ty::AdtKind::Enum {
                span_err!(tcx.sess, base_expr.span, E0436,
                          "functional record update syntax requires a struct");
            }
        }
    }

    type ExprCheckerWithTy = fn(&FnCtxt, &hir::Expr, Ty);

    let tcx = fcx.ccx.tcx;
    let id = expr.id;
    match expr.node {
      hir::ExprBox(ref subexpr) => {
        let expected_inner = expected.to_option(fcx).map_or(NoExpectation, |ty| {
            match ty.sty {
                ty::TyBox(ty) => Expectation::rvalue_hint(tcx, ty),
                _ => NoExpectation
            }
        });
        check_expr_with_expectation(fcx, subexpr, expected_inner);
        let referent_ty = fcx.expr_ty(&**subexpr);
        fcx.write_ty(id, tcx.mk_box(referent_ty));
      }

      hir::ExprLit(ref lit) => {
        let typ = check_lit(fcx, &**lit, expected);
        fcx.write_ty(id, typ);
      }
      hir::ExprBinary(op, ref lhs, ref rhs) => {
        op::check_binop(fcx, expr, op, lhs, rhs);
      }
      hir::ExprAssignOp(op, ref lhs, ref rhs) => {
        op::check_binop_assign(fcx, expr, op, lhs, rhs);
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
        check_expr_with_expectation_and_lvalue_pref(
            fcx, &**oprnd, expected_inner, lvalue_pref);
        let mut oprnd_t = fcx.expr_ty(&**oprnd);

        if !oprnd_t.references_error() {
            match unop {
                hir::UnDeref => {
                    oprnd_t = structurally_resolved_type(fcx, expr.span, oprnd_t);
                    oprnd_t = match oprnd_t.builtin_deref(true, NoPreference) {
                        Some(mt) => mt.ty,
                        None => match try_overloaded_deref(fcx, expr.span,
                                                           Some(MethodCall::expr(expr.id)),
                                                           Some(&**oprnd), oprnd_t, lvalue_pref) {
                            Some(mt) => mt.ty,
                            None => {
                                fcx.type_error_message(expr.span, |actual| {
                                    format!("type `{}` cannot be \
                                            dereferenced", actual)
                                }, oprnd_t, None);
                                tcx.types.err
                            }
                        }
                    };
                }
                hir::UnNot => {
                    oprnd_t = structurally_resolved_type(fcx, oprnd.span,
                                                         oprnd_t);
                    if !(oprnd_t.is_integral() || oprnd_t.sty == ty::TyBool) {
                        oprnd_t = op::check_user_unop(fcx, "!", "not",
                                                      tcx.lang_items.not_trait(),
                                                      expr, &**oprnd, oprnd_t, unop);
                    }
                }
                hir::UnNeg => {
                    oprnd_t = structurally_resolved_type(fcx, oprnd.span,
                                                         oprnd_t);
                    if !(oprnd_t.is_integral() || oprnd_t.is_fp()) {
                        oprnd_t = op::check_user_unop(fcx, "-", "neg",
                                                      tcx.lang_items.neg_trait(),
                                                      expr, &**oprnd, oprnd_t, unop);
                    }
                }
            }
        }
        fcx.write_ty(id, oprnd_t);
      }
      hir::ExprAddrOf(mutbl, ref oprnd) => {
        let hint = expected.only_has_type(fcx).map_or(NoExpectation, |ty| {
            match ty.sty {
                ty::TyRef(_, ref mt) | ty::TyRawPtr(ref mt) => {
                    if fcx.tcx().expr_is_lval(&**oprnd) {
                        // Lvalues may legitimately have unsized types.
                        // For example, dereferences of a fat pointer and
                        // the last field of a struct can be unsized.
                        ExpectHasType(mt.ty)
                    } else {
                        Expectation::rvalue_hint(tcx, mt.ty)
                    }
                }
                _ => NoExpectation
            }
        });
        let lvalue_pref = LvaluePreference::from_mutbl(mutbl);
        check_expr_with_expectation_and_lvalue_pref(fcx,
                                                    &**oprnd,
                                                    hint,
                                                    lvalue_pref);

        let tm = ty::TypeAndMut { ty: fcx.expr_ty(&**oprnd), mutbl: mutbl };
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
            let region = fcx.infcx().next_region_var(infer::AddrOfRegion(expr.span));
            tcx.mk_ref(tcx.mk_region(region), tm)
        };
        fcx.write_ty(id, oprnd_t);
      }
      hir::ExprPath(ref maybe_qself, ref path) => {
          let opt_self_ty = maybe_qself.as_ref().map(|qself| {
              fcx.to_ty(&qself.ty)
          });

          let path_res = if let Some(&d) = tcx.def_map.borrow().get(&id) {
              d
          } else if let Some(hir::QSelf { position: 0, .. }) = *maybe_qself {
                // Create some fake resolution that can't possibly be a type.
                def::PathResolution {
                    base_def: def::DefMod(tcx.map.local_def_id(ast::CRATE_NODE_ID)),
                    last_private: LastMod(AllPublic),
                    depth: path.segments.len()
                }
            } else {
              tcx.sess.span_bug(expr.span,
                                &format!("unbound path {:?}", expr))
          };

          if let Some((opt_ty, segments, def)) =
                  resolve_ty_and_def_ufcs(fcx, path_res, opt_self_ty, path,
                                          expr.span, expr.id) {
              let (scheme, predicates) = type_scheme_and_predicates_for_def(fcx,
                                                                            expr.span,
                                                                            def);
              instantiate_path(fcx,
                               segments,
                               scheme,
                               &predicates,
                               opt_ty,
                               def,
                               expr.span,
                               id);
          }

          // We always require that the type provided as the value for
          // a type parameter outlives the moment of instantiation.
          fcx.opt_node_ty_substs(expr.id, |item_substs| {
              fcx.add_wf_bounds(&item_substs.substs, expr);
          });
      }
      hir::ExprInlineAsm(ref ia) => {
          for &(_, ref input) in &ia.inputs {
              check_expr(fcx, &**input);
          }
          for &(_, ref out, _) in &ia.outputs {
              check_expr(fcx, &**out);
          }
          fcx.write_nil(id);
      }
      hir::ExprBreak(_) => { fcx.write_ty(id, fcx.infcx().next_diverging_ty_var()); }
      hir::ExprAgain(_) => { fcx.write_ty(id, fcx.infcx().next_diverging_ty_var()); }
      hir::ExprRet(ref expr_opt) => {
        match fcx.ret_ty {
            ty::FnConverging(result_type) => {
                match *expr_opt {
                    None =>
                        if let Err(_) = fcx.mk_eqty(false, TypeOrigin::Misc(expr.span),
                                                    result_type, fcx.tcx().mk_nil()) {
                            span_err!(tcx.sess, expr.span, E0069,
                                "`return;` in a function whose return type is \
                                 not `()`");
                        },
                    Some(ref e) => {
                        check_expr_coercable_to_type(fcx, &**e, result_type);
                    }
                }
            }
            ty::FnDiverging => {
                if let Some(ref e) = *expr_opt {
                    check_expr(fcx, &**e);
                }
                span_err!(tcx.sess, expr.span, E0166,
                    "`return` in a function declared as diverging");
            }
        }
        fcx.write_ty(id, fcx.infcx().next_diverging_ty_var());
      }
      hir::ExprAssign(ref lhs, ref rhs) => {
        check_expr_with_lvalue_pref(fcx, &**lhs, PreferMutLvalue);

        let tcx = fcx.tcx();
        if !tcx.expr_is_lval(&**lhs) {
            span_err!(tcx.sess, expr.span, E0070,
                "invalid left-hand side expression");
        }

        let lhs_ty = fcx.expr_ty(&**lhs);
        check_expr_coercable_to_type(fcx, &**rhs, lhs_ty);
        let rhs_ty = fcx.expr_ty(&**rhs);

        fcx.require_expr_have_sized_type(&**lhs, traits::AssignmentLhsSized);

        if lhs_ty.references_error() || rhs_ty.references_error() {
            fcx.write_error(id);
        } else {
            fcx.write_nil(id);
        }
      }
      hir::ExprIf(ref cond, ref then_blk, ref opt_else_expr) => {
        check_then_else(fcx, &**cond, &**then_blk, opt_else_expr.as_ref().map(|e| &**e),
                        id, expr.span, expected);
      }
      hir::ExprWhile(ref cond, ref body, _) => {
        check_expr_has_type(fcx, &**cond, tcx.types.bool);
        check_block_no_value(fcx, &**body);
        let cond_ty = fcx.expr_ty(&**cond);
        let body_ty = fcx.node_ty(body.id);
        if cond_ty.references_error() || body_ty.references_error() {
            fcx.write_error(id);
        }
        else {
            fcx.write_nil(id);
        }
      }
      hir::ExprLoop(ref body, _) => {
        check_block_no_value(fcx, &**body);
        if !may_break(tcx, expr.id, &**body) {
            fcx.write_ty(id, fcx.infcx().next_diverging_ty_var());
        } else {
            fcx.write_nil(id);
        }
      }
      hir::ExprMatch(ref discrim, ref arms, match_src) => {
        _match::check_match(fcx, expr, &**discrim, arms, expected, match_src);
      }
      hir::ExprClosure(capture, ref decl, ref body) => {
          closure::check_expr_closure(fcx, expr, capture, &**decl, &**body, expected);
      }
      hir::ExprBlock(ref b) => {
        check_block_with_expected(fcx, &**b, expected);
        fcx.write_ty(id, fcx.node_ty(b.id));
      }
      hir::ExprCall(ref callee, ref args) => {
          callee::check_call(fcx, expr, &**callee, &args[..], expected);

          // we must check that return type of called functions is WF:
          let ret_ty = fcx.expr_ty(expr);
          fcx.register_wf_obligation(ret_ty, expr.span, traits::MiscObligation);
      }
      hir::ExprMethodCall(name, ref tps, ref args) => {
          check_method_call(fcx, expr, name, &args[..], &tps[..], expected, lvalue_pref);
          let arg_tys = args.iter().map(|a| fcx.expr_ty(&**a));
          let args_err = arg_tys.fold(false, |rest_err, a| rest_err || a.references_error());
          if args_err {
              fcx.write_error(id);
          }
      }
      hir::ExprCast(ref e, ref t) => {
        if let hir::TyFixedLengthVec(_, ref count_expr) = t.node {
            check_expr_with_hint(fcx, &**count_expr, tcx.types.usize);
        }

        // Find the type of `e`. Supply hints based on the type we are casting to,
        // if appropriate.
        let t_cast = fcx.to_ty(t);
        let t_cast = structurally_resolved_type(fcx, expr.span, t_cast);
        check_expr_with_expectation(fcx, e, ExpectCastableToType(t_cast));
        let t_expr = fcx.expr_ty(e);

        // Eagerly check for some obvious errors.
        if t_expr.references_error() {
            fcx.write_error(id);
        } else if !fcx.type_is_known_to_be_sized(t_cast, expr.span) {
            report_cast_to_unsized_type(fcx, expr.span, t.span, e.span, t_cast, t_expr, id);
        } else {
            // Write a type for the whole expression, assuming everything is going
            // to work out Ok.
            fcx.write_ty(id, t_cast);

            // Defer other checks until we're done type checking.
            let mut deferred_cast_checks = fcx.inh.deferred_cast_checks.borrow_mut();
            let cast_check = cast::CastCheck::new((**e).clone(), t_expr, t_cast, expr.span);
            deferred_cast_checks.push(cast_check);
        }
      }
      hir::ExprVec(ref args) => {
        let uty = expected.to_option(fcx).and_then(|uty| {
            match uty.sty {
                ty::TyArray(ty, _) | ty::TySlice(ty) => Some(ty),
                _ => None
            }
        });

        let typ = match uty {
            Some(uty) => {
                for e in args {
                    check_expr_coercable_to_type(fcx, &**e, uty);
                }
                uty
            }
            None => {
                let t: Ty = fcx.infcx().next_ty_var();
                for e in args {
                    check_expr_has_type(fcx, &**e, t);
                }
                t
            }
        };
        let typ = tcx.mk_array(typ, args.len());
        fcx.write_ty(id, typ);
      }
      hir::ExprRepeat(ref element, ref count_expr) => {
        check_expr_has_type(fcx, &**count_expr, tcx.types.usize);
        let count = fcx.tcx().eval_repeat_count(&**count_expr);

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
                check_expr_coercable_to_type(fcx, &**element, uty);
                (uty, uty)
            }
            None => {
                let t: Ty = fcx.infcx().next_ty_var();
                check_expr_has_type(fcx, &**element, t);
                (fcx.expr_ty(&**element), t)
            }
        };

        if count > 1 {
            // For [foo, ..n] where n > 1, `foo` must have
            // Copy type:
            fcx.require_type_meets(
                t,
                expr.span,
                traits::RepeatVec,
                ty::BoundCopy);
        }

        if element_ty.references_error() {
            fcx.write_error(id);
        } else {
            let t = tcx.mk_array(t, count);
            fcx.write_ty(id, t);
        }
      }
      hir::ExprTup(ref elts) => {
        let flds = expected.only_has_type(fcx).and_then(|ty| {
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
                    check_expr_coercable_to_type(fcx, &**e, ety);
                    ety
                }
                _ => {
                    check_expr_with_expectation(fcx, &**e, NoExpectation);
                    fcx.expr_ty(&**e)
                }
            };
            err_field = err_field || t.references_error();
            t
        }).collect();
        if err_field {
            fcx.write_error(id);
        } else {
            let typ = tcx.mk_tup(elt_ts);
            fcx.write_ty(id, typ);
        }
      }
      hir::ExprStruct(ref path, ref fields, ref base_expr) => {
        check_expr_struct(fcx, expr, path, fields, base_expr);

        fcx.require_expr_have_sized_type(expr, traits::StructInitializerSized);
      }
      hir::ExprField(ref base, ref field) => {
        check_field(fcx, expr, lvalue_pref, &**base, field);
      }
      hir::ExprTupField(ref base, idx) => {
        check_tup_field(fcx, expr, lvalue_pref, &**base, idx);
      }
      hir::ExprIndex(ref base, ref idx) => {
          check_expr_with_lvalue_pref(fcx, &**base, lvalue_pref);
          check_expr(fcx, &**idx);

          let base_t = fcx.expr_ty(&**base);
          let idx_t = fcx.expr_ty(&**idx);

          if base_t.references_error() {
              fcx.write_ty(id, base_t);
          } else if idx_t.references_error() {
              fcx.write_ty(id, idx_t);
          } else {
              let base_t = structurally_resolved_type(fcx, expr.span, base_t);
              match lookup_indexing(fcx, expr, base, base_t, idx_t, lvalue_pref) {
                  Some((index_ty, element_ty)) => {
                      let idx_expr_ty = fcx.expr_ty(idx);
                      demand::eqtype(fcx, expr.span, index_ty, idx_expr_ty);
                      fcx.write_ty(id, element_ty);
                  }
                  None => {
                      check_expr_has_type(fcx, &**idx, fcx.tcx().types.err);
                      fcx.type_error_message(
                          expr.span,
                          |actual| {
                              format!("cannot index a value of type `{}`",
                                      actual)
                          },
                          base_t,
                          None);
                      fcx.write_ty(id, fcx.tcx().types.err);
                  }
              }
          }
       }
       hir::ExprRange(ref start, ref end) => {
          let t_start = start.as_ref().map(|e| {
            check_expr(fcx, &**e);
            fcx.expr_ty(&**e)
          });
          let t_end = end.as_ref().map(|e| {
            check_expr(fcx, &**e);
            fcx.expr_ty(&**e)
          });

          let idx_type = match (t_start, t_end) {
              (Some(ty), None) | (None, Some(ty)) => {
                  Some(ty)
              }
              (Some(t_start), Some(t_end)) if (t_start.references_error() ||
                                               t_end.references_error()) => {
                  Some(fcx.tcx().types.err)
              }
              (Some(t_start), Some(t_end)) => {
                  Some(infer::common_supertype(fcx.infcx(),
                                               TypeOrigin::RangeExpression(expr.span),
                                               true,
                                               t_start,
                                               t_end))
              }
              _ => None
          };

          // Note that we don't check the type of start/end satisfy any
          // bounds because right now the range structs do not have any. If we add
          // some bounds, then we'll need to check `t_start` against them here.

          let range_type = match idx_type {
            Some(idx_type) if idx_type.references_error() => {
                fcx.tcx().types.err
            }
            Some(idx_type) => {
                // Find the did from the appropriate lang item.
                let did = match (start, end) {
                    (&Some(_), &Some(_)) => tcx.lang_items.range_struct(),
                    (&Some(_), &None) => tcx.lang_items.range_from_struct(),
                    (&None, &Some(_)) => tcx.lang_items.range_to_struct(),
                    (&None, &None) => {
                        tcx.sess.span_bug(expr.span, "full range should be dealt with above")
                    }
                };

                if let Some(did) = did {
                    let def = tcx.lookup_adt_def(did);
                    let predicates = tcx.lookup_predicates(did);
                    let substs = Substs::new_type(vec![idx_type], vec![]);
                    let bounds = fcx.instantiate_bounds(expr.span, &substs, &predicates);
                    fcx.add_obligations_for_parameters(
                        traits::ObligationCause::new(expr.span,
                                                     fcx.body_id,
                                                     traits::ItemObligation(did)),
                        &bounds);

                    tcx.mk_struct(def, tcx.mk_substs(substs))
                } else {
                    span_err!(tcx.sess, expr.span, E0236, "no lang item for range syntax");
                    fcx.tcx().types.err
                }
            }
            None => {
                // Neither start nor end => RangeFull
                if let Some(did) = tcx.lang_items.range_full_struct() {
                    tcx.mk_struct(
                        tcx.lookup_adt_def(did),
                        tcx.mk_substs(Substs::empty())
                    )
                } else {
                    span_err!(tcx.sess, expr.span, E0237, "no lang item for range syntax");
                    fcx.tcx().types.err
                }
            }
          };

          fcx.write_ty(id, range_type);
       }

    }

    debug!("type of expr({}) {} is...", expr.id,
           pprust::expr_to_string(expr));
    debug!("... {:?}, expected is {:?}",
           fcx.expr_ty(expr),
           expected);

    unifier();
}

pub fn resolve_ty_and_def_ufcs<'a, 'b, 'tcx>(fcx: &FnCtxt<'b, 'tcx>,
                                             path_res: def::PathResolution,
                                             opt_self_ty: Option<Ty<'tcx>>,
                                             path: &'a hir::Path,
                                             span: Span,
                                             node_id: ast::NodeId)
                                             -> Option<(Option<Ty<'tcx>>,
                                                        &'a [hir::PathSegment],
                                                        def::Def)>
{

    // Associated constants can't depend on generic types.
    fn have_disallowed_generic_consts<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                                def: def::Def,
                                                ty: Ty<'tcx>,
                                                span: Span,
                                                node_id: ast::NodeId) -> bool {
        match def {
            def::DefAssociatedConst(..) => {
                if ty.has_param_types() || ty.has_self_ty() {
                    span_err!(fcx.sess(), span, E0329,
                              "Associated consts cannot depend \
                               on type parameters or Self.");
                    fcx.write_error(node_id);
                    return true;
                }
            }
            _ => {}
        }
        false
    }

    // If fully resolved already, we don't have to do anything.
    if path_res.depth == 0 {
        if let Some(ty) = opt_self_ty {
            if have_disallowed_generic_consts(fcx, path_res.full_def(), ty,
                                              span, node_id) {
                return None;
            }
        }
        Some((opt_self_ty, &path.segments, path_res.base_def))
    } else {
        let mut def = path_res.base_def;
        let ty_segments = path.segments.split_last().unwrap().1;
        let base_ty_end = path.segments.len() - path_res.depth;
        let ty = astconv::finish_resolving_def_to_ty(fcx, fcx, span,
                                                     PathParamMode::Optional,
                                                     &mut def,
                                                     opt_self_ty,
                                                     &ty_segments[..base_ty_end],
                                                     &ty_segments[base_ty_end..]);
        let item_segment = path.segments.last().unwrap();
        let item_name = item_segment.identifier.name;
        match method::resolve_ufcs(fcx, span, item_name, ty, node_id) {
            Ok((def, lp)) => {
                if have_disallowed_generic_consts(fcx, def, ty, span, node_id) {
                    return None;
                }
                // Write back the new resolution.
                fcx.ccx.tcx.def_map.borrow_mut()
                       .insert(node_id, def::PathResolution {
                   base_def: def,
                   last_private: path_res.last_private.or(lp),
                   depth: 0
                });
                Some((Some(ty), slice::ref_slice(item_segment), def))
            }
            Err(error) => {
                method::report_error(fcx, span, ty,
                                     item_name, None, error);
                fcx.write_error(node_id);
                None
            }
        }
    }
}

impl<'tcx> Expectation<'tcx> {
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
    fn rvalue_hint(tcx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> Expectation<'tcx> {
        match tcx.struct_tail(ty).sty {
            ty::TySlice(_) | ty::TyTrait(..) => {
                ExpectRvalueLikeUnsized(ty)
            }
            _ => ExpectHasType(ty)
        }
    }

    // Resolves `expected` by a single level if it is a variable. If
    // there is no expected type or resolution is not possible (e.g.,
    // no constraints yet present), just returns `None`.
    fn resolve<'a>(self, fcx: &FnCtxt<'a, 'tcx>) -> Expectation<'tcx> {
        match self {
            NoExpectation => {
                NoExpectation
            }
            ExpectCastableToType(t) => {
                ExpectCastableToType(
                    fcx.infcx().resolve_type_vars_if_possible(&t))
            }
            ExpectHasType(t) => {
                ExpectHasType(
                    fcx.infcx().resolve_type_vars_if_possible(&t))
            }
            ExpectRvalueLikeUnsized(t) => {
                ExpectRvalueLikeUnsized(
                    fcx.infcx().resolve_type_vars_if_possible(&t))
            }
        }
    }

    fn to_option<'a>(self, fcx: &FnCtxt<'a, 'tcx>) -> Option<Ty<'tcx>> {
        match self.resolve(fcx) {
            NoExpectation => None,
            ExpectCastableToType(ty) |
            ExpectHasType(ty) |
            ExpectRvalueLikeUnsized(ty) => Some(ty),
        }
    }

    fn only_has_type<'a>(self, fcx: &FnCtxt<'a, 'tcx>) -> Option<Ty<'tcx>> {
        match self.resolve(fcx) {
            ExpectHasType(ty) => Some(ty),
            _ => None
        }
    }
}

pub fn check_decl_initializer<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                                       local: &'tcx hir::Local,
                                       init: &'tcx hir::Expr)
{
    let ref_bindings = fcx.tcx().pat_contains_ref_binding(&local.pat);

    let local_ty = fcx.local_ty(init.span, local.id);
    if let Some(m) = ref_bindings {
        // Somewhat subtle: if we have a `ref` binding in the pattern,
        // we want to avoid introducing coercions for the RHS. This is
        // both because it helps preserve sanity and, in the case of
        // ref mut, for soundness (issue #23116). In particular, in
        // the latter case, we need to be clear that the type of the
        // referent for the reference that results is *equal to* the
        // type of the lvalue it is referencing, and not some
        // supertype thereof.
        check_expr_with_lvalue_pref(fcx, init, LvaluePreference::from_mutbl(m));
        let init_ty = fcx.expr_ty(init);
        demand::eqtype(fcx, init.span, init_ty, local_ty);
    } else {
        check_expr_coercable_to_type(fcx, init, local_ty)
    };
}

pub fn check_decl_local<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>, local: &'tcx hir::Local)  {
    let tcx = fcx.ccx.tcx;

    let t = fcx.local_ty(local.span, local.id);
    fcx.write_ty(local.id, t);

    if let Some(ref init) = local.init {
        check_decl_initializer(fcx, local, &**init);
        let init_ty = fcx.expr_ty(&**init);
        if init_ty.references_error() {
            fcx.write_ty(local.id, init_ty);
        }
    }

    let pcx = pat_ctxt {
        fcx: fcx,
        map: pat_id_map(&tcx.def_map, &*local.pat),
    };
    _match::check_pat(&pcx, &*local.pat, t);
    let pat_ty = fcx.node_ty(local.pat.id);
    if pat_ty.references_error() {
        fcx.write_ty(local.id, pat_ty);
    }
}

pub fn check_stmt<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>, stmt: &'tcx hir::Stmt)  {
    let node_id;
    let mut saw_bot = false;
    let mut saw_err = false;
    match stmt.node {
      hir::StmtDecl(ref decl, id) => {
        node_id = id;
        match decl.node {
          hir::DeclLocal(ref l) => {
              check_decl_local(fcx, &**l);
              let l_t = fcx.node_ty(l.id);
              saw_bot = saw_bot || fcx.infcx().type_var_diverges(l_t);
              saw_err = saw_err || l_t.references_error();
          }
          hir::DeclItem(_) => {/* ignore for now */ }
        }
      }
      hir::StmtExpr(ref expr, id) => {
        node_id = id;
        // Check with expected type of ()
        check_expr_has_type(fcx, &**expr, fcx.tcx().mk_nil());
        let expr_ty = fcx.expr_ty(&**expr);
        saw_bot = saw_bot || fcx.infcx().type_var_diverges(expr_ty);
        saw_err = saw_err || expr_ty.references_error();
      }
      hir::StmtSemi(ref expr, id) => {
        node_id = id;
        check_expr(fcx, &**expr);
        let expr_ty = fcx.expr_ty(&**expr);
        saw_bot |= fcx.infcx().type_var_diverges(expr_ty);
        saw_err |= expr_ty.references_error();
      }
    }
    if saw_bot {
        fcx.write_ty(node_id, fcx.infcx().next_diverging_ty_var());
    }
    else if saw_err {
        fcx.write_error(node_id);
    }
    else {
        fcx.write_nil(node_id)
    }
}

pub fn check_block_no_value<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>, blk: &'tcx hir::Block)  {
    check_block_with_expected(fcx, blk, ExpectHasType(fcx.tcx().mk_nil()));
    let blkty = fcx.node_ty(blk.id);
    if blkty.references_error() {
        fcx.write_error(blk.id);
    } else {
        let nilty = fcx.tcx().mk_nil();
        demand::suptype(fcx, blk.span, nilty, blkty);
    }
}

fn check_block_with_expected<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                       blk: &'tcx hir::Block,
                                       expected: Expectation<'tcx>) {
    let prev = {
        let mut fcx_ps = fcx.ps.borrow_mut();
        let unsafety_state = fcx_ps.recurse(blk);
        replace(&mut *fcx_ps, unsafety_state)
    };

    let mut warned = false;
    let mut any_diverges = false;
    let mut any_err = false;
    for s in &blk.stmts {
        check_stmt(fcx, &**s);
        let s_id = ::rustc_front::util::stmt_id(&**s);
        let s_ty = fcx.node_ty(s_id);
        if any_diverges && !warned && match s.node {
            hir::StmtDecl(ref decl, _) => {
                match decl.node {
                    hir::DeclLocal(_) => true,
                    _ => false,
                }
            }
            hir::StmtExpr(_, _) | hir::StmtSemi(_, _) => true,
        } {
            fcx.ccx
                .tcx
                .sess
                .add_lint(lint::builtin::UNREACHABLE_CODE,
                          s_id,
                          s.span,
                          "unreachable statement".to_string());
            warned = true;
        }
        any_diverges = any_diverges || fcx.infcx().type_var_diverges(s_ty);
        any_err = any_err || s_ty.references_error();
    }
    match blk.expr {
        None => if any_err {
            fcx.write_error(blk.id);
        } else if any_diverges {
            fcx.write_ty(blk.id, fcx.infcx().next_diverging_ty_var());
        } else {
            fcx.write_nil(blk.id);
        },
        Some(ref e) => {
            if any_diverges && !warned {
                fcx.ccx
                    .tcx
                    .sess
                    .add_lint(lint::builtin::UNREACHABLE_CODE,
                              e.id,
                              e.span,
                              "unreachable expression".to_string());
            }
            let ety = match expected {
                ExpectHasType(ety) => {
                    check_expr_coercable_to_type(fcx, &**e, ety);
                    ety
                }
                _ => {
                    check_expr_with_expectation(fcx, &**e, expected);
                    fcx.expr_ty(&**e)
                }
            };

            if any_err {
                fcx.write_error(blk.id);
            } else if any_diverges {
                fcx.write_ty(blk.id, fcx.infcx().next_diverging_ty_var());
            } else {
                fcx.write_ty(blk.id, ety);
            }
        }
    };

    *fcx.ps.borrow_mut() = prev;
}

/// Checks a constant appearing in a type. At the moment this is just the
/// length expression in a fixed-length vector, but someday it might be
/// extended to type-level numeric literals.
fn check_const_in_type<'a,'tcx>(ccx: &'a CrateCtxt<'a,'tcx>,
                                expr: &'tcx hir::Expr,
                                expected_type: Ty<'tcx>) {
    let tables = RefCell::new(ty::Tables::empty());
    let inh = static_inherited_fields(ccx, &tables);
    let fcx = blank_fn_ctxt(ccx, &inh, ty::FnConverging(expected_type), expr.id);
    check_const_with_ty(&fcx, expr.span, expr, expected_type);
}

fn check_const<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                        sp: Span,
                        e: &'tcx hir::Expr,
                        id: ast::NodeId) {
    let tables = RefCell::new(ty::Tables::empty());
    let inh = static_inherited_fields(ccx, &tables);
    let rty = ccx.tcx.node_id_to_type(id);
    let fcx = blank_fn_ctxt(ccx, &inh, ty::FnConverging(rty), e.id);
    let declty = fcx.ccx.tcx.lookup_item_type(ccx.tcx.map.local_def_id(id)).ty;
    check_const_with_ty(&fcx, sp, e, declty);
}

fn check_const_with_ty<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                 _: Span,
                                 e: &'tcx hir::Expr,
                                 declty: Ty<'tcx>) {
    // Gather locals in statics (because of block expressions).
    // This is technically unnecessary because locals in static items are forbidden,
    // but prevents type checking from blowing up before const checking can properly
    // emit an error.
    GatherLocalsVisitor { fcx: fcx }.visit_expr(e);

    check_expr_with_hint(fcx, e, declty);
    demand::coerce(fcx, e.span, declty, e);

    fcx.select_all_obligations_and_apply_defaults();
    upvar::closure_analyze_const(&fcx, e);
    fcx.select_obligations_where_possible();
    fcx.check_casts();
    fcx.select_all_obligations_or_error();

    regionck::regionck_expr(fcx, e);
    writeback::resolve_type_vars_in_expr(fcx, e);
}

/// Checks whether a type can be represented in memory. In particular, it
/// identifies types that contain themselves without indirection through a
/// pointer, which would mean their size is unbounded.
pub fn check_representable(tcx: &ty::ctxt,
                           sp: Span,
                           item_id: ast::NodeId,
                           designation: &str) -> bool {
    let rty = tcx.node_id_to_type(item_id);

    // Check that it is possible to represent this type. This call identifies
    // (1) types that contain themselves and (2) types that contain a different
    // recursive type. It is only necessary to throw an error on those that
    // contain themselves. For case 2, there must be an inner type that will be
    // caught by case 1.
    match rty.is_representable(tcx, sp) {
        Representability::SelfRecursive => {
            span_err!(tcx.sess, sp, E0072, "invalid recursive {} type", designation);
            tcx.sess.fileline_help(
                sp, "wrap the inner value in a box to make it representable");
            return false
        }
        Representability::Representable | Representability::ContainsRecursive => (),
    }
    return true
}

pub fn check_simd(tcx: &ty::ctxt, sp: Span, id: ast::NodeId) {
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

pub fn check_enum_variants<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                    sp: Span,
                                    vs: &'tcx [P<hir::Variant>],
                                    id: ast::NodeId) {

    fn disr_in_range(ccx: &CrateCtxt,
                     ty: attr::IntType,
                     disr: ty::Disr) -> bool {
        fn uint_in_range(ccx: &CrateCtxt, ty: ast::UintTy, disr: ty::Disr) -> bool {
            match ty {
                ast::TyU8 => disr as u8 as Disr == disr,
                ast::TyU16 => disr as u16 as Disr == disr,
                ast::TyU32 => disr as u32 as Disr == disr,
                ast::TyU64 => disr as u64 as Disr == disr,
                ast::TyUs => uint_in_range(ccx, ccx.tcx.sess.target.uint_type, disr)
            }
        }
        fn int_in_range(ccx: &CrateCtxt, ty: ast::IntTy, disr: ty::Disr) -> bool {
            match ty {
                ast::TyI8 => disr as i8 as Disr == disr,
                ast::TyI16 => disr as i16 as Disr == disr,
                ast::TyI32 => disr as i32 as Disr == disr,
                ast::TyI64 => disr as i64 as Disr == disr,
                ast::TyIs => int_in_range(ccx, ccx.tcx.sess.target.int_type, disr)
            }
        }
        match ty {
            attr::UnsignedInt(ty) => uint_in_range(ccx, ty, disr),
            attr::SignedInt(ty) => int_in_range(ccx, ty, disr)
        }
    }

    fn do_check<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                          vs: &'tcx [P<hir::Variant>],
                          id: ast::NodeId,
                          hint: attr::ReprAttr) {
        #![allow(trivial_numeric_casts)]

        let rty = ccx.tcx.node_id_to_type(id);
        let mut disr_vals: Vec<ty::Disr> = Vec::new();

        let tables = RefCell::new(ty::Tables::empty());
        let inh = static_inherited_fields(ccx, &tables);
        let fcx = blank_fn_ctxt(ccx, &inh, ty::FnConverging(rty), id);

        let (_, repr_type_ty) = ccx.tcx.enum_repr_type(Some(&hint));
        for v in vs {
            if let Some(ref e) = v.node.disr_expr {
                check_const_with_ty(&fcx, e.span, e, repr_type_ty);
            }
        }

        let def_id = ccx.tcx.map.local_def_id(id);

        let variants = &ccx.tcx.lookup_adt_def(def_id).variants;
        for (v, variant) in vs.iter().zip(variants.iter()) {
            let current_disr_val = variant.disr_val;

            // Check for duplicate discriminant values
            match disr_vals.iter().position(|&x| x == current_disr_val) {
                Some(i) => {
                    span_err!(ccx.tcx.sess, v.span, E0081,
                        "discriminant value `{}` already exists", disr_vals[i]);
                    let variant_i_node_id = ccx.tcx.map.as_local_node_id(variants[i].did).unwrap();
                    span_note!(ccx.tcx.sess, ccx.tcx.map.span(variant_i_node_id),
                        "conflicting discriminant here")
                }
                None => {}
            }
            // Check for unrepresentable discriminant values
            match hint {
                attr::ReprAny | attr::ReprExtern => (),
                attr::ReprInt(sp, ity) => {
                    if !disr_in_range(ccx, ity, current_disr_val) {
                        span_err!(ccx.tcx.sess, v.span, E0082,
                            "discriminant value outside specified type");
                        span_note!(ccx.tcx.sess, sp,
                            "discriminant type specified here");
                    }
                }
                attr::ReprSimd => {
                    ccx.tcx.sess.bug("range_to_inttype: found ReprSimd on an enum");
                }
                attr::ReprPacked => {
                    ccx.tcx.sess.bug("range_to_inttype: found ReprPacked on an enum");
                }
            }
            disr_vals.push(current_disr_val);
        }
    }

    let def_id = ccx.tcx.map.local_def_id(id);
    let hint = *ccx.tcx.lookup_repr_hints(def_id).get(0).unwrap_or(&attr::ReprAny);

    if hint != attr::ReprAny && vs.len() <= 1 {
        if vs.len() == 1 {
            span_err!(ccx.tcx.sess, sp, E0083,
                "unsupported representation for univariant enum");
        } else {
            span_err!(ccx.tcx.sess, sp, E0084,
                "unsupported representation for zero-variant enum");
        };
    }

    do_check(ccx, vs, id, hint);

    check_representable(ccx.tcx, sp, id, "enum");
}

// Returns the type parameter count and the type for the given definition.
fn type_scheme_and_predicates_for_def<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                                sp: Span,
                                                defn: def::Def)
                                                -> (TypeScheme<'tcx>, GenericPredicates<'tcx>) {
    match defn {
        def::DefLocal(_, nid) | def::DefUpvar(_, nid, _, _) => {
            let typ = fcx.local_ty(sp, nid);
            (ty::TypeScheme { generics: ty::Generics::empty(), ty: typ },
             ty::GenericPredicates::empty())
        }
        def::DefFn(id, _) | def::DefMethod(id) |
        def::DefStatic(id, _) | def::DefVariant(_, id, _) |
        def::DefStruct(id) | def::DefConst(id) | def::DefAssociatedConst(id) => {
            (fcx.tcx().lookup_item_type(id), fcx.tcx().lookup_predicates(id))
        }
        def::DefTrait(_) |
        def::DefTy(..) |
        def::DefAssociatedTy(..) |
        def::DefPrimTy(_) |
        def::DefTyParam(..) |
        def::DefMod(..) |
        def::DefForeignMod(..) |
        def::DefUse(..) |
        def::DefLabel(..) |
        def::DefSelfTy(..) => {
            fcx.ccx.tcx.sess.span_bug(sp, &format!("expected value, found {:?}", defn));
        }
    }
}

// Instantiates the given path, which must refer to an item with the given
// number of type parameters and type.
pub fn instantiate_path<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                  segments: &[hir::PathSegment],
                                  type_scheme: TypeScheme<'tcx>,
                                  type_predicates: &ty::GenericPredicates<'tcx>,
                                  opt_self_ty: Option<Ty<'tcx>>,
                                  def: def::Def,
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
        def::DefSelfTy(..) |
        def::DefStruct(..) |
        def::DefVariant(..) |
        def::DefTy(..) |
        def::DefAssociatedTy(..) |
        def::DefTrait(..) |
        def::DefPrimTy(..) |
        def::DefTyParam(..) => {
            // Everything but the final segment should have no
            // parameters at all.
            segment_spaces = vec![None; segments.len() - 1];
            segment_spaces.push(Some(subst::TypeSpace));
        }

        // Case 2. Reference to a top-level value.
        def::DefFn(..) |
        def::DefConst(..) |
        def::DefStatic(..) => {
            segment_spaces = vec![None; segments.len() - 1];
            segment_spaces.push(Some(subst::FnSpace));
        }

        // Case 3. Reference to a method.
        def::DefMethod(def_id) => {
            let container = fcx.tcx().impl_or_trait_item(def_id).container();
            match container {
                ty::TraitContainer(trait_did) => {
                    callee::check_legal_trait_for_method_call(fcx.ccx, span, trait_did)
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

        def::DefAssociatedConst(def_id) => {
            let container = fcx.tcx().impl_or_trait_item(def_id).container();
            match container {
                ty::TraitContainer(trait_did) => {
                    callee::check_legal_trait_for_method_call(fcx.ccx, span, trait_did)
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
        def::DefMod(..) |
        def::DefForeignMod(..) |
        def::DefLocal(..) |
        def::DefUse(..) |
        def::DefLabel(..) |
        def::DefUpvar(..) => {
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
    for (opt_space, segment) in segment_spaces.iter().zip(segments) {
        match *opt_space {
            None => {
                prohibit_type_params(fcx.tcx(), slice::ref_slice(segment));
            }

            Some(space) => {
                push_explicit_parameters_from_segment_to_substs(fcx,
                                                                space,
                                                                span,
                                                                type_defs,
                                                                region_defs,
                                                                segment,
                                                                &mut substs);
            }
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
        adjust_type_parameters(fcx, span, space, type_defs,
                               require_type_space, &mut substs);
        assert_eq!(substs.types.len(space), type_defs.len(space));

        adjust_region_parameters(fcx, span, space, region_defs, &mut substs);
        assert_eq!(substs.regions().len(space), region_defs.len(space));
    }

    // The things we are substituting into the type should not contain
    // escaping late-bound regions, and nor should the base type scheme.
    assert!(!substs.has_regions_escaping_depth(0));
    assert!(!type_scheme.has_escaping_regions());

    // Add all the obligations that are required, substituting and
    // normalized appropriately.
    let bounds = fcx.instantiate_bounds(span, &substs, &type_predicates);
    fcx.add_obligations_for_parameters(
        traits::ObligationCause::new(span, fcx.body_id, traits::ItemObligation(def.def_id())),
        &bounds);

    // Substitute the values for the type parameters into the type of
    // the referenced item.
    let ty_substituted = fcx.instantiate_type_scheme(span, &substs, &type_scheme.ty);


    if let Some((ty::ImplContainer(impl_def_id), self_ty)) = ufcs_associated {
        // In the case of `Foo<T>::method` and `<Foo<T>>::method`, if `method`
        // is inherent, there is no `Self` parameter, instead, the impl needs
        // type parameters, which we can infer by unifying the provided `Self`
        // with the substituted impl type.
        let impl_scheme = fcx.tcx().lookup_item_type(impl_def_id);
        assert_eq!(substs.types.len(subst::TypeSpace),
                   impl_scheme.generics.types.len(subst::TypeSpace));
        assert_eq!(substs.regions().len(subst::TypeSpace),
                   impl_scheme.generics.regions.len(subst::TypeSpace));

        let impl_ty = fcx.instantiate_type_scheme(span, &substs, &impl_scheme.ty);
        if fcx.mk_subty(false, TypeOrigin::Misc(span), self_ty, impl_ty).is_err() {
            fcx.tcx().sess.span_bug(span,
            &format!(
                "instantiate_path: (UFCS) {:?} was a subtype of {:?} but now is not?",
                self_ty,
                impl_ty));
        }
    }

    debug!("instantiate_path: type of {:?} is {:?}",
           node_id,
           ty_substituted);
    fcx.write_ty(node_id, ty_substituted);
    fcx.write_substs(node_id, ty::ItemSubsts { substs: substs });
    return;

    /// Finds the parameters that the user provided and adds them to `substs`. If too many
    /// parameters are provided, then reports an error and clears the output vector.
    ///
    /// We clear the output vector because that will cause the `adjust_XXX_parameters()` later to
    /// use inference variables. This seems less likely to lead to derived errors.
    ///
    /// Note that we *do not* check for *too few* parameters here. Due to the presence of defaults
    /// etc that is more complicated. I wanted however to do the reporting of *too many* parameters
    /// here because we can easily use the precise span of the N+1'th parameter.
    fn push_explicit_parameters_from_segment_to_substs<'a, 'tcx>(
        fcx: &FnCtxt<'a, 'tcx>,
        space: subst::ParamSpace,
        span: Span,
        type_defs: &VecPerParamSpace<ty::TypeParameterDef<'tcx>>,
        region_defs: &VecPerParamSpace<ty::RegionParameterDef>,
        segment: &hir::PathSegment,
        substs: &mut Substs<'tcx>)
    {
        match segment.parameters {
            hir::AngleBracketedParameters(ref data) => {
                push_explicit_angle_bracketed_parameters_from_segment_to_substs(
                    fcx, space, type_defs, region_defs, data, substs);
            }

            hir::ParenthesizedParameters(ref data) => {
                span_err!(fcx.tcx().sess, span, E0238,
                    "parenthesized parameters may only be used with a trait");
                push_explicit_parenthesized_parameters_from_segment_to_substs(
                    fcx, space, span, type_defs, data, substs);
            }
        }
    }

    fn push_explicit_angle_bracketed_parameters_from_segment_to_substs<'a, 'tcx>(
        fcx: &FnCtxt<'a, 'tcx>,
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
                let t = fcx.to_ty(&**typ);
                if i < type_count {
                    substs.types.push(space, t);
                } else if i == type_count {
                    span_err!(fcx.tcx().sess, typ.span, E0087,
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
            span_err!(fcx.tcx().sess, data.bindings[0].span, E0182,
                      "unexpected binding of associated item in expression path \
                       (only allowed in type paths)");
        }

        {
            let region_count = region_defs.len(space);
            assert_eq!(substs.regions().len(space), 0);
            for (i, lifetime) in data.lifetimes.iter().enumerate() {
                let r = ast_region_to_region(fcx.tcx(), lifetime);
                if i < region_count {
                    substs.mut_regions().push(space, r);
                } else if i == region_count {
                    span_err!(fcx.tcx().sess, lifetime.span, E0088,
                        "too many lifetime parameters provided: \
                         expected {} parameter{}, found {} parameter{}",
                        region_count,
                        if region_count == 1 {""} else {"s"},
                        data.lifetimes.len(),
                        if data.lifetimes.len() == 1 {""} else {"s"});
                    substs.mut_regions().truncate(space, 0);
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
    fn push_explicit_parenthesized_parameters_from_segment_to_substs<'a, 'tcx>(
        fcx: &FnCtxt<'a, 'tcx>,
        space: subst::ParamSpace,
        span: Span,
        type_defs: &VecPerParamSpace<ty::TypeParameterDef<'tcx>>,
        data: &hir::ParenthesizedParameterData,
        substs: &mut Substs<'tcx>)
    {
        let type_count = type_defs.len(space);
        if type_count < 2 {
            span_err!(fcx.tcx().sess, span, E0167,
                      "parenthesized form always supplies 2 type parameters, \
                      but only {} parameter(s) were expected",
                      type_count);
        }

        let input_tys: Vec<Ty> =
            data.inputs.iter().map(|ty| fcx.to_ty(&**ty)).collect();

        let tuple_ty = fcx.tcx().mk_tup(input_tys);

        if type_count >= 1 {
            substs.types.push(space, tuple_ty);
        }

        let output_ty: Option<Ty> =
            data.output.as_ref().map(|ty| fcx.to_ty(&**ty));

        let output_ty =
            output_ty.unwrap_or(fcx.tcx().mk_nil());

        if type_count >= 2 {
            substs.types.push(space, output_ty);
        }
    }

    fn adjust_type_parameters<'a, 'tcx>(
        fcx: &FnCtxt<'a, 'tcx>,
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
            fcx.infcx().type_vars_for_defs(span, space, substs, &desired[..]);
            return;
        }

        // Too few parameters specified: report an error and use Err
        // for everything.
        if provided_len < required_len {
            let qualifier =
                if desired.len() != required_len { "at least " } else { "" };
            span_err!(fcx.tcx().sess, span, E0089,
                "too few type parameters provided: expected {}{} parameter{}, \
                 found {} parameter{}",
                qualifier, required_len,
                if required_len == 1 {""} else {"s"},
                provided_len,
                if provided_len == 1 {""} else {"s"});
            substs.types.replace(space, vec![fcx.tcx().types.err; desired.len()]);
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
            let default = default.subst_spanned(fcx.tcx(), substs, Some(span));
            substs.types.push(space, default);
        }
        assert_eq!(substs.types.len(space), desired.len());

        debug!("Final substs: {:?}", substs);
    }

    fn adjust_region_parameters(
        fcx: &FnCtxt,
        span: Span,
        space: ParamSpace,
        defs: &VecPerParamSpace<ty::RegionParameterDef>,
        substs: &mut Substs)
    {
        let provided_len = substs.mut_regions().len(space);
        let desired = defs.get_slice(space);

        // Enforced by `push_explicit_parameters_from_segment_to_substs()`.
        assert!(provided_len <= desired.len());

        // If nothing was provided, just use inference variables.
        if provided_len == 0 {
            substs.mut_regions().replace(
                space,
                fcx.infcx().region_vars_for_defs(span, desired));
            return;
        }

        // If just the right number were provided, everybody is happy.
        if provided_len == desired.len() {
            return;
        }

        // Otherwise, too few were provided. Report an error and then
        // use inference variables.
        span_err!(fcx.tcx().sess, span, E0090,
            "too few lifetime parameters provided: expected {} parameter{}, \
             found {} parameter{}",
            desired.len(),
            if desired.len() == 1 {""} else {"s"},
            provided_len,
            if provided_len == 1 {""} else {"s"});

        substs.mut_regions().replace(
            space,
            fcx.infcx().region_vars_for_defs(span, desired));
    }
}

fn structurally_resolve_type_or_else<'a, 'tcx, F>(fcx: &FnCtxt<'a, 'tcx>,
                                                  sp: Span,
                                                  ty: Ty<'tcx>,
                                                  f: F) -> Ty<'tcx>
    where F: Fn() -> Ty<'tcx>
{
    let mut ty = fcx.resolve_type_vars_if_possible(ty);

    if ty.is_ty_var() {
        let alternative = f();

        // If not, error.
        if alternative.is_ty_var() || alternative.references_error() {
            fcx.type_error_message(sp, |_actual| {
                "the type of this value must be known in this context".to_string()
            }, ty, None);
            demand::suptype(fcx, sp, fcx.tcx().types.err, ty);
            ty = fcx.tcx().types.err;
        } else {
            demand::suptype(fcx, sp, alternative, ty);
            ty = alternative;
        }
    }

    ty
}

// Resolves `typ` by a single level if `typ` is a type variable.  If no
// resolution is possible, then an error is reported.
pub fn structurally_resolved_type<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                            sp: Span,
                                            ty: Ty<'tcx>)
                                            -> Ty<'tcx>
{
    structurally_resolve_type_or_else(fcx, sp, ty, || {
        fcx.tcx().types.err
    })
}

// Returns true if b contains a break that can exit from b
pub fn may_break(cx: &ty::ctxt, id: ast::NodeId, b: &hir::Block) -> bool {
    // First: is there an unlabeled break immediately
    // inside the loop?
    (loop_query(&*b, |e| {
        match *e {
            hir::ExprBreak(None) => true,
            _ => false
        }
    })) ||
    // Second: is there a labeled break with label
    // <id> nested anywhere inside the loop?
    (block_query(b, |e| {
        if let hir::ExprBreak(Some(_)) = e.node {
            lookup_full_def(cx, e.span, e.id) == def::DefLabel(id)
        } else {
            false
        }
    }))
}

pub fn check_bounds_are_used<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                       span: Span,
                                       tps: &OwnedSlice<hir::TyParam>,
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
            span_err!(ccx.tcx.sess, span, E0091,
                "type parameter `{}` is unused",
                tps[i].name);
        }
    }
}
