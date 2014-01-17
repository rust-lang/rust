// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
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
  through the `demand` module.  The `typeck::infer` module is in charge
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
stored in `fcx.node_types` and `fcx.node_type_substs`.  These types
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


use middle::const_eval;
use middle::lang_items::{ExchangeHeapLangItem, GcLangItem};
use middle::lang_items::{ManagedHeapLangItem};
use middle::lint::UnreachableCode;
use middle::pat_util::pat_id_map;
use middle::pat_util;
use middle::subst::Subst;
use middle::ty::{FnSig, VariantInfo};
use middle::ty::{ty_param_bounds_and_ty, ty_param_substs_and_ty};
use middle::ty::{substs, param_ty, Disr, ExprTyProvider};
use middle::ty;
use middle::ty_fold::TypeFolder;
use middle::typeck::astconv::AstConv;
use middle::typeck::astconv::{ast_region_to_region, ast_ty_to_ty};
use middle::typeck::astconv;
use middle::typeck::check::_match::pat_ctxt;
use middle::typeck::check::method::{AutoderefReceiver};
use middle::typeck::check::method::{AutoderefReceiverFlag};
use middle::typeck::check::method::{CheckTraitsAndInherentMethods};
use middle::typeck::check::method::{CheckTraitsOnly, DontAutoderefReceiver};
use middle::typeck::check::regionmanip::replace_bound_regions_in_fn_sig;
use middle::typeck::check::regionmanip::relate_free_regions;
use middle::typeck::check::vtable::{LocationInfo, VtableContext};
use middle::typeck::CrateCtxt;
use middle::typeck::infer::{resolve_type, force_tvar};
use middle::typeck::infer;
use middle::typeck::rscope::RegionScope;
use middle::typeck::{lookup_def_ccx};
use middle::typeck::no_params;
use middle::typeck::{require_same_types, method_map, vtable_map};
use middle::lang_items::TypeIdLangItem;
use util::common::{block_query, indenter, loop_query};
use util::ppaux;
use util::ppaux::{UserString, Repr};

use std::cell::{Cell, RefCell};
use std::hashmap::HashMap;
use std::result;
use std::util::replace;
use std::vec;
use syntax::abi::AbiSet;
use syntax::ast::{Provided, Required};
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::local_def;
use syntax::ast_util;
use syntax::attr;
use syntax::codemap::Span;
use syntax::codemap;
use syntax::opt_vec::OptVec;
use syntax::opt_vec;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::visit;
use syntax::visit::Visitor;
use syntax;

pub mod _match;
pub mod vtable;
pub mod writeback;
pub mod regionmanip;
pub mod regionck;
pub mod demand;
pub mod method;

pub struct SelfInfo {
    self_ty: ty::t,
    self_id: ast::NodeId,
    span: Span
}

/// Fields that are part of a `FnCtxt` which are inherited by
/// closures defined within the function.  For example:
///
///     fn foo() {
///         do bar() { ... }
///     }
///
/// Here, the function `foo()` and the closure passed to
/// `bar()` will each have their own `FnCtxt`, but they will
/// share the inherited fields.
pub struct Inherited {
    infcx: @infer::InferCtxt,
    locals: @RefCell<HashMap<ast::NodeId, ty::t>>,
    param_env: ty::ParameterEnvironment,

    // Temporary tables:
    node_types: RefCell<HashMap<ast::NodeId, ty::t>>,
    node_type_substs: RefCell<HashMap<ast::NodeId, ty::substs>>,
    adjustments: RefCell<HashMap<ast::NodeId, @ty::AutoAdjustment>>,
    method_map: method_map,
    vtable_map: vtable_map,
}

#[deriving(Clone)]
pub enum FnKind {
    // A do-closure.
    DoBlock,

    // A normal closure or fn item.
    Vanilla
}

#[deriving(Clone)]
pub struct PurityState {
    def: ast::NodeId,
    purity: ast::Purity,
    priv from_fn: bool
}

impl PurityState {
    pub fn function(purity: ast::Purity, def: ast::NodeId) -> PurityState {
        PurityState { def: def, purity: purity, from_fn: true }
    }

    pub fn recurse(&mut self, blk: &ast::Block) -> PurityState {
        match self.purity {
            // If this unsafe, then if the outer function was already marked as
            // unsafe we shouldn't attribute the unsafe'ness to the block. This
            // way the block can be warned about instead of ignoring this
            // extraneous block (functions are never warned about).
            ast::UnsafeFn if self.from_fn => *self,

            purity => {
                let (purity, def) = match blk.rules {
                    ast::UnsafeBlock(..) => (ast::UnsafeFn, blk.id),
                    ast::DefaultBlock => (purity, self.def),
                };
                PurityState{ def: def,
                             purity: purity,
                             from_fn: false }
            }
        }
    }
}

/// Whether `check_binop` is part of an assignment or not.
/// Used to know wether we allow user overloads and to print
/// better messages on error.
#[deriving(Eq)]
enum IsBinopAssignment{
    SimpleBinop,
    BinopAssignment,
}

#[deriving(Clone)]
pub struct FnCtxt {
    // Number of errors that had been reported when we started
    // checking this function. On exit, if we find that *more* errors
    // have been reported, we will skip regionck and other work that
    // expects the types within the function to be consistent.
    err_count_on_creation: uint,

    ret_ty: ty::t,
    ps: RefCell<PurityState>,

    // Sometimes we generate region pointers where the precise region
    // to use is not known. For example, an expression like `&x.f`
    // where `x` is of type `@T`: in this case, we will be rooting
    // `x` onto the stack frame, and we could choose to root it until
    // the end of (almost) any enclosing block or expression.  We
    // want to pick the narrowest block that encompasses all uses.
    //
    // What we do in such cases is to generate a region variable with
    // `region_lb` as a lower bound.  The regionck pass then adds
    // other constriants based on how the variable is used and region
    // inference selects the ultimate value.  Finally, borrowck is
    // charged with guaranteeing that the value whose address was taken
    // can actually be made to live as long as it needs to live.
    region_lb: Cell<ast::NodeId>,

    // Says whether we're inside a for loop, in a do block
    // or neither. Helps with error messages involving the
    // function return type.
    fn_kind: FnKind,

    inh: @Inherited,

    ccx: @CrateCtxt,
}

impl Inherited {
    fn new(tcx: ty::ctxt,
           param_env: ty::ParameterEnvironment)
           -> Inherited {
        Inherited {
            infcx: infer::new_infer_ctxt(tcx),
            locals: @RefCell::new(HashMap::new()),
            param_env: param_env,
            node_types: RefCell::new(HashMap::new()),
            node_type_substs: RefCell::new(HashMap::new()),
            adjustments: RefCell::new(HashMap::new()),
            method_map: @RefCell::new(HashMap::new()),
            vtable_map: @RefCell::new(HashMap::new()),
        }
    }
}

// Used by check_const and check_enum_variants
pub fn blank_fn_ctxt(ccx: @CrateCtxt,
                     rty: ty::t,
                     region_bnd: ast::NodeId)
                     -> @FnCtxt {
    // It's kind of a kludge to manufacture a fake function context
    // and statement context, but we might as well do write the code only once
    let param_env = ty::ParameterEnvironment { free_substs: substs::empty(),
                                               self_param_bound: None,
                                               type_param_bounds: ~[] };
    @FnCtxt {
        err_count_on_creation: ccx.tcx.sess.err_count(),
        ret_ty: rty,
        ps: RefCell::new(PurityState::function(ast::ImpureFn, 0)),
        region_lb: Cell::new(region_bnd),
        fn_kind: Vanilla,
        inh: @Inherited::new(ccx.tcx, param_env),
        ccx: ccx
    }
}

impl ExprTyProvider for FnCtxt {
    fn expr_ty(&self, ex: &ast::Expr) -> ty::t {
        self.expr_ty(ex)
    }

    fn ty_ctxt(&self) -> ty::ctxt {
        self.ccx.tcx
    }
}

struct CheckItemTypesVisitor { ccx: @CrateCtxt }

impl Visitor<()> for CheckItemTypesVisitor {
    fn visit_item(&mut self, i: &ast::Item, _: ()) {
        check_item(self.ccx, i);
        visit::walk_item(self, i, ());
    }
}

pub fn check_item_types(ccx: @CrateCtxt, crate: &ast::Crate) {
    let mut visit = CheckItemTypesVisitor { ccx: ccx };
    visit::walk_crate(&mut visit, crate, ());
}

pub fn check_bare_fn(ccx: @CrateCtxt,
                     decl: &ast::FnDecl,
                     body: &ast::Block,
                     id: ast::NodeId,
                     self_info: Option<SelfInfo>,
                     fty: ty::t,
                     param_env: ty::ParameterEnvironment) {
    match ty::get(fty).sty {
        ty::ty_bare_fn(ref fn_ty) => {
            let fcx =
                check_fn(ccx, self_info, fn_ty.purity,
                         &fn_ty.sig, decl, id, body, Vanilla,
                         @Inherited::new(ccx.tcx, param_env));

            vtable::resolve_in_block(fcx, body);
            regionck::regionck_fn(fcx, body);
            writeback::resolve_type_vars_in_fn(fcx, decl, body, self_info);
        }
        _ => ccx.tcx.sess.impossible_case(body.span,
                                 "check_bare_fn: function type expected")
    }
}

struct GatherLocalsVisitor {
                     fcx: @FnCtxt,
                     tcx: ty::ctxt,
}

impl GatherLocalsVisitor {
    fn assign(&mut self, nid: ast::NodeId, ty_opt: Option<ty::t>) {
            match ty_opt {
                None => {
                    // infer the variable's type
                    let var_id = self.fcx.infcx().next_ty_var_id();
                    let var_ty = ty::mk_var(self.fcx.tcx(), var_id);
                    let mut locals = self.fcx.inh.locals.borrow_mut();
                    locals.get().insert(nid, var_ty);
                }
                Some(typ) => {
                    // take type that the user specified
                    let mut locals = self.fcx.inh.locals.borrow_mut();
                    locals.get().insert(nid, typ);
                }
            }
    }
}

impl Visitor<()> for GatherLocalsVisitor {
        // Add explicitly-declared locals.
    fn visit_local(&mut self, local: &ast::Local, _: ()) {
            let o_ty = match local.ty.node {
              ast::TyInfer => None,
              _ => Some(self.fcx.to_ty(local.ty))
            };
            self.assign(local.id, o_ty);
            {
                let locals = self.fcx.inh.locals.borrow();
                debug!("Local variable {} is assigned type {}",
                       self.fcx.pat_to_str(local.pat),
                       self.fcx.infcx().ty_to_str(
                           locals.get().get_copy(&local.id)));
            }
            visit::walk_local(self, local, ());

    }
        // Add pattern bindings.
    fn visit_pat(&mut self, p: &ast::Pat, _: ()) {
            match p.node {
              ast::PatIdent(_, ref path, _)
                  if pat_util::pat_is_binding(self.fcx.ccx.tcx.def_map, p) => {
                self.assign(p.id, None);
                {
                    let locals = self.fcx.inh.locals.borrow();
                    debug!("Pattern binding {} is assigned to {}",
                           self.tcx.sess.str_of(path.segments[0].identifier),
                           self.fcx.infcx().ty_to_str(
                               locals.get().get_copy(&p.id)));
                }
              }
              _ => {}
            }
            visit::walk_pat(self, p, ());

    }

    fn visit_block(&mut self, b: &ast::Block, _: ()) {
        // non-obvious: the `blk` variable maps to region lb, so
        // we have to keep this up-to-date.  This
        // is... unfortunate.  It'd be nice to not need this.
        self.fcx.with_region_lb(b.id, || visit::walk_block(self, b, ()));
    }

    // Don't descend into fns and items
    fn visit_fn(&mut self, _: &visit::FnKind, _: &ast::FnDecl,
                _: &ast::Block, _: Span, _: ast::NodeId, _: ()) { }
    fn visit_item(&mut self, _: &ast::Item, _: ()) { }

}

pub fn check_fn(ccx: @CrateCtxt,
                opt_self_info: Option<SelfInfo>,
                purity: ast::Purity,
                fn_sig: &ty::FnSig,
                decl: &ast::FnDecl,
                id: ast::NodeId,
                body: &ast::Block,
                fn_kind: FnKind,
                inherited: @Inherited) -> @FnCtxt
{
    /*!
     * Helper used by check_bare_fn and check_expr_fn.  Does the
     * grungy work of checking a function body and returns the
     * function context used for that purpose, since in the case of a
     * fn item there is still a bit more to do.
     *
     * - ...
     * - inherited: other fields inherited from the enclosing fn (if any)
     */

    let tcx = ccx.tcx;
    let err_count_on_creation = tcx.sess.err_count();

    // First, we have to replace any bound regions in the fn and self
    // types with free ones.  The free region references will be bound
    // the node_id of the body block.
    let (opt_self_info, fn_sig) = {
        let opt_self_ty = opt_self_info.map(|i| i.self_ty);
        let (_, opt_self_ty, fn_sig) =
            replace_bound_regions_in_fn_sig(
                tcx, opt_self_ty, fn_sig,
                |br| ty::ReFree(ty::FreeRegion {scope_id: body.id,
                                                 bound_region: br}));
        let opt_self_info =
            opt_self_info.map(
                |si| SelfInfo {self_ty: opt_self_ty.unwrap(), .. si});
        (opt_self_info, fn_sig)
    };

    relate_free_regions(tcx, opt_self_info.map(|s| s.self_ty), &fn_sig);

    let arg_tys = fn_sig.inputs.map(|a| *a);
    let ret_ty = fn_sig.output;

    debug!("check_fn(arg_tys={:?}, ret_ty={:?}, opt_self_ty={:?})",
           arg_tys.map(|&a| ppaux::ty_to_str(tcx, a)),
           ppaux::ty_to_str(tcx, ret_ty),
           opt_self_info.map(|si| ppaux::ty_to_str(tcx, si.self_ty)));

    // Create the function context.  This is either derived from scratch or,
    // in the case of function expressions, based on the outer context.
    let fcx: @FnCtxt = {
        @FnCtxt {
            err_count_on_creation: err_count_on_creation,
            ret_ty: ret_ty,
            ps: RefCell::new(PurityState::function(purity, id)),
            region_lb: Cell::new(body.id),
            fn_kind: fn_kind,
            inh: inherited,
            ccx: ccx
        }
    };

    gather_locals(fcx, decl, body, arg_tys, opt_self_info);
    check_block_with_expected(fcx, body, Some(ret_ty));

    // We unify the tail expr's type with the
    // function result type, if there is a tail expr.
    match body.expr {
      Some(tail_expr) => {
        let tail_expr_ty = fcx.expr_ty(tail_expr);
        // Special case: we print a special error if there appears
        // to be do-block/for-loop confusion
        demand::suptype_with_fn(fcx, tail_expr.span, false,
            fcx.ret_ty, tail_expr_ty,
            |sp, e, a, s| {
                fcx.report_mismatched_return_types(sp, e, a, s) });
      }
      None => ()
    }

    for self_info in opt_self_info.iter() {
        fcx.write_ty(self_info.self_id, self_info.self_ty);
    }
    for (input, arg) in decl.inputs.iter().zip(arg_tys.iter()) {
        fcx.write_ty(input.id, *arg);
    }

    return fcx;

    fn gather_locals(fcx: @FnCtxt,
                     decl: &ast::FnDecl,
                     body: &ast::Block,
                     arg_tys: &[ty::t],
                     opt_self_info: Option<SelfInfo>) {
        let tcx = fcx.ccx.tcx;

        let mut visit = GatherLocalsVisitor { fcx: fcx, tcx: tcx, };

        // Add the self parameter
        for self_info in opt_self_info.iter() {
            visit.assign(self_info.self_id, Some(self_info.self_ty));
            let locals = fcx.inh.locals.borrow();
            debug!("self is assigned to {}",
                   fcx.infcx().ty_to_str(
                       locals.get().get_copy(&self_info.self_id)));
        }

        // Add formal parameters.
        for (arg_ty, input) in arg_tys.iter().zip(decl.inputs.iter()) {
            // Create type variables for each argument.
            pat_util::pat_bindings(tcx.def_map,
                                   input.pat,
                                   |_bm, pat_id, _sp, _path| {
                visit.assign(pat_id, None);
            });

            // Check the pattern.
            let pcx = pat_ctxt {
                fcx: fcx,
                map: pat_id_map(tcx.def_map, input.pat),
            };
            _match::check_pat(&pcx, input.pat, *arg_ty);
        }

        visit.visit_block(body, ());
    }
}

pub fn check_no_duplicate_fields(tcx: ty::ctxt,
                                 fields: ~[(ast::Ident, Span)]) {
    let mut field_names = HashMap::new();

    for p in fields.iter() {
        let (id, sp) = *p;
        let orig_sp = field_names.find(&id).map(|x| *x);
        match orig_sp {
            Some(orig_sp) => {
                tcx.sess.span_err(sp, format!("Duplicate field name {} in record type declaration",
                                              tcx.sess.str_of(id)));
                tcx.sess.span_note(orig_sp, "First declaration of this field occurred here");
                break;
            }
            None => {
                field_names.insert(id, sp);
            }
        }
    }
}

pub fn check_struct(ccx: @CrateCtxt, id: ast::NodeId, span: Span) {
    let tcx = ccx.tcx;

    // Check that the class is instantiable
    check_instantiable(tcx, span, id);

    if ty::lookup_simd(tcx, local_def(id)) {
        check_simd(tcx, span, id);
    }
}

pub fn check_item(ccx: @CrateCtxt, it: &ast::Item) {
    debug!("check_item(it.id={}, it.ident={})",
           it.id,
           ty::item_path_str(ccx.tcx, local_def(it.id)));
    let _indenter = indenter();

    match it.node {
      ast::ItemStatic(_, _, e) => check_const(ccx, it.span, e, it.id),
      ast::ItemEnum(ref enum_definition, _) => {
        check_enum_variants(ccx,
                            it.span,
                            enum_definition.variants,
                            it.id);
      }
      ast::ItemFn(decl, _, _, _, body) => {
        let fn_tpt = ty::lookup_item_type(ccx.tcx, ast_util::local_def(it.id));

        // FIXME(#5121) -- won't work for lifetimes that appear in type bounds
        let param_env = ty::construct_parameter_environment(
                ccx.tcx,
                None,
                *fn_tpt.generics.type_param_defs,
                [],
                [],
                body.id);

        check_bare_fn(ccx, decl, body, it.id, None, fn_tpt.ty, param_env);
      }
      ast::ItemImpl(_, ref opt_trait_ref, _, ref ms) => {
        debug!("ItemImpl {} with id {}", ccx.tcx.sess.str_of(it.ident), it.id);

        let impl_tpt = ty::lookup_item_type(ccx.tcx, ast_util::local_def(it.id));
        for m in ms.iter() {
            check_method_body(ccx, &impl_tpt.generics, None, *m);
        }

        match *opt_trait_ref {
            Some(ref ast_trait_ref) => {
                let impl_trait_ref =
                    ty::node_id_to_trait_ref(ccx.tcx, ast_trait_ref.ref_id);
                check_impl_methods_against_trait(ccx,
                                             it.span,
                                             &impl_tpt.generics,
                                             ast_trait_ref,
                                             impl_trait_ref,
                                             *ms);
                vtable::resolve_impl(ccx, it, &impl_tpt.generics,
                                     impl_trait_ref);
            }
            None => { }
        }

      }
      ast::ItemTrait(_, _, ref trait_methods) => {
        let trait_def = ty::lookup_trait_def(ccx.tcx, local_def(it.id));
        for trait_method in (*trait_methods).iter() {
            match *trait_method {
                Required(..) => {
                    // Nothing to do, since required methods don't have
                    // bodies to check.
                }
                Provided(m) => {
                    check_method_body(ccx, &trait_def.generics,
                                      Some(trait_def.trait_ref), m);
                }
            }
        }
      }
      ast::ItemStruct(..) => {
        check_struct(ccx, it.id, it.span);
      }
      ast::ItemTy(ref t, ref generics) => {
        let tpt_ty = ty::node_id_to_type(ccx.tcx, it.id);
        check_bounds_are_used(ccx, t.span, &generics.ty_params, tpt_ty);
      }
      ast::ItemForeignMod(ref m) => {
        if m.abis.is_intrinsic() {
            for item in m.items.iter() {
                check_intrinsic_type(ccx, *item);
            }
        } else {
            for item in m.items.iter() {
                let tpt = ty::lookup_item_type(ccx.tcx, local_def(item.id));
                if tpt.generics.has_type_params() {
                    ccx.tcx.sess.span_err(item.span, "foreign items may not have type parameters");
                }

                match item.node {
                    ast::ForeignItemFn(ref fn_decl, _) => {
                        if fn_decl.variadic && !m.abis.is_c() {
                            ccx.tcx.sess.span_err(
                                item.span, "variadic function must have C calling convention");
                        }
                    }
                    _ => {}
                }
            }
        }
      }
      _ => {/* nothing to do */ }
    }
}

fn check_method_body(ccx: @CrateCtxt,
                     item_generics: &ty::Generics,
                     self_bound: Option<@ty::TraitRef>,
                     method: &ast::Method) {
    /*!
     * Type checks a method body.
     *
     * # Parameters
     * - `item_generics`: generics defined on the impl/trait that contains
     *   the method
     * - `self_bound`: bound for the `Self` type parameter, if any
     * - `method`: the method definition
     */

    debug!("check_method_body(item_generics={}, \
            self_bound={}, \
            method.id={})",
            item_generics.repr(ccx.tcx),
            self_bound.repr(ccx.tcx),
            method.id);
    let method_def_id = local_def(method.id);
    let method_ty = ty::method(ccx.tcx, method_def_id);
    let method_generics = &method_ty.generics;

    let param_env =
        ty::construct_parameter_environment(
            ccx.tcx,
            self_bound,
            *item_generics.type_param_defs,
            *method_generics.type_param_defs,
            item_generics.region_param_defs,
            method.body.id);

    // Compute the self type and fty from point of view of inside fn
    let opt_self_info = method_ty.transformed_self_ty.map(|ty| {
        SelfInfo {self_ty: ty.subst(ccx.tcx, &param_env.free_substs),
                  self_id: method.self_id,
                  span: method.explicit_self.span}
    });
    let fty = ty::node_id_to_type(ccx.tcx, method.id);
    let fty = fty.subst(ccx.tcx, &param_env.free_substs);

    check_bare_fn(
        ccx,
        method.decl,
        method.body,
        method.id,
        opt_self_info,
        fty,
        param_env);
}

fn check_impl_methods_against_trait(ccx: @CrateCtxt,
                                    impl_span: Span,
                                    impl_generics: &ty::Generics,
                                    ast_trait_ref: &ast::TraitRef,
                                    impl_trait_ref: &ty::TraitRef,
                                    impl_methods: &[@ast::Method]) {
    // Locate trait methods
    let tcx = ccx.tcx;
    let trait_methods = ty::trait_methods(tcx, impl_trait_ref.def_id);

    // Check existing impl methods to see if they are both present in trait
    // and compatible with trait signature
    for impl_method in impl_methods.iter() {
        let impl_method_def_id = local_def(impl_method.id);
        let impl_method_ty = ty::method(ccx.tcx, impl_method_def_id);

        // If this is an impl of a trait method, find the corresponding
        // method definition in the trait.
        let opt_trait_method_ty =
            trait_methods.iter().
            find(|tm| tm.ident.name == impl_method_ty.ident.name);
        match opt_trait_method_ty {
            Some(trait_method_ty) => {
                compare_impl_method(ccx.tcx,
                                    impl_generics,
                                    impl_method_ty,
                                    impl_method.span,
                                    impl_method.body.id,
                                    *trait_method_ty,
                                    &impl_trait_ref.substs);
            }
            None => {
                tcx.sess.span_err(
                    impl_method.span,
                    format!("method `{}` is not a member of trait `{}`",
                            tcx.sess.str_of(impl_method_ty.ident),
                            pprust::path_to_str(&ast_trait_ref.path,
                                                tcx.sess.intr())));
            }
        }
    }

    // Check for missing methods from trait
    let provided_methods = ty::provided_trait_methods(tcx,
                                                      impl_trait_ref.def_id);
    let mut missing_methods = ~[];
    for trait_method in trait_methods.iter() {
        let is_implemented =
            impl_methods.iter().any(
                |m| m.ident.name == trait_method.ident.name);
        let is_provided =
            provided_methods.iter().any(
                |m| m.ident.name == trait_method.ident.name);
        if !is_implemented && !is_provided {
            missing_methods.push(
                format!("`{}`", ccx.tcx.sess.str_of(trait_method.ident)));
        }
    }

    if !missing_methods.is_empty() {
        tcx.sess.span_err(
            impl_span,
            format!("not all trait methods implemented, missing: {}",
                    missing_methods.connect(", ")));
    }
}

/**
 * Checks that a method from an impl/class conforms to the signature of
 * the same method as declared in the trait.
 *
 * # Parameters
 *
 * - impl_generics: the generics declared on the impl itself (not the method!)
 * - impl_m: type of the method we are checking
 * - impl_m_span: span to use for reporting errors
 * - impl_m_body_id: id of the method body
 * - trait_m: the method in the trait
 * - trait_substs: the substitutions used on the type of the trait
 */
pub fn compare_impl_method(tcx: ty::ctxt,
                           impl_generics: &ty::Generics,
                           impl_m: @ty::Method,
                           impl_m_span: Span,
                           impl_m_body_id: ast::NodeId,
                           trait_m: &ty::Method,
                           trait_substs: &ty::substs) {
    debug!("compare_impl_method()");
    let infcx = infer::new_infer_ctxt(tcx);

    let impl_tps = impl_generics.type_param_defs.len();

    // Try to give more informative error messages about self typing
    // mismatches.  Note that any mismatch will also be detected
    // below, where we construct a canonical function type that
    // includes the self parameter as a normal parameter.  It's just
    // that the error messages you get out of this code are a bit more
    // inscrutable, particularly for cases where one method has no
    // self.
    match (&trait_m.explicit_self, &impl_m.explicit_self) {
        (&ast::SelfStatic, &ast::SelfStatic) => {}
        (&ast::SelfStatic, _) => {
            tcx.sess.span_err(
                impl_m_span,
                format!("method `{}` has a `{}` declaration in the impl, \
                        but not in the trait",
                        tcx.sess.str_of(trait_m.ident),
                        pprust::explicit_self_to_str(&impl_m.explicit_self,
                                                     tcx.sess.intr())));
            return;
        }
        (_, &ast::SelfStatic) => {
            tcx.sess.span_err(
                impl_m_span,
                format!("method `{}` has a `{}` declaration in the trait, \
                        but not in the impl",
                        tcx.sess.str_of(trait_m.ident),
                        pprust::explicit_self_to_str(&trait_m.explicit_self,
                                                     tcx.sess.intr())));
            return;
        }
        _ => {
            // Let the type checker catch other errors below
        }
    }

    let num_impl_m_type_params = impl_m.generics.type_param_defs.len();
    let num_trait_m_type_params = trait_m.generics.type_param_defs.len();
    if num_impl_m_type_params != num_trait_m_type_params {
        tcx.sess.span_err(
            impl_m_span,
            format!("method `{}` has {} type parameter(s), but its trait \
                    declaration has {} type parameter(s)",
                    tcx.sess.str_of(trait_m.ident),
                    num_impl_m_type_params,
                    num_trait_m_type_params));
        return;
    }

    if impl_m.fty.sig.inputs.len() != trait_m.fty.sig.inputs.len() {
        tcx.sess.span_err(
            impl_m_span,
            format!("method `{}` has {} parameter{} \
                  but the declaration in trait `{}` has {}",
                 tcx.sess.str_of(trait_m.ident),
                 impl_m.fty.sig.inputs.len(),
                 if impl_m.fty.sig.inputs.len() == 1 { "" } else { "s" },
                 ty::item_path_str(tcx, trait_m.def_id),
                 trait_m.fty.sig.inputs.len()));
        return;
    }

    for (i, trait_param_def) in trait_m.generics.type_param_defs.iter().enumerate() {
        // For each of the corresponding impl ty param's bounds...
        let impl_param_def = &impl_m.generics.type_param_defs[i];

        // Check that the impl does not require any builtin-bounds
        // that the trait does not guarantee:
        let extra_bounds =
            impl_param_def.bounds.builtin_bounds -
            trait_param_def.bounds.builtin_bounds;
        if !extra_bounds.is_empty() {
           tcx.sess.span_err(
               impl_m_span,
               format!("in method `{}`, \
                       type parameter {} requires `{}`, \
                       which is not required by \
                       the corresponding type parameter \
                       in the trait declaration",
                       tcx.sess.str_of(trait_m.ident),
                       i,
                       extra_bounds.user_string(tcx)));
           return;
        }

        // FIXME(#2687)---we should be checking that the bounds of the
        // trait imply the bounds of the subtype, but it appears we
        // are...not checking this.
        if impl_param_def.bounds.trait_bounds.len() !=
            trait_param_def.bounds.trait_bounds.len()
        {
            tcx.sess.span_err(
                impl_m_span,
                format!("in method `{}`, \
                        type parameter {} has {} trait bound(s), but the \
                        corresponding type parameter in \
                        the trait declaration has {} trait bound(s)",
                        tcx.sess.str_of(trait_m.ident),
                        i, impl_param_def.bounds.trait_bounds.len(),
                        trait_param_def.bounds.trait_bounds.len()));
            return;
        }
    }

    // Create a substitution that maps the type parameters on the impl
    // to themselves and which replace any references to bound regions
    // in the self type with free regions.  So, for example, if the
    // impl type is "&'a str", then this would replace the self
    // type with a free region `self`.
    let dummy_impl_tps: ~[ty::t] =
        impl_generics.type_param_defs.iter().enumerate().
        map(|(i,t)| ty::mk_param(tcx, i, t.def_id)).
        collect();
    let dummy_method_tps: ~[ty::t] =
        impl_m.generics.type_param_defs.iter().enumerate().
        map(|(i,t)| ty::mk_param(tcx, i + impl_tps, t.def_id)).
        collect();
    let dummy_impl_regions: OptVec<ty::Region> =
        impl_generics.region_param_defs.iter().
        map(|l| ty::ReFree(ty::FreeRegion {
                scope_id: impl_m_body_id,
                bound_region: ty::BrNamed(l.def_id, l.ident)})).
        collect();
    let dummy_substs = ty::substs {
        tps: vec::append(dummy_impl_tps, dummy_method_tps),
        regions: ty::NonerasedRegions(dummy_impl_regions),
        self_ty: None };

    // We are going to create a synthetic fn type that includes
    // both the method's self argument and its normal arguments.
    // So a method like `fn(&self, a: uint)` would be converted
    // into a function `fn(self: &T, a: uint)`.
    let mut trait_fn_args = ~[];
    let mut impl_fn_args = ~[];

    // For both the trait and the impl, create an argument to
    // represent the self argument (unless this is a static method).
    // This argument will have the *transformed* self type.
    for &t in trait_m.transformed_self_ty.iter() {
        trait_fn_args.push(t);
    }
    for &t in impl_m.transformed_self_ty.iter() {
        impl_fn_args.push(t);
    }

    // Add in the normal arguments.
    trait_fn_args.push_all(trait_m.fty.sig.inputs);
    impl_fn_args.push_all(impl_m.fty.sig.inputs);

    // Create a bare fn type for trait/impl that includes self argument
    let trait_fty =
        ty::mk_bare_fn(tcx,
                       ty::BareFnTy {
                            purity: trait_m.fty.purity,
                            abis: trait_m.fty.abis,
                            sig: ty::FnSig {
                                binder_id: trait_m.fty.sig.binder_id,
                                inputs: trait_fn_args,
                                output: trait_m.fty.sig.output,
                                variadic: false
                            }
                        });
    let impl_fty =
        ty::mk_bare_fn(tcx,
                       ty::BareFnTy {
                            purity: impl_m.fty.purity,
                            abis: impl_m.fty.abis,
                            sig: ty::FnSig {
                                binder_id: impl_m.fty.sig.binder_id,
                                inputs: impl_fn_args,
                                output: impl_m.fty.sig.output,
                                variadic: false
                            }
                        });

    // Perform substitutions so that the trait/impl methods are expressed
    // in terms of the same set of type/region parameters:
    // - replace trait type parameters with those from `trait_substs`,
    //   except with any reference to bound self replaced with `dummy_self_r`
    // - replace method parameters on the trait with fresh, dummy parameters
    //   that correspond to the parameters we will find on the impl
    // - replace self region with a fresh, dummy region
    let impl_fty = {
        debug!("impl_fty (pre-subst): {}", ppaux::ty_to_str(tcx, impl_fty));
        impl_fty.subst(tcx, &dummy_substs)
    };
    debug!("impl_fty (post-subst): {}", ppaux::ty_to_str(tcx, impl_fty));
    let trait_fty = {
        let substs { regions: trait_regions,
                     tps: trait_tps,
                     self_ty: self_ty } = trait_substs.subst(tcx, &dummy_substs);
        let substs = substs {
            regions: trait_regions,
            tps: vec::append(trait_tps, dummy_method_tps),
            self_ty: self_ty,
        };
        debug!("trait_fty (pre-subst): {} substs={}",
               trait_fty.repr(tcx), substs.repr(tcx));
        trait_fty.subst(tcx, &substs)
    };
    debug!("trait_fty (post-subst): {}", trait_fty.repr(tcx));

    match infer::mk_subty(infcx, false, infer::MethodCompatCheck(impl_m_span),
                          impl_fty, trait_fty) {
        result::Ok(()) => {}
        result::Err(ref terr) => {
            tcx.sess.span_err(
                impl_m_span,
                format!("method `{}` has an incompatible type: {}",
                        tcx.sess.str_of(trait_m.ident),
                        ty::type_err_to_str(tcx, terr)));
            ty::note_and_explain_type_err(tcx, terr);
        }
    }
}

impl AstConv for FnCtxt {
    fn tcx(&self) -> ty::ctxt { self.ccx.tcx }

    fn get_item_ty(&self, id: ast::DefId) -> ty::ty_param_bounds_and_ty {
        ty::lookup_item_type(self.tcx(), id)
    }

    fn get_trait_def(&self, id: ast::DefId) -> @ty::TraitDef {
        ty::lookup_trait_def(self.tcx(), id)
    }

    fn ty_infer(&self, _span: Span) -> ty::t {
        self.infcx().next_ty_var()
    }
}

impl FnCtxt {
    pub fn infcx(&self) -> @infer::InferCtxt {
        self.inh.infcx
    }

    pub fn err_count_since_creation(&self) -> uint {
        self.ccx.tcx.sess.err_count() - self.err_count_on_creation
    }

    pub fn vtable_context<'a>(&'a self) -> VtableContext<'a> {
        VtableContext {
            infcx: self.infcx(),
            param_env: &self.inh.param_env
        }
    }
}

impl RegionScope for @infer::InferCtxt {
    fn anon_regions(&self,
                    span: Span,
                    count: uint) -> Result<~[ty::Region], ()> {
        Ok(vec::from_fn(
                count,
                |_| self.next_region_var(infer::MiscVariable(span))))
    }
}

impl FnCtxt {
    pub fn tag(&self) -> ~str {
        unsafe {
            format!("{}", self as *FnCtxt)
        }
    }

    pub fn local_ty(&self, span: Span, nid: ast::NodeId) -> ty::t {
        let locals = self.inh.locals.borrow();
        match locals.get().find(&nid) {
            Some(&t) => t,
            None => {
                self.tcx().sess.span_bug(
                    span,
                    format!("No type for local variable {:?}", nid));
            }
        }
    }

    pub fn block_region(&self) -> ty::Region {
        ty::ReScope(self.region_lb.get())
    }

    #[inline]
    pub fn write_ty(&self, node_id: ast::NodeId, ty: ty::t) {
        debug!("write_ty({}, {}) in fcx {}",
               node_id, ppaux::ty_to_str(self.tcx(), ty), self.tag());
        let mut node_types = self.inh.node_types.borrow_mut();
        node_types.get().insert(node_id, ty);
    }

    pub fn write_substs(&self, node_id: ast::NodeId, substs: ty::substs) {
        if !ty::substs_is_noop(&substs) {
            debug!("write_substs({}, {}) in fcx {}",
                   node_id,
                   ty::substs_to_str(self.tcx(), &substs),
                   self.tag());

            let mut node_type_substs = self.inh.node_type_substs.borrow_mut();
            node_type_substs.get().insert(node_id, substs);
        }
    }

    pub fn write_ty_substs(&self,
                           node_id: ast::NodeId,
                           ty: ty::t,
                           substs: ty::substs) {
        let ty = ty::subst(self.tcx(), &substs, ty);
        self.write_ty(node_id, ty);
        self.write_substs(node_id, substs);
    }

    pub fn write_autoderef_adjustment(&self,
                                      node_id: ast::NodeId,
                                      derefs: uint) {
        if derefs == 0 { return; }
        self.write_adjustment(
            node_id,
            @ty::AutoDerefRef(ty::AutoDerefRef {
                autoderefs: derefs,
                autoref: None })
        );
    }

    pub fn write_adjustment(&self,
                            node_id: ast::NodeId,
                            adj: @ty::AutoAdjustment) {
        debug!("write_adjustment(node_id={:?}, adj={:?})", node_id, adj);
        let mut adjustments = self.inh.adjustments.borrow_mut();
        adjustments.get().insert(node_id, adj);
    }

    pub fn write_nil(&self, node_id: ast::NodeId) {
        self.write_ty(node_id, ty::mk_nil());
    }
    pub fn write_bot(&self, node_id: ast::NodeId) {
        self.write_ty(node_id, ty::mk_bot());
    }
    pub fn write_error(&self, node_id: ast::NodeId) {
        self.write_ty(node_id, ty::mk_err());
    }

    pub fn to_ty(&self, ast_t: &ast::Ty) -> ty::t {
        ast_ty_to_ty(self, &self.infcx(), ast_t)
    }

    pub fn pat_to_str(&self, pat: &ast::Pat) -> ~str {
        pat.repr(self.tcx())
    }

    pub fn expr_ty(&self, ex: &ast::Expr) -> ty::t {
        let node_types = self.inh.node_types.borrow();
        match node_types.get().find(&ex.id) {
            Some(&t) => t,
            None => {
                self.tcx().sess.bug(format!("no type for expr in fcx {}",
                                            self.tag()));
            }
        }
    }

    pub fn node_ty(&self, id: ast::NodeId) -> ty::t {
        let node_types = self.inh.node_types.borrow();
        match node_types.get().find(&id) {
            Some(&t) => t,
            None => {
                self.tcx().sess.bug(
                    format!("no type for node {}: {} in fcx {}",
                            id, ast_map::node_id_to_str(
                                self.tcx().items, id,
                                token::get_ident_interner()),
                            self.tag()));
            }
        }
    }

    pub fn node_ty_substs(&self, id: ast::NodeId) -> ty::substs {
        let mut node_type_substs = self.inh.node_type_substs.borrow_mut();
        match node_type_substs.get().find(&id) {
            Some(ts) => (*ts).clone(),
            None => {
                self.tcx().sess.bug(
                    format!("no type substs for node {}: {} in fcx {}",
                            id, ast_map::node_id_to_str(self.tcx().items, id,
                                                        token::get_ident_interner()),
                            self.tag()));
            }
        }
    }

    pub fn opt_node_ty_substs(&self,
                              id: ast::NodeId,
                              f: |&ty::substs| -> bool)
                              -> bool {
        let node_type_substs = self.inh.node_type_substs.borrow();
        match node_type_substs.get().find(&id) {
            Some(s) => f(s),
            None => true
        }
    }

    pub fn mk_subty(&self,
                    a_is_expected: bool,
                    origin: infer::TypeOrigin,
                    sub: ty::t,
                    sup: ty::t)
                    -> Result<(), ty::type_err> {
        infer::mk_subty(self.infcx(), a_is_expected, origin, sub, sup)
    }

    pub fn can_mk_subty(&self, sub: ty::t, sup: ty::t)
                        -> Result<(), ty::type_err> {
        infer::can_mk_subty(self.infcx(), sub, sup)
    }

    pub fn mk_assignty(&self,
                       expr: &ast::Expr,
                       sub: ty::t,
                       sup: ty::t)
                       -> Result<(), ty::type_err> {
        match infer::mk_coercety(self.infcx(),
                                 false,
                                 infer::ExprAssignable(expr.span),
                                 sub,
                                 sup) {
            Ok(None) => result::Ok(()),
            Err(ref e) => result::Err((*e)),
            Ok(Some(adjustment)) => {
                self.write_adjustment(expr.id, adjustment);
                Ok(())
            }
        }
    }

    pub fn can_mk_assignty(&self, sub: ty::t, sup: ty::t)
                           -> Result<(), ty::type_err> {
        infer::can_mk_coercety(self.infcx(), sub, sup)
    }

    pub fn mk_eqty(&self,
                   a_is_expected: bool,
                   origin: infer::TypeOrigin,
                   sub: ty::t,
                   sup: ty::t)
                   -> Result<(), ty::type_err> {
        infer::mk_eqty(self.infcx(), a_is_expected, origin, sub, sup)
    }

    pub fn mk_subr(&self,
                   a_is_expected: bool,
                   origin: infer::SubregionOrigin,
                   sub: ty::Region,
                   sup: ty::Region) {
        infer::mk_subr(self.infcx(), a_is_expected, origin, sub, sup)
    }

    pub fn with_region_lb<R>(&self, lb: ast::NodeId, f: || -> R) -> R {
        let old_region_lb = self.region_lb.get();
        self.region_lb.set(lb);
        let v = f();
        self.region_lb.set(old_region_lb);
        v
    }

    pub fn type_error_message(&self,
                              sp: Span,
                              mk_msg: |~str| -> ~str,
                              actual_ty: ty::t,
                              err: Option<&ty::type_err>) {
        self.infcx().type_error_message(sp, mk_msg, actual_ty, err);
    }

    pub fn report_mismatched_return_types(&self,
                                          sp: Span,
                                          e: ty::t,
                                          a: ty::t,
                                          err: &ty::type_err) {
        // Derived error
        if ty::type_is_error(e) || ty::type_is_error(a) {
            return;
        }
        self.infcx().report_mismatched_types(sp, e, a, err)
    }

    pub fn report_mismatched_types(&self,
                                   sp: Span,
                                   e: ty::t,
                                   a: ty::t,
                                   err: &ty::type_err) {
        self.infcx().report_mismatched_types(sp, e, a, err)
    }
}

pub fn do_autoderef(fcx: @FnCtxt, sp: Span, t: ty::t) -> (ty::t, uint) {
    /*!
     *
     * Autoderefs the type `t` as many times as possible, returning
     * a new type and a counter for how many times the type was
     * deref'd.  If the counter is non-zero, the receiver is responsible
     * for inserting an AutoAdjustment record into `tcx.adjustments`
     * so that trans/borrowck/etc know about this autoderef. */

    let mut t1 = t;
    let mut enum_dids = ~[];
    let mut autoderefs = 0;
    loop {
        let sty = structure_of(fcx, sp, t1);

        // Some extra checks to detect weird cycles and so forth:
        match *sty {
            ty::ty_box(inner) | ty::ty_uniq(inner) => {
                match ty::get(t1).sty {
                    ty::ty_infer(ty::TyVar(v1)) => {
                        ty::occurs_check(fcx.ccx.tcx, sp, v1,
                                         ty::mk_box(fcx.ccx.tcx, inner));
                    }
                    _ => ()
                }
            }
            ty::ty_rptr(_, inner) => {
                match ty::get(t1).sty {
                    ty::ty_infer(ty::TyVar(v1)) => {
                        ty::occurs_check(fcx.ccx.tcx, sp, v1,
                                         ty::mk_box(fcx.ccx.tcx, inner.ty));
                    }
                    _ => ()
                }
            }
            ty::ty_enum(ref did, _) => {
                // Watch out for a type like `enum t = @t`.  Such a
                // type would otherwise infinitely auto-deref.  Only
                // autoderef loops during typeck (basically, this one
                // and the loops in typeck::check::method) need to be
                // concerned with this, as an error will be reported
                // on the enum definition as well because the enum is
                // not instantiable.
                if enum_dids.contains(did) {
                    return (t1, autoderefs);
                }
                enum_dids.push(*did);
            }
            _ => { /*ok*/ }
        }

        // Otherwise, deref if type is derefable:
        match ty::deref_sty(sty, false) {
            None => {
                return (t1, autoderefs);
            }
            Some(mt) => {
                autoderefs += 1;
                t1 = mt.ty
            }
        }
    };
}

// AST fragment checking
pub fn check_lit(fcx: @FnCtxt, lit: &ast::Lit) -> ty::t {
    let tcx = fcx.ccx.tcx;

    match lit.node {
        ast::LitStr(..) => ty::mk_str(tcx, ty::vstore_slice(ty::ReStatic)),
        ast::LitBinary(..) => {
            ty::mk_vec(tcx, ty::mt{ ty: ty::mk_u8(), mutbl: ast::MutImmutable },
                       ty::vstore_slice(ty::ReStatic))
        }
        ast::LitChar(_) => ty::mk_char(),
        ast::LitInt(_, t) => ty::mk_mach_int(t),
        ast::LitUint(_, t) => ty::mk_mach_uint(t),
        ast::LitIntUnsuffixed(_) => {
            // An unsuffixed integer literal could have any integral type,
            // so we create an integral type variable for it.
            ty::mk_int_var(tcx, fcx.infcx().next_int_var_id())
        }
        ast::LitFloat(_, t) => ty::mk_mach_float(t),
        ast::LitFloatUnsuffixed(_) => {
            // An unsuffixed floating point literal could have any floating point
            // type, so we create a floating point type variable for it.
            ty::mk_float_var(tcx, fcx.infcx().next_float_var_id())
        }
        ast::LitNil => ty::mk_nil(),
        ast::LitBool(_) => ty::mk_bool()
    }
}

pub fn valid_range_bounds(ccx: @CrateCtxt,
                          from: &ast::Expr,
                          to: &ast::Expr)
                       -> Option<bool> {
    match const_eval::compare_lit_exprs(ccx.tcx, from, to) {
        Some(val) => Some(val <= 0),
        None => None
    }
}

pub fn check_expr_has_type(
    fcx: @FnCtxt, expr: &ast::Expr,
    expected: ty::t) {
    check_expr_with_unifier(fcx, expr, Some(expected), || {
        demand::suptype(fcx, expr.span, expected, fcx.expr_ty(expr));
    });
}

pub fn check_expr_coercable_to_type(
    fcx: @FnCtxt, expr: &ast::Expr,
    expected: ty::t) {
    check_expr_with_unifier(fcx, expr, Some(expected), || {
        demand::coerce(fcx, expr.span, expected, expr)
    });
}

pub fn check_expr_with_hint(
    fcx: @FnCtxt, expr: &ast::Expr,
    expected: ty::t) {
    check_expr_with_unifier(fcx, expr, Some(expected), || ())
}

pub fn check_expr_with_opt_hint(
    fcx: @FnCtxt, expr: &ast::Expr,
    expected: Option<ty::t>)  {
    check_expr_with_unifier(fcx, expr, expected, || ())
}

pub fn check_expr(fcx: @FnCtxt, expr: &ast::Expr)  {
    check_expr_with_unifier(fcx, expr, None, || ())
}

// determine the `self` type, using fresh variables for all variables
// declared on the impl declaration e.g., `impl<A,B> for ~[(A,B)]`
// would return ($0, $1) where $0 and $1 are freshly instantiated type
// variables.
pub fn impl_self_ty(vcx: &VtableContext,
                    location_info: &LocationInfo, // (potential) receiver for
                                                  // this impl
                    did: ast::DefId)
                 -> ty_param_substs_and_ty {
    let tcx = vcx.tcx();

    let (n_tps, n_rps, raw_ty) = {
        let ity = ty::lookup_item_type(tcx, did);
        (ity.generics.type_param_defs.len(),
         ity.generics.region_param_defs.len(),
         ity.ty)
    };

    let rps =
        vcx.infcx.next_region_vars(
            infer::BoundRegionInTypeOrImpl(location_info.span),
            n_rps);
    let tps = vcx.infcx.next_ty_vars(n_tps);

    let substs = substs {regions: ty::NonerasedRegions(opt_vec::from(rps)),
                         self_ty: None,
                         tps: tps};
    let substd_ty = ty::subst(tcx, &substs, raw_ty);

    ty_param_substs_and_ty { substs: substs, ty: substd_ty }
}

// Only for fields! Returns <none> for methods>
// Indifferent to privacy flags
pub fn lookup_field_ty(tcx: ty::ctxt,
                       class_id: ast::DefId,
                       items: &[ty::field_ty],
                       fieldname: ast::Name,
                       substs: &ty::substs) -> Option<ty::t> {

    let o_field = items.iter().find(|f| f.name == fieldname);
    o_field.map(|f| ty::lookup_field_type(tcx, class_id, f.id, substs))
}

// Controls whether the arguments are automatically referenced. This is useful
// for overloaded binary and unary operators.
pub enum DerefArgs {
    DontDerefArgs,
    DoDerefArgs
}

// Given the provenance of a static method, returns the generics of the static
// method's container.
fn generics_of_static_method_container(type_context: ty::ctxt,
                                       provenance: ast::MethodProvenance)
                                       -> ty::Generics {
    match provenance {
        ast::FromTrait(trait_def_id) => {
            ty::lookup_trait_def(type_context, trait_def_id).generics
        }
        ast::FromImpl(impl_def_id) => {
            ty::lookup_item_type(type_context, impl_def_id).generics
        }
    }
}

// Verifies that type parameters supplied in paths are in the right
// locations.
fn check_type_parameter_positions_in_path(function_context: @FnCtxt,
                                          path: &ast::Path,
                                          def: ast::Def) {
    // We only care about checking the case in which the path has two or
    // more segments.
    if path.segments.len() < 2 {
        return
    }

    // Verify that no lifetimes or type parameters are present anywhere
    // except the final two elements of the path.
    for i in range(0, path.segments.len() - 2) {
        for lifetime in path.segments[i].lifetimes.iter() {
            function_context.tcx()
                .sess
                .span_err(lifetime.span,
                          "lifetime parameters may not \
                          appear here");
            break;
        }

        for typ in path.segments[i].types.iter() {
            function_context.tcx()
                            .sess
                            .span_err(typ.span,
                                      "type parameters may not appear here");
            break;
        }
    }

    // If there are no parameters at all, there is nothing more to do; the
    // rest of typechecking will (attempt to) infer everything.
    if path.segments
           .iter()
           .all(|s| s.lifetimes.is_empty() && s.types.is_empty()) {
        return
    }

    match def {
        // If this is a static method of a trait or implementation, then
        // ensure that the segment of the path which names the trait or
        // implementation (the penultimate segment) is annotated with the
        // right number of type parameters.
        ast::DefStaticMethod(_, provenance, _) => {
            let generics =
                generics_of_static_method_container(function_context.ccx.tcx,
                                                    provenance);
            let name = match provenance {
                ast::FromTrait(_) => "trait",
                ast::FromImpl(_) => "impl",
            };

            let trait_segment = &path.segments[path.segments.len() - 2];

            // Make sure lifetime parameterization agrees with the trait or
            // implementation type.
            let trait_region_parameter_count = generics.region_param_defs.len();
            let supplied_region_parameter_count = trait_segment.lifetimes.len();
            if trait_region_parameter_count != supplied_region_parameter_count
                && supplied_region_parameter_count != 0 {
                function_context.tcx()
                    .sess
                    .span_err(path.span,
                              format!("expected {} lifetime parameter(s), \
                                      found {} lifetime parameter(s)",
                                      trait_region_parameter_count,
                                      supplied_region_parameter_count));
            }

            // Make sure the number of type parameters supplied on the trait
            // or implementation segment equals the number of type parameters
            // on the trait or implementation definition.
            let trait_type_parameter_count = generics.type_param_defs.len();
            let supplied_type_parameter_count = trait_segment.types.len();
            if trait_type_parameter_count != supplied_type_parameter_count {
                let trait_count_suffix = if trait_type_parameter_count == 1 {
                    ""
                } else {
                    "s"
                };
                let supplied_count_suffix =
                    if supplied_type_parameter_count == 1 {
                        ""
                    } else {
                        "s"
                    };
                function_context.tcx()
                                .sess
                                .span_err(path.span,
                                          format!("the {} referenced by this \
                                                path has {} type \
                                                parameter{}, but {} type \
                                                parameter{} were supplied",
                                               name,
                                               trait_type_parameter_count,
                                               trait_count_suffix,
                                               supplied_type_parameter_count,
                                               supplied_count_suffix))
            }
        }
        _ => {
            // Verify that no lifetimes or type parameters are present on
            // the penultimate segment of the path.
            let segment = &path.segments[path.segments.len() - 2];
            for lifetime in segment.lifetimes.iter() {
                function_context.tcx()
                    .sess
                    .span_err(lifetime.span,
                              "lifetime parameters may not
                              appear here");
                break;
            }
            for typ in segment.types.iter() {
                function_context.tcx()
                                .sess
                                .span_err(typ.span,
                                          "type parameters may not appear \
                                           here");
                break;
            }
        }
    }
}

/// Invariant:
/// If an expression has any sub-expressions that result in a type error,
/// inspecting that expression's type with `ty::type_is_error` will return
/// true. Likewise, if an expression is known to diverge, inspecting its
/// type with `ty::type_is_bot` will return true (n.b.: since Rust is
/// strict, _|_ can appear in the type of an expression that does not,
/// itself, diverge: for example, fn() -> _|_.)
/// Note that inspecting a type's structure *directly* may expose the fact
/// that there are actually multiple representations for both `ty_err` and
/// `ty_bot`, so avoid that when err and bot need to be handled differently.
pub fn check_expr_with_unifier(fcx: @FnCtxt,
                               expr: &ast::Expr,
                               expected: Option<ty::t>,
                               unifier: ||) {
    debug!(">> typechecking");

    fn check_method_argument_types(
        fcx: @FnCtxt,
        sp: Span,
        method_fn_ty: ty::t,
        callee_expr: &ast::Expr,
        args: &[@ast::Expr],
        sugar: ast::CallSugar,
        deref_args: DerefArgs) -> ty::t
    {
        if ty::type_is_error(method_fn_ty) {
            let err_inputs = err_args(args.len());
            check_argument_types(fcx, sp, err_inputs, callee_expr,
                                 args, sugar, deref_args, false);
            method_fn_ty
        } else {
            match ty::get(method_fn_ty).sty {
                ty::ty_bare_fn(ref fty) => {
                    check_argument_types(fcx, sp, fty.sig.inputs, callee_expr,
                                         args, sugar, deref_args, fty.sig.variadic);
                    fty.sig.output
                }
                _ => {
                    fcx.tcx().sess.span_bug(
                        sp,
                        format!("Method without bare fn type"));
                }
            }
        }
    }

    fn check_argument_types(fcx: @FnCtxt,
                            sp: Span,
                            fn_inputs: &[ty::t],
                            callee_expr: &ast::Expr,
                            args: &[@ast::Expr],
                            sugar: ast::CallSugar,
                            deref_args: DerefArgs,
                            variadic: bool) {
        /*!
         *
         * Generic function that factors out common logic from
         * function calls, method calls and overloaded operators.
         */

        let tcx = fcx.ccx.tcx;

        // Grab the argument types, supplying fresh type variables
        // if the wrong number of arguments were supplied
        let supplied_arg_count = args.len();
        let expected_arg_count = fn_inputs.len();
        let formal_tys = if expected_arg_count == supplied_arg_count {
            fn_inputs.map(|a| *a)
        } else if variadic {
            if supplied_arg_count >= expected_arg_count {
                fn_inputs.map(|a| *a)
            } else {
                let msg = format!(
                    "this function takes at least {} parameter{} \
                     but {} parameter{} supplied",
                     expected_arg_count,
                     if expected_arg_count == 1 {""} else {"s"},
                     supplied_arg_count,
                     if supplied_arg_count == 1 {" was"} else {"s were"});

                tcx.sess.span_err(sp, msg);

                err_args(supplied_arg_count)
            }
        } else {
            let suffix = match sugar {
                ast::NoSugar => "",
                ast::DoSugar => " (including the closure passed by \
                                 the `do` keyword)",
                ast::ForSugar => " (including the closure passed by \
                                  the `for` keyword)"
            };
            let msg = format!(
                "this function takes {} parameter{} \
                 but {} parameter{} supplied{}",
                 expected_arg_count, if expected_arg_count == 1 {""} else {"s"},
                 supplied_arg_count,
                 if supplied_arg_count == 1 {" was"} else {"s were"},
                 suffix);

            tcx.sess.span_err(sp, msg);

            err_args(supplied_arg_count)
        };

        debug!("check_argument_types: formal_tys={:?}",
               formal_tys.map(|t| fcx.infcx().ty_to_str(*t)));

        // Check the arguments.
        // We do this in a pretty awful way: first we typecheck any arguments
        // that are not anonymous functions, then we typecheck the anonymous
        // functions. This is so that we have more information about the types
        // of arguments when we typecheck the functions. This isn't really the
        // right way to do this.
        let xs = [false, true];
        for check_blocks in xs.iter() {
            let check_blocks = *check_blocks;
            debug!("check_blocks={}", check_blocks);

            // More awful hacks: before we check the blocks, try to do
            // an "opportunistic" vtable resolution of any trait
            // bounds on the call.
            if check_blocks {
                vtable::early_resolve_expr(callee_expr, fcx, true);
            }

            // For variadic functions, we don't have a declared type for all of
            // the arguments hence we only do our usual type checking with
            // the arguments who's types we do know.
            let t = if variadic {
                expected_arg_count
            } else {
                supplied_arg_count
            };
            for (i, arg) in args.iter().take(t).enumerate() {
                let is_block = match arg.node {
                    ast::ExprFnBlock(..) |
                    ast::ExprProc(..) |
                    ast::ExprDoBody(..) => true,
                    _ => false
                };

                if is_block == check_blocks {
                    debug!("checking the argument");
                    let mut formal_ty = formal_tys[i];

                    match deref_args {
                        DoDerefArgs => {
                            match ty::get(formal_ty).sty {
                                ty::ty_rptr(_, mt) => formal_ty = mt.ty,
                                ty::ty_err => (),
                                _ => {
                                    fcx.ccx.tcx.sess.span_bug(arg.span, "no ref");
                                }
                            }
                        }
                        DontDerefArgs => {}
                    }

                    check_expr_coercable_to_type(fcx, *arg, formal_ty);

                }
            }
        }

        // We also need to make sure we at least write the ty of the other
        // arguments which we skipped above.
        if variadic {
            for arg in args.iter().skip(expected_arg_count) {
                check_expr(fcx, *arg);

                // There are a few types which get autopromoted when passed via varargs
                // in C but we just error out instead and require explicit casts.
                let arg_ty = structurally_resolved_type(fcx, arg.span, fcx.expr_ty(*arg));
                match ty::get(arg_ty).sty {
                    ty::ty_float(ast::TyF32) => {
                        fcx.type_error_message(arg.span,
                                |t| format!("can't pass an {} to variadic function, \
                                             cast to c_double", t), arg_ty, None);
                    }
                    ty::ty_int(ast::TyI8) | ty::ty_int(ast::TyI16) | ty::ty_bool => {
                        fcx.type_error_message(arg.span,
                                |t| format!("can't pass {} to variadic function, cast to c_int",
                                            t), arg_ty, None);
                    }
                    ty::ty_uint(ast::TyU8) | ty::ty_uint(ast::TyU16) => {
                        fcx.type_error_message(arg.span,
                                |t| format!("can't pass {} to variadic function, cast to c_uint",
                                            t), arg_ty, None);
                    }
                    _ => {}
                }
            }
        }
    }

    fn err_args(len: uint) -> ~[ty::t] {
        vec::from_fn(len, |_| ty::mk_err())
    }

    // A generic function for checking assignment expressions
    fn check_assignment(fcx: @FnCtxt,
                        lhs: &ast::Expr,
                        rhs: &ast::Expr,
                        id: ast::NodeId) {
        check_expr(fcx, lhs);
        let lhs_type = fcx.expr_ty(lhs);
        check_expr_has_type(fcx, rhs, lhs_type);
        fcx.write_ty(id, ty::mk_nil());
        // The callee checks for bot / err, we don't need to
    }

    fn write_call(fcx: @FnCtxt,
                  call_expr: &ast::Expr,
                  output: ty::t,
                  sugar: ast::CallSugar) {
        let ret_ty = match sugar {
            ast::ForSugar => {
                match ty::get(output).sty {
                    ty::ty_bool => {}
                    _ => fcx.type_error_message(call_expr.span, |actual| {
                            format!("expected `for` closure to return `bool`, \
                                  but found `{}`", actual) },
                            output, None)
                }
                ty::mk_nil()
            }
            _ => output
        };
        fcx.write_ty(call_expr.id, ret_ty);
    }

    // A generic function for doing all of the checking for call expressions
    fn check_call(fcx: @FnCtxt,
                  callee_id: ast::NodeId,
                  call_expr: &ast::Expr,
                  f: &ast::Expr,
                  args: &[@ast::Expr],
                  sugar: ast::CallSugar) {
        // Index expressions need to be handled separately, to inform them
        // that they appear in call position.
        check_expr(fcx, f);

        // Store the type of `f` as the type of the callee
        let fn_ty = fcx.expr_ty(f);

        // FIXME(#6273) should write callee type AFTER regions have
        // been subst'd.  However, it is awkward to deal with this
        // now. Best thing would I think be to just have a separate
        // "callee table" that contains the FnSig and not a general
        // purpose ty::t
        fcx.write_ty(callee_id, fn_ty);

        // Extract the function signature from `in_fty`.
        let fn_sty = structure_of(fcx, f.span, fn_ty);

        // This is the "default" function signature, used in case of error.
        // In that case, we check each argument against "error" in order to
        // set up all the node type bindings.
        let error_fn_sig = FnSig {
            binder_id: ast::CRATE_NODE_ID,
            inputs: err_args(args.len()),
            output: ty::mk_err(),
            variadic: false
        };

        let fn_sig = match *fn_sty {
            ty::ty_bare_fn(ty::BareFnTy {sig: ref sig, ..}) |
            ty::ty_closure(ty::ClosureTy {sig: ref sig, ..}) => sig,
            _ => {
                fcx.type_error_message(call_expr.span, |actual| {
                    format!("expected function but \
                          found `{}`", actual) }, fn_ty, None);
                &error_fn_sig
            }
        };

        // Replace any bound regions that appear in the function
        // signature with region variables
        let (_, _, fn_sig) =
            replace_bound_regions_in_fn_sig(fcx.tcx(),
                                            None,
                                            fn_sig,
                                            |br| fcx.infcx()
                                                    .next_region_var(
                    infer::BoundRegionInFnCall(call_expr.span, br)));

        // Call the generic checker.
        check_argument_types(fcx, call_expr.span, fn_sig.inputs, f,
                             args, sugar, DontDerefArgs, fn_sig.variadic);

        write_call(fcx, call_expr, fn_sig.output, sugar);
    }

    // Checks a method call.
    fn check_method_call(fcx: @FnCtxt,
                         callee_id: ast::NodeId,
                         expr: &ast::Expr,
                         rcvr: &ast::Expr,
                         method_name: ast::Ident,
                         args: &[@ast::Expr],
                         tps: &[ast::P<ast::Ty>],
                         sugar: ast::CallSugar) {
        check_expr(fcx, rcvr);

        // no need to check for bot/err -- callee does that
        let expr_t = structurally_resolved_type(fcx,
                                                expr.span,
                                                fcx.expr_ty(rcvr));

        let tps = tps.map(|&ast_ty| fcx.to_ty(ast_ty));
        match method::lookup(fcx,
                             expr,
                             rcvr,
                             callee_id,
                             method_name.name,
                             expr_t,
                             tps,
                             DontDerefArgs,
                             CheckTraitsAndInherentMethods,
                             AutoderefReceiver) {
            Some(ref entry) => {
                let method_map = fcx.inh.method_map;
                let mut method_map = method_map.borrow_mut();
                method_map.get().insert(expr.id, (*entry));
            }
            None => {
                debug!("(checking method call) failing expr is {}", expr.id);

                fcx.type_error_message(expr.span,
                  |actual| {
                      format!("type `{}` does not implement any method in scope \
                            named `{}`",
                           actual,
                           fcx.ccx.tcx.sess.str_of(method_name))
                  },
                  expr_t,
                  None);

                // Add error type for the result
                fcx.write_error(expr.id);
                fcx.write_error(callee_id);
            }
        }

        // Call the generic checker.
        let fn_ty = fcx.node_ty(callee_id);
        let ret_ty = check_method_argument_types(fcx, expr.span,
                                                 fn_ty, expr, args, sugar,
                                                 DontDerefArgs);

        write_call(fcx, expr, ret_ty, sugar);
    }

    // A generic function for checking the then and else in an if
    // or if-check
    fn check_then_else(fcx: @FnCtxt,
                       cond_expr: &ast::Expr,
                       then_blk: &ast::Block,
                       opt_else_expr: Option<@ast::Expr>,
                       id: ast::NodeId,
                       sp: Span,
                       expected: Option<ty::t>) {
        check_expr_has_type(fcx, cond_expr, ty::mk_bool());

        let branches_ty = match opt_else_expr {
            Some(else_expr) => {
                check_block_with_expected(fcx, then_blk, expected);
                let then_ty = fcx.node_ty(then_blk.id);
                check_expr_with_opt_hint(fcx, else_expr, expected);
                let else_ty = fcx.expr_ty(else_expr);
                infer::common_supertype(fcx.infcx(),
                                        infer::IfExpression(sp),
                                        true,
                                        then_ty,
                                        else_ty)
            }
            None => {
                check_block_no_value(fcx, then_blk);
                ty::mk_nil()
            }
        };

        let cond_ty = fcx.expr_ty(cond_expr);
        let if_ty = if ty::type_is_error(cond_ty) {
            ty::mk_err()
        } else if ty::type_is_bot(cond_ty) {
            ty::mk_bot()
        } else {
            branches_ty
        };

        fcx.write_ty(id, if_ty);
    }

    fn lookup_op_method(fcx: @FnCtxt,
                        callee_id: ast::NodeId,
                        op_ex: &ast::Expr,
                        self_ex: &ast::Expr,
                        self_t: ty::t,
                        opname: ast::Name,
                        args: &[@ast::Expr],
                        deref_args: DerefArgs,
                        autoderef_receiver: AutoderefReceiverFlag,
                        unbound_method: ||,
                        _expected_result: Option<ty::t>
                       )
                     -> ty::t {
        match method::lookup(fcx, op_ex, self_ex,
                             callee_id, opname, self_t, [],
                             deref_args, CheckTraitsOnly, autoderef_receiver) {
            Some(ref origin) => {
                let method_ty = fcx.node_ty(callee_id);
                let method_map = fcx.inh.method_map;
                {
                    let mut method_map = method_map.borrow_mut();
                    method_map.get().insert(op_ex.id, *origin);
                }
                check_method_argument_types(fcx, op_ex.span,
                                            method_ty, op_ex, args,
                                            ast::NoSugar, deref_args)
            }
            _ => {
                unbound_method();
                // Check the args anyway
                // so we get all the error messages
                let expected_ty = ty::mk_err();
                check_method_argument_types(fcx, op_ex.span,
                                            expected_ty, op_ex, args,
                                            ast::NoSugar, deref_args);
                ty::mk_err()
            }
        }
    }

    // could be either a expr_binop or an expr_assign_binop
    fn check_binop(fcx: @FnCtxt,
                   callee_id: ast::NodeId,
                   expr: &ast::Expr,
                   op: ast::BinOp,
                   lhs: &ast::Expr,
                   rhs: @ast::Expr,
                   // Used only in the error case
                   expected_result: Option<ty::t>,
                   is_binop_assignment: IsBinopAssignment
                  ) {
        let tcx = fcx.ccx.tcx;

        check_expr(fcx, lhs);
        // Callee does bot / err checking
        let lhs_t = structurally_resolved_type(fcx, lhs.span,
                                               fcx.expr_ty(lhs));

        if ty::type_is_integral(lhs_t) && ast_util::is_shift_binop(op) {
            // Shift is a special case: rhs can be any integral type
            check_expr(fcx, rhs);
            let rhs_t = fcx.expr_ty(rhs);
            require_integral(fcx, rhs.span, rhs_t);
            fcx.write_ty(expr.id, lhs_t);
            return;
        }

        if ty::is_binopable(tcx, lhs_t, op) {
            let tvar = fcx.infcx().next_ty_var();
            demand::suptype(fcx, expr.span, tvar, lhs_t);
            check_expr_has_type(fcx, rhs, tvar);

            let result_t = match op {
                ast::BiEq | ast::BiNe | ast::BiLt | ast::BiLe | ast::BiGe |
                ast::BiGt => {
                    ty::mk_bool()
                }
                _ => {
                    lhs_t
                }
            };

            fcx.write_ty(expr.id, result_t);
            return;
        }

        if op == ast::BiOr || op == ast::BiAnd {
            // This is an error; one of the operands must have the wrong
            // type
            fcx.write_error(expr.id);
            fcx.write_error(rhs.id);
            fcx.type_error_message(expr.span, |actual| {
                format!("binary operation `{}` cannot be applied \
                      to type `{}`",
                     ast_util::binop_to_str(op), actual)},
                                   lhs_t, None)

        }

        // Check for overloaded operators if not an assignment.
        let result_t;
        if is_binop_assignment == SimpleBinop {
            result_t = check_user_binop(fcx,
                                        callee_id,
                                        expr,
                                        lhs,
                                        lhs_t,
                                        op,
                                        rhs,
                                        expected_result);
        } else {
            fcx.type_error_message(expr.span,
                                   |actual| {
                                        format!("binary assignment operation \
                                                `{}=` cannot be applied to type `{}`",
                                                ast_util::binop_to_str(op),
                                                actual)
                                   },
                                   lhs_t,
                                   None);
            check_expr(fcx, rhs);
            result_t = ty::mk_err();
        }

        fcx.write_ty(expr.id, result_t);
        if ty::type_is_error(result_t) {
            fcx.write_ty(rhs.id, result_t);
        }
    }

    fn check_user_binop(fcx: @FnCtxt,
                        callee_id: ast::NodeId,
                        ex: &ast::Expr,
                        lhs_expr: &ast::Expr,
                        lhs_resolved_t: ty::t,
                        op: ast::BinOp,
                        rhs: @ast::Expr,
                       expected_result: Option<ty::t>) -> ty::t {
        let tcx = fcx.ccx.tcx;
        match ast_util::binop_to_method_name(op) {
            Some(ref name) => {
                let if_op_unbound = || {
                    fcx.type_error_message(ex.span, |actual| {
                        format!("binary operation `{}` cannot be applied \
                              to type `{}`",
                             ast_util::binop_to_str(op), actual)},
                            lhs_resolved_t, None)
                };
                return lookup_op_method(fcx, callee_id, ex, lhs_expr, lhs_resolved_t,
                                       token::intern(*name),
                                       &[rhs], DoDerefArgs, DontAutoderefReceiver, if_op_unbound,
                                       expected_result);
            }
            None => ()
        };
        check_expr(fcx, rhs);

        // If the or operator is used it might be that the user forgot to
        // supply the do keyword.  Let's be more helpful in that situation.
        if op == ast::BiOr {
            match ty::get(lhs_resolved_t).sty {
                ty::ty_bare_fn(_) | ty::ty_closure(_) => {
                    tcx.sess.span_note(
                        ex.span, "did you forget the `do` keyword for the call?");
                }
                _ => ()
            }
        }

        ty::mk_err()
    }

    fn check_user_unop(fcx: @FnCtxt,
                       callee_id: ast::NodeId,
                       op_str: &str,
                       mname: &str,
                       ex: &ast::Expr,
                       rhs_expr: &ast::Expr,
                       rhs_t: ty::t,
                       expected_t: Option<ty::t>)
                    -> ty::t {
       lookup_op_method(
            fcx, callee_id, ex, rhs_expr, rhs_t,
            token::intern(mname), &[],
            DoDerefArgs, DontAutoderefReceiver,
            || {
                fcx.type_error_message(ex.span, |actual| {
                    format!("cannot apply unary operator `{}` to type `{}`",
                         op_str, actual)
                }, rhs_t, None);
            }, expected_t)
    }

    // Resolves `expected` by a single level if it is a variable and passes it
    // through the `unpack` function.  It there is no expected type or
    // resolution is not possible (e.g., no constraints yet present), just
    // returns `none`.
    fn unpack_expected<O>(
                       fcx: @FnCtxt,
                       expected: Option<ty::t>,
                       unpack: |&ty::sty| -> Option<O>)
                       -> Option<O> {
        match expected {
            Some(t) => {
                match resolve_type(fcx.infcx(), t, force_tvar) {
                    Ok(t) => unpack(&ty::get(t).sty),
                    _ => None
                }
            }
            _ => None
        }
    }

    fn check_expr_fn(fcx: @FnCtxt,
                     expr: &ast::Expr,
                     ast_sigil_opt: Option<ast::Sigil>,
                     decl: &ast::FnDecl,
                     body: ast::P<ast::Block>,
                     fn_kind: FnKind,
                     expected: Option<ty::t>) {
        let tcx = fcx.ccx.tcx;

        // Find the expected input/output types (if any). Substitute
        // fresh bound regions for any bound regions we find in the
        // expected types so as to avoid capture.
        //
        // Also try to pick up inferred purity and sigil, defaulting
        // to impure and block. Note that we only will use those for
        // block syntax lambdas; that is, lambdas without explicit
        // sigils.
        let expected_sty = unpack_expected(fcx,
                                           expected,
                                           |x| Some((*x).clone()));
        let error_happened = false;
        let (expected_sig,
             expected_purity,
             expected_sigil,
             expected_onceness,
             expected_bounds) = {
            match expected_sty {
                Some(ty::ty_closure(ref cenv)) => {
                    let (_, _, sig) =
                        replace_bound_regions_in_fn_sig(
                            tcx, None, &cenv.sig,
                            |_| fcx.inh.infcx.fresh_bound_region(expr.id));
                    (Some(sig), cenv.purity, cenv.sigil,
                     cenv.onceness, cenv.bounds)
                }
                _ => {
                    // Not an error! Means we're inferring the closure type
                    let mut sigil = ast::BorrowedSigil;
                    let mut onceness = ast::Many;
                    let mut bounds = ty::EmptyBuiltinBounds();
                    match expr.node {
                        ast::ExprProc(..) => {
                            sigil = ast::OwnedSigil;
                            onceness = ast::Once;
                            bounds.add(ty::BoundSend);
                        }
                        _ => ()
                    }
                    (None, ast::ImpureFn, sigil,
                     onceness, bounds)
                }
            }
        };

        // If the proto is specified, use that, otherwise select a
        // proto based on inference.
        let (sigil, purity) = match ast_sigil_opt {
            Some(p) => (p, ast::ImpureFn),
            None => (expected_sigil, expected_purity)
        };

        // construct the function type
        let fn_ty = astconv::ty_of_closure(fcx,
                                           &fcx.infcx(),
                                           expr.id,
                                           sigil,
                                           purity,
                                           expected_onceness,
                                           expected_bounds,
                                           &None,
                                           decl,
                                           expected_sig,
                                           expr.span);

        let fty_sig;
        let fty = if error_happened {
            fty_sig = FnSig {
                binder_id: ast::CRATE_NODE_ID,
                inputs: fn_ty.sig.inputs.map(|_| ty::mk_err()),
                output: ty::mk_err(),
                variadic: false
            };
            ty::mk_err()
        } else {
            let fn_ty_copy = fn_ty.clone();
            fty_sig = fn_ty.sig.clone();
            ty::mk_closure(tcx, fn_ty_copy)
        };

        debug!("check_expr_fn_with_unifier fty={}",
               fcx.infcx().ty_to_str(fty));

        fcx.write_ty(expr.id, fty);

        let (inherited_purity, id) =
            ty::determine_inherited_purity((fcx.ps.get().purity,
                                            fcx.ps.get().def),
                                           (purity, expr.id),
                                           sigil);

        check_fn(fcx.ccx, None, inherited_purity, &fty_sig,
                 decl, id, body, fn_kind, fcx.inh);
    }


    // Check field access expressions
    fn check_field(fcx: @FnCtxt,
                   expr: &ast::Expr,
                   base: &ast::Expr,
                   field: ast::Name,
                   tys: &[ast::P<ast::Ty>]) {
        let tcx = fcx.ccx.tcx;
        let bot = check_expr(fcx, base);
        let expr_t = structurally_resolved_type(fcx, expr.span,
                                                fcx.expr_ty(base));
        let (base_t, derefs) = do_autoderef(fcx, expr.span, expr_t);

        match *structure_of(fcx, expr.span, base_t) {
            ty::ty_struct(base_id, ref substs) => {
                // This is just for fields -- the same code handles
                // methods in both classes and traits

                // (1) verify that the class id actually has a field called
                // field
                debug!("class named {}", ppaux::ty_to_str(tcx, base_t));
                let cls_items = ty::lookup_struct_fields(tcx, base_id);
                match lookup_field_ty(tcx, base_id, cls_items,
                                      field, &(*substs)) {
                    Some(field_ty) => {
                        // (2) look up what field's type is, and return it
                        fcx.write_ty(expr.id, field_ty);
                        fcx.write_autoderef_adjustment(base.id, derefs);
                        return bot;
                    }
                    None => ()
                }
            }
            _ => ()
        }

        let tps : ~[ty::t] = tys.iter().map(|&ty| fcx.to_ty(ty)).collect();
        match method::lookup(fcx,
                             expr,
                             base,
                             expr.id,
                             field,
                             expr_t,
                             tps,
                             DontDerefArgs,
                             CheckTraitsAndInherentMethods,
                             AutoderefReceiver) {
            Some(_) => {
                fcx.type_error_message(
                    expr.span,
                    |actual| {
                        format!("attempted to take value of method `{}` on type `{}` \
                              (try writing an anonymous function)",
                             token::interner_get(field), actual)
                    },
                    expr_t, None);
            }

            None => {
                fcx.type_error_message(
                    expr.span,
                    |actual| {
                        format!("attempted access of field `{}` on type `{}`, \
                              but no field with that name was found",
                             token::interner_get(field), actual)
                    },
                    expr_t, None);
            }
        }

        fcx.write_error(expr.id);
    }

    fn check_struct_or_variant_fields(fcx: @FnCtxt,
                                      struct_ty: ty::t,
                                      span: Span,
                                      class_id: ast::DefId,
                                      node_id: ast::NodeId,
                                      substitutions: ty::substs,
                                      field_types: &[ty::field_ty],
                                      ast_fields: &[ast::Field],
                                      check_completeness: bool)  {
        let tcx = fcx.ccx.tcx;

        let mut class_field_map = HashMap::new();
        let mut fields_found = 0;
        for field in field_types.iter() {
            class_field_map.insert(field.name, (field.id, false));
        }

        let mut error_happened = false;

        // Typecheck each field.
        for field in ast_fields.iter() {
            let mut expected_field_type = ty::mk_err();

            let pair = class_field_map.find(&field.ident.node.name).map(|x| *x);
            match pair {
                None => {
                    fcx.type_error_message(
                      field.ident.span,
                      |actual| {
                          format!("structure `{}` has no field named `{}`",
                                  actual, tcx.sess.str_of(field.ident.node))
                    }, struct_ty, None);
                    error_happened = true;
                }
                Some((_, true)) => {
                    tcx.sess.span_err(
                        field.ident.span,
                        format!("field `{}` specified more than once",
                             tcx.sess.str_of(field.ident.node)));
                    error_happened = true;
                }
                Some((field_id, false)) => {
                    expected_field_type =
                        ty::lookup_field_type(
                            tcx, class_id, field_id, &substitutions);
                    class_field_map.insert(
                        field.ident.node.name, (field_id, true));
                    fields_found += 1;
                }
            }
            // Make sure to give a type to the field even if there's
            // an error, so we can continue typechecking
            check_expr_coercable_to_type(
                    fcx,
                    field.expr,
                    expected_field_type);
        }

        if error_happened {
            fcx.write_error(node_id);
        }

        if check_completeness && !error_happened {
            // Make sure the programmer specified all the fields.
            assert!(fields_found <= field_types.len());
            if fields_found < field_types.len() {
                let mut missing_fields = ~[];
                for class_field in field_types.iter() {
                    let name = class_field.name;
                    let (_, seen) = *class_field_map.get(&name);
                    if !seen {
                        missing_fields.push(
                            ~"`" + token::interner_get(name) + "`");
                    }
                }

                tcx.sess.span_err(span,
                                  format!("missing field{}: {}",
                                       if missing_fields.len() == 1 {
                                           ""
                                       } else {
                                           "s"
                                       },
                                       missing_fields.connect(", ")));
             }
        }

        if !error_happened {
            fcx.write_ty(node_id, ty::mk_struct(fcx.ccx.tcx,
                                class_id, substitutions));
        }
    }

    fn check_struct_constructor(fcx: @FnCtxt,
                                id: ast::NodeId,
                                span: codemap::Span,
                                class_id: ast::DefId,
                                fields: &[ast::Field],
                                base_expr: Option<@ast::Expr>) {
        let tcx = fcx.ccx.tcx;

        // Look up the number of type parameters and the raw type, and
        // determine whether the class is region-parameterized.
        let item_type = ty::lookup_item_type(tcx, class_id);
        let type_parameter_count = item_type.generics.type_param_defs.len();
        let region_parameter_count = item_type.generics.region_param_defs.len();
        let raw_type = item_type.ty;

        // Generate the struct type.
        let regions = fcx.infcx().next_region_vars(
            infer::BoundRegionInTypeOrImpl(span),
            region_parameter_count);
        let type_parameters = fcx.infcx().next_ty_vars(type_parameter_count);
        let substitutions = substs {
            regions: ty::NonerasedRegions(opt_vec::from(regions)),
            self_ty: None,
            tps: type_parameters
        };

        let mut struct_type = ty::subst(tcx, &substitutions, raw_type);

        // Look up and check the fields.
        let class_fields = ty::lookup_struct_fields(tcx, class_id);
        check_struct_or_variant_fields(fcx,
                                           struct_type,
                                           span,
                                           class_id,
                                           id,
                                           substitutions,
                                           class_fields,
                                           fields,
                                           base_expr.is_none());
        if ty::type_is_error(fcx.node_ty(id)) {
            struct_type = ty::mk_err();
        }

        // Check the base expression if necessary.
        match base_expr {
            None => {}
            Some(base_expr) => {
                check_expr_has_type(fcx, base_expr, struct_type);
                if ty::type_is_bot(fcx.node_ty(base_expr.id)) {
                    struct_type = ty::mk_bot();
                }
            }
        }

        // Write in the resulting type.
        fcx.write_ty(id, struct_type);
    }

    fn check_struct_enum_variant(fcx: @FnCtxt,
                                 id: ast::NodeId,
                                 span: codemap::Span,
                                 enum_id: ast::DefId,
                                 variant_id: ast::DefId,
                                 fields: &[ast::Field]) {
        let tcx = fcx.ccx.tcx;

        // Look up the number of type parameters and the raw type, and
        // determine whether the enum is region-parameterized.
        let item_type = ty::lookup_item_type(tcx, enum_id);
        let type_parameter_count = item_type.generics.type_param_defs.len();
        let region_parameter_count = item_type.generics.region_param_defs.len();
        let raw_type = item_type.ty;

        // Generate the enum type.
        let regions = fcx.infcx().next_region_vars(
            infer::BoundRegionInTypeOrImpl(span),
            region_parameter_count);
        let type_parameters = fcx.infcx().next_ty_vars(type_parameter_count);
        let substitutions = substs {
            regions: ty::NonerasedRegions(opt_vec::from(regions)),
            self_ty: None,
            tps: type_parameters
        };

        let enum_type = ty::subst(tcx, &substitutions, raw_type);

        // Look up and check the enum variant fields.
        let variant_fields = ty::lookup_struct_fields(tcx, variant_id);
        check_struct_or_variant_fields(fcx,
                                       enum_type,
                                       span,
                                       variant_id,
                                       id,
                                       substitutions,
                                       variant_fields,
                                       fields,
                                       true);
        fcx.write_ty(id, enum_type);
    }

    let tcx = fcx.ccx.tcx;
    let id = expr.id;
    match expr.node {
      ast::ExprVstore(ev, vst) => {
        let typ = match ev.node {
          ast::ExprLit(lit) if ast_util::lit_is_str(lit) => {
            let tt = ast_expr_vstore_to_vstore(fcx, ev, vst);
            ty::mk_str(tcx, tt)
          }
          ast::ExprVec(ref args, mutbl) => {
            let tt = ast_expr_vstore_to_vstore(fcx, ev, vst);
            let mut any_error = false;
            let mut any_bot = false;
            let mutability = match vst {
                ast::ExprVstoreMutSlice => ast::MutMutable,
                _ => mutbl,
            };
            let t: ty::t = fcx.infcx().next_ty_var();
            for e in args.iter() {
                check_expr_has_type(fcx, *e, t);
                let arg_t = fcx.expr_ty(*e);
                if ty::type_is_error(arg_t) {
                    any_error = true;
                }
                else if ty::type_is_bot(arg_t) {
                    any_bot = true;
                }
            }
            if any_error {
                ty::mk_err()
            } else if any_bot {
                ty::mk_bot()
            } else {
                ty::mk_vec(tcx, ty::mt {ty: t, mutbl: mutability}, tt)
            }
          }
          ast::ExprRepeat(element, count_expr, mutbl) => {
            check_expr_with_hint(fcx, count_expr, ty::mk_uint());
            let _ = ty::eval_repeat_count(fcx, count_expr);
            let tt = ast_expr_vstore_to_vstore(fcx, ev, vst);
            let mutability = match vst {
                ast::ExprVstoreMutSlice => ast::MutMutable,
                _ => mutbl,
            };
            let t: ty::t = fcx.infcx().next_ty_var();
            check_expr_has_type(fcx, element, t);
            let arg_t = fcx.expr_ty(element);
            if ty::type_is_error(arg_t) {
                ty::mk_err()
            } else if ty::type_is_bot(arg_t) {
                ty::mk_bot()
            } else {
                ty::mk_vec(tcx, ty::mt {ty: t, mutbl: mutability}, tt)
            }
          }
          _ =>
            tcx.sess.span_bug(expr.span, "vstore modifier on non-sequence")
        };
        fcx.write_ty(ev.id, typ);
        fcx.write_ty(id, typ);
      }

      ast::ExprBox(place, subexpr) => {
          check_expr(fcx, place);
          check_expr(fcx, subexpr);

          let mut checked = false;
          match place.node {
              ast::ExprPath(ref path) => {
                  // XXX(pcwalton): For now we hardcode the two permissible
                  // places: the exchange heap and the managed heap.
                  let definition = lookup_def(fcx, path.span, place.id);
                  let def_id = ast_util::def_id_of_def(definition);
                  match tcx.lang_items.items[ExchangeHeapLangItem as uint] {
                      Some(item_def_id) if def_id == item_def_id => {
                          fcx.write_ty(id, ty::mk_uniq(tcx,
                                                       fcx.expr_ty(subexpr)));
                          checked = true
                      }
                      Some(_) | None => {}
                  }
                  if !checked {
                      match tcx.lang_items
                               .items[ManagedHeapLangItem as uint] {
                          Some(item_def_id) if def_id == item_def_id => {
                              // Assign the magic `Gc<T>` struct.
                              let gc_struct_id =
                                  match tcx.lang_items
                                           .require(GcLangItem) {
                                      Ok(id) => id,
                                      Err(msg) => {
                                          tcx.sess.span_err(expr.span, msg);
                                          ast::DefId {
                                              crate: ast::CRATE_NODE_ID,
                                              node: ast::DUMMY_NODE_ID,
                                          }
                                      }
                                  };
                              let regions =
                                  ty::NonerasedRegions(opt_vec::Empty);
                              let sty = ty::mk_struct(tcx,
                                                      gc_struct_id,
                                                      substs {
                                                        self_ty: None,
                                                        tps: ~[
                                                            fcx.expr_ty(
                                                                subexpr)
                                                        ],
                                                        regions: regions,
                                                      });
                              fcx.write_ty(id, sty);
                              checked = true
                          }
                          Some(_) | None => {}
                      }
                  }
              }
              _ => {}
          }

          if !checked {
              tcx.sess.span_err(expr.span,
                                "only the managed heap and exchange heap are \
                                 currently supported")
          }
      }

      ast::ExprLit(lit) => {
        let typ = check_lit(fcx, lit);
        fcx.write_ty(id, typ);
      }
      ast::ExprBinary(callee_id, op, lhs, rhs) => {
        check_binop(fcx,
                    callee_id,
                    expr,
                    op,
                    lhs,
                    rhs,
                    expected,
                    SimpleBinop);

        let lhs_ty = fcx.expr_ty(lhs);
        let rhs_ty = fcx.expr_ty(rhs);
        if ty::type_is_error(lhs_ty) ||
            ty::type_is_error(rhs_ty) {
            fcx.write_error(id);
        }
        else if ty::type_is_bot(lhs_ty) ||
          (ty::type_is_bot(rhs_ty) && !ast_util::lazy_binop(op)) {
            fcx.write_bot(id);
        }
      }
      ast::ExprAssignOp(callee_id, op, lhs, rhs) => {
        check_binop(fcx,
                    callee_id,
                    expr,
                    op,
                    lhs,
                    rhs,
                    expected,
                    BinopAssignment);

        let lhs_t = fcx.expr_ty(lhs);
        let result_t = fcx.expr_ty(expr);
        demand::suptype(fcx, expr.span, result_t, lhs_t);

        let tcx = fcx.tcx();
        if !ty::expr_is_lval(tcx, fcx.ccx.method_map, lhs) {
            tcx.sess.span_err(lhs.span, "illegal left-hand side expression");
        }

        // Overwrite result of check_binop...this preserves existing behavior
        // but seems quite dubious with regard to user-defined methods
        // and so forth. - Niko
        if !ty::type_is_error(result_t)
            && !ty::type_is_bot(result_t) {
            fcx.write_nil(expr.id);
        }
      }
      ast::ExprUnary(callee_id, unop, oprnd) => {
        let exp_inner = unpack_expected(fcx, expected, |sty| {
            match unop {
                ast::UnBox | ast::UnUniq => match *sty {
                    ty::ty_box(ty) | ty::ty_uniq(ty) => Some(ty),
                    _ => None
                },
                ast::UnNot | ast::UnNeg => expected,
                ast::UnDeref => None
            }
        });
        check_expr_with_opt_hint(fcx, oprnd, exp_inner);
        let mut oprnd_t = fcx.expr_ty(oprnd);
        if !ty::type_is_error(oprnd_t) &&
              !ty::type_is_bot(oprnd_t) {
            match unop {
                ast::UnBox => {
                    oprnd_t = ty::mk_box(tcx, oprnd_t)
                }
                ast::UnUniq => {
                    oprnd_t = ty::mk_uniq(tcx, oprnd_t);
                }
                ast::UnDeref => {
                    let sty = structure_of(fcx, expr.span, oprnd_t);
                    let operand_ty = ty::deref_sty(sty, true);
                    match operand_ty {
                        Some(mt) => {
                            oprnd_t = mt.ty
                        }
                        None => {
                            match *sty {
                                ty::ty_struct(did, ref substs) if {
                                    let fields = ty::struct_fields(fcx.tcx(), did, substs);
                                    fields.len() == 1
                                      && fields[0].ident == token::special_idents::unnamed_field
                                } => {
                                    // This is an obsolete struct deref
                                    tcx.sess.span_err(
                                        expr.span,
                                        "single-field tuple-structs can no longer be dereferenced");
                                }
                                _ => {
                                    fcx.type_error_message(expr.span,
                                        |actual| {
                                            format!("type `{}` cannot be dereferenced", actual)
                                    }, oprnd_t, None);
                                }
                            }
                        }
                    }
                }
                ast::UnNot => {
                    oprnd_t = structurally_resolved_type(fcx, oprnd.span,
                                                         oprnd_t);
                    if !(ty::type_is_integral(oprnd_t) ||
                         ty::get(oprnd_t).sty == ty::ty_bool) {
                        oprnd_t = check_user_unop(fcx, callee_id,
                            "!", "not", expr, oprnd, oprnd_t,
                                                  expected);
                    }
                }
                ast::UnNeg => {
                    oprnd_t = structurally_resolved_type(fcx, oprnd.span,
                                                         oprnd_t);
                    if !(ty::type_is_integral(oprnd_t) ||
                         ty::type_is_fp(oprnd_t)) {
                        oprnd_t = check_user_unop(fcx, callee_id,
                            "-", "neg", expr, oprnd, oprnd_t, expected);
                    }
                }
            }
        }
        fcx.write_ty(id, oprnd_t);
      }
      ast::ExprAddrOf(mutbl, oprnd) => {
          let hint = unpack_expected(
              fcx, expected,
              |sty| match *sty { ty::ty_rptr(_, ref mt) => Some(mt.ty),
                                 _ => None });
        check_expr_with_opt_hint(fcx, oprnd, hint);

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
        let region = fcx.infcx().next_region_var(
            infer::AddrOfRegion(expr.span));

        let tm = ty::mt { ty: fcx.expr_ty(oprnd), mutbl: mutbl };
        let oprnd_t = if ty::type_is_error(tm.ty) {
            ty::mk_err()
        } else if ty::type_is_bot(tm.ty) {
            ty::mk_bot()
        }
        else {
            ty::mk_rptr(tcx, region, tm)
        };
        fcx.write_ty(id, oprnd_t);
      }
      ast::ExprPath(ref pth) => {
        let defn = lookup_def(fcx, pth.span, id);

        check_type_parameter_positions_in_path(fcx, pth, defn);
        let tpt = ty_param_bounds_and_ty_for_def(fcx, expr.span, defn);
        instantiate_path(fcx, pth, tpt, defn, expr.span, expr.id);
      }
      ast::ExprSelf => {
        let definition = lookup_def(fcx, expr.span, id);
        let ty_param_bounds_and_ty =
            ty_param_bounds_and_ty_for_def(fcx, expr.span, definition);
        fcx.write_ty(id, ty_param_bounds_and_ty.ty);
      }
      ast::ExprInlineAsm(ref ia) => {
          for &(_, input) in ia.inputs.iter() {
              check_expr(fcx, input);
          }
          for &(_, out) in ia.outputs.iter() {
              check_expr(fcx, out);
          }
          fcx.write_nil(id);
      }
      ast::ExprMac(_) => tcx.sess.bug("unexpanded macro"),
      ast::ExprBreak(_) => { fcx.write_bot(id); }
      ast::ExprAgain(_) => { fcx.write_bot(id); }
      ast::ExprRet(expr_opt) => {
        let ret_ty = fcx.ret_ty;
        match expr_opt {
          None => match fcx.mk_eqty(false, infer::Misc(expr.span),
                                    ret_ty, ty::mk_nil()) {
            result::Ok(_) => { /* fall through */ }
            result::Err(_) => {
                tcx.sess.span_err(
                    expr.span,
                    "`return;` in function returning non-nil");
            }
          },
          Some(e) => {
              check_expr_has_type(fcx, e, ret_ty);
          }
        }
        fcx.write_bot(id);
      }
      ast::ExprLogLevel => {
        fcx.write_ty(id, ty::mk_u32())
      }
      ast::ExprParen(a) => {
        check_expr_with_opt_hint(fcx, a, expected);
        fcx.write_ty(id, fcx.expr_ty(a));
      }
      ast::ExprAssign(lhs, rhs) => {
        check_assignment(fcx, lhs, rhs, id);

        let tcx = fcx.tcx();
        if !ty::expr_is_lval(tcx, fcx.ccx.method_map, lhs) {
            tcx.sess.span_err(lhs.span, "illegal left-hand side expression");
        }

        let lhs_ty = fcx.expr_ty(lhs);
        let rhs_ty = fcx.expr_ty(rhs);
        if ty::type_is_error(lhs_ty) || ty::type_is_error(rhs_ty) {
            fcx.write_error(id);
        }
        else if ty::type_is_bot(lhs_ty) || ty::type_is_bot(rhs_ty) {
            fcx.write_bot(id);
        }
        else {
            fcx.write_nil(id);
        }
      }
      ast::ExprIf(cond, then_blk, opt_else_expr) => {
        check_then_else(fcx, cond, then_blk, opt_else_expr,
                        id, expr.span, expected);
      }
      ast::ExprWhile(cond, body) => {
        check_expr_has_type(fcx, cond, ty::mk_bool());
        check_block_no_value(fcx, body);
        let cond_ty = fcx.expr_ty(cond);
        let body_ty = fcx.node_ty(body.id);
        if ty::type_is_error(cond_ty) || ty::type_is_error(body_ty) {
            fcx.write_error(id);
        }
        else if ty::type_is_bot(cond_ty) {
            fcx.write_bot(id);
        }
        else {
            fcx.write_nil(id);
        }
      }
      ast::ExprForLoop(..) =>
          fail!("non-desugared expr_for_loop"),
      ast::ExprLoop(body, _) => {
        check_block_no_value(fcx, (body));
        if !may_break(tcx, expr.id, body) {
            fcx.write_bot(id);
        }
        else {
            fcx.write_nil(id);
        }
      }
      ast::ExprMatch(discrim, ref arms) => {
        _match::check_match(fcx, expr, discrim, *arms);
      }
      ast::ExprFnBlock(decl, body) => {
        check_expr_fn(fcx,
                      expr,
                      Some(ast::BorrowedSigil),
                      decl,
                      body,
                      Vanilla,
                      expected);
      }
      ast::ExprProc(decl, body) => {
        check_expr_fn(fcx,
                      expr,
                      Some(ast::OwnedSigil),
                      decl,
                      body,
                      Vanilla,
                      expected);
      }
      ast::ExprDoBody(b) => {
        let expected_sty = unpack_expected(fcx,
                                           expected,
                                           |x| Some((*x).clone()));
        let inner_ty = match expected_sty {
            Some(ty::ty_closure(ref closure_ty))
                    if closure_ty.sigil == ast::OwnedSigil => {
                expected.unwrap()
            }
            _ => match expected {
                Some(expected_t) => {
                    fcx.type_error_message(expr.span, |actual| {
                        format!("last argument in `do` call \
                              has non-procedure type: {}",
                             actual)
                    }, expected_t, None);
                    let err_ty = ty::mk_err();
                    fcx.write_ty(id, err_ty);
                    err_ty
                }
                None => {
                    fcx.tcx().sess.impossible_case(
                        expr.span,
                        "do body must have expected type")
                }
            }
        };
        match b.node {
          ast::ExprFnBlock(decl, body) => {
            check_expr_fn(fcx, b, None,
                          decl, body, DoBlock, Some(inner_ty));
            demand::suptype(fcx, b.span, inner_ty, fcx.expr_ty(b));
          }
          // argh
          _ => fail!("expected fn ty")
        }
        fcx.write_ty(expr.id, fcx.node_ty(b.id));
      }
      ast::ExprBlock(b) => {
        check_block_with_expected(fcx, b, expected);
        fcx.write_ty(id, fcx.node_ty(b.id));
      }
      ast::ExprCall(f, ref args, sugar) => {
          check_call(fcx, expr.id, expr, f, *args, sugar);
          let f_ty = fcx.expr_ty(f);
          let (args_bot, args_err) = args.iter().fold((false, false),
             |(rest_bot, rest_err), a| {
                 // is this not working?
                 let a_ty = fcx.expr_ty(*a);
                 (rest_bot || ty::type_is_bot(a_ty),
                  rest_err || ty::type_is_error(a_ty))});
          if ty::type_is_error(f_ty) || args_err {
              fcx.write_error(id);
          }
          else if ty::type_is_bot(f_ty) || args_bot {
              fcx.write_bot(id);
          }
      }
      ast::ExprMethodCall(callee_id, rcvr, ident, ref tps, ref args, sugar) => {
        check_method_call(fcx, callee_id, expr, rcvr, ident, *args, *tps, sugar);
        let f_ty = fcx.expr_ty(rcvr);
        let arg_tys = args.map(|a| fcx.expr_ty(*a));
        let (args_bot, args_err) = arg_tys.iter().fold((false, false),
             |(rest_bot, rest_err), a| {
              (rest_bot || ty::type_is_bot(*a),
               rest_err || ty::type_is_error(*a))});
        if ty::type_is_error(f_ty) || args_err {
            fcx.write_error(id);
        }
        else if ty::type_is_bot(f_ty) || args_bot {
            fcx.write_bot(id);
        }
      }
      ast::ExprCast(e, t) => {
        check_expr(fcx, e);
        let t_1 = fcx.to_ty(t);
        let t_e = fcx.expr_ty(e);

        debug!("t_1={}", fcx.infcx().ty_to_str(t_1));
        debug!("t_e={}", fcx.infcx().ty_to_str(t_e));

        if ty::type_is_error(t_e) {
            fcx.write_error(id);
        }
        else if ty::type_is_bot(t_e) {
            fcx.write_bot(id);
        }
        else {
            match ty::get(t_1).sty {
                // This will be looked up later on
                ty::ty_trait(..) => (),

                _ => {
                    if ty::type_is_nil(t_e) {
                        fcx.type_error_message(expr.span, |actual| {
                            format!("cast from nil: `{}` as `{}`", actual,
                                 fcx.infcx().ty_to_str(t_1))
                        }, t_e, None);
                    } else if ty::type_is_nil(t_1) {
                        fcx.type_error_message(expr.span, |actual| {
                            format!("cast to nil: `{}` as `{}`", actual,
                                 fcx.infcx().ty_to_str(t_1))
                        }, t_e, None);
                    }

                    let t1 = structurally_resolved_type(fcx, e.span, t_1);
                    let te = structurally_resolved_type(fcx, e.span, t_e);
                    let t_1_is_scalar = type_is_scalar(fcx, expr.span, t_1);
                    let t_1_is_char = type_is_char(fcx, expr.span, t_1);
                    let t_1_is_bare_fn = type_is_bare_fn(fcx, expr.span, t_1);

                    // casts to scalars other than `char` and `bare fn` are trivial
                    let t_1_is_trivial = t_1_is_scalar &&
                        !t_1_is_char && !t_1_is_bare_fn;

                    if type_is_c_like_enum(fcx, expr.span, t_e) && t_1_is_trivial {
                        // casts from C-like enums are allowed
                    } else if t_1_is_char {
                        if ty::get(te).sty != ty::ty_uint(ast::TyU8) {
                            fcx.type_error_message(expr.span, |actual| {
                                format!("only `u8` can be cast as `char`, not `{}`", actual)
                            }, t_e, None);
                        }
                    } else if ty::get(t1).sty == ty::ty_bool {
                        fcx.tcx().sess.span_err(expr.span,
                                                "cannot cast as `bool`, compare with zero instead");
                    } else if type_is_region_ptr(fcx, expr.span, t_e) &&
                        type_is_unsafe_ptr(fcx, expr.span, t_1) {

                        fn is_vec(t: ty::t) -> bool {
                            match ty::get(t).sty {
                                ty::ty_vec(..) => true,
                                _ => false
                            }
                        }
                        fn types_compatible(fcx: @FnCtxt, sp: Span,
                                            t1: ty::t, t2: ty::t) -> bool {
                            if !is_vec(t1) {
                                false
                            } else {
                                let el = ty::sequence_element_type(fcx.tcx(),
                                                                   t1);
                                infer::mk_eqty(fcx.infcx(), false,
                                               infer::Misc(sp), el, t2).is_ok()
                            }
                        }

                        // Due to the limitations of LLVM global constants,
                        // region pointers end up pointing at copies of
                        // vector elements instead of the original values.
                        // To allow unsafe pointers to work correctly, we
                        // need to special-case obtaining an unsafe pointer
                        // from a region pointer to a vector.

                        /* this cast is only allowed from &[T] to *T or
                        &T to *T. */
                        match (&ty::get(te).sty, &ty::get(t_1).sty) {
                            (&ty::ty_rptr(_, mt1), &ty::ty_ptr(mt2))
                            if types_compatible(fcx, e.span,
                                                mt1.ty, mt2.ty) => {
                                /* this case is allowed */
                            }
                            _ => {
                                demand::coerce(fcx, e.span, t_1, e);
                            }
                        }
                    } else if !(type_is_scalar(fcx,expr.span,t_e)
                                && t_1_is_trivial) {
                        /*
                        If more type combinations should be supported than are
                        supported here, then file an enhancement issue and
                        record the issue number in this comment.
                        */
                        fcx.type_error_message(expr.span, |actual| {
                            format!("non-scalar cast: `{}` as `{}`", actual,
                                 fcx.infcx().ty_to_str(t_1))
                        }, t_e, None);
                    }
                }
            }
            fcx.write_ty(id, t_1);
        }
      }
      ast::ExprVec(ref args, mutbl) => {
        let t: ty::t = fcx.infcx().next_ty_var();
        for e in args.iter() {
            check_expr_has_type(fcx, *e, t);
        }
        let typ = ty::mk_vec(tcx, ty::mt {ty: t, mutbl: mutbl},
                             ty::vstore_fixed(args.len()));
        fcx.write_ty(id, typ);
      }
      ast::ExprRepeat(element, count_expr, mutbl) => {
        check_expr_with_hint(fcx, count_expr, ty::mk_uint());
        let count = ty::eval_repeat_count(fcx, count_expr);
        let t: ty::t = fcx.infcx().next_ty_var();
        check_expr_has_type(fcx, element, t);
        let element_ty = fcx.expr_ty(element);
        if ty::type_is_error(element_ty) {
            fcx.write_error(id);
        }
        else if ty::type_is_bot(element_ty) {
            fcx.write_bot(id);
        }
        else {
            let t = ty::mk_vec(tcx, ty::mt {ty: t, mutbl: mutbl},
                               ty::vstore_fixed(count));
            fcx.write_ty(id, t);
        }
      }
      ast::ExprTup(ref elts) => {
        let flds = unpack_expected(fcx, expected, |sty| {
            match *sty {
                ty::ty_tup(ref flds) => Some((*flds).clone()),
                _ => None
            }
        });
        let mut bot_field = false;
        let mut err_field = false;

        let elt_ts = elts.iter().enumerate().map(|(i, e)| {
            let opt_hint = match flds {
                Some(ref fs) if i < fs.len() => Some(fs[i]),
                _ => None
            };
            check_expr_with_opt_hint(fcx, *e, opt_hint);
            let t = fcx.expr_ty(*e);
            err_field = err_field || ty::type_is_error(t);
            bot_field = bot_field || ty::type_is_bot(t);
            t
        }).collect();
        if bot_field {
            fcx.write_bot(id);
        } else if err_field {
            fcx.write_error(id);
        } else {
            let typ = ty::mk_tup(tcx, elt_ts);
            fcx.write_ty(id, typ);
        }
      }
      ast::ExprStruct(ref path, ref fields, base_expr) => {
        // Resolve the path.
        let def_map = tcx.def_map.borrow();
        match def_map.get().find(&id) {
            Some(&ast::DefStruct(type_def_id)) => {
                check_struct_constructor(fcx, id, expr.span, type_def_id,
                                         *fields, base_expr);
            }
            Some(&ast::DefVariant(enum_id, variant_id, _)) => {
                check_struct_enum_variant(fcx, id, expr.span, enum_id,
                                          variant_id, *fields);
            }
            _ => {
                tcx.sess.span_bug(path.span,
                                  "structure constructor does not name a structure type");
            }
        }
      }
      ast::ExprField(base, field, ref tys) => {
        check_field(fcx, expr, base, field.name, *tys);
      }
      ast::ExprIndex(callee_id, base, idx) => {
          check_expr(fcx, base);
          check_expr(fcx, idx);
          let raw_base_t = fcx.expr_ty(base);
          let idx_t = fcx.expr_ty(idx);
          if ty::type_is_error(raw_base_t) || ty::type_is_bot(raw_base_t) {
              fcx.write_ty(id, raw_base_t);
          } else if ty::type_is_error(idx_t) || ty::type_is_bot(idx_t) {
              fcx.write_ty(id, idx_t);
          } else {
              let (base_t, derefs) = do_autoderef(fcx, expr.span, raw_base_t);
              let base_sty = structure_of(fcx, expr.span, base_t);
              match ty::index_sty(base_sty) {
                  Some(mt) => {
                      require_integral(fcx, idx.span, idx_t);
                      fcx.write_ty(id, mt.ty);
                      fcx.write_autoderef_adjustment(base.id, derefs);
                  }
                  None => {
                      let resolved = structurally_resolved_type(fcx,
                                                                expr.span,
                                                                raw_base_t);
                      let index_ident = tcx.sess.ident_of("index");
                      let error_message = || {
                        fcx.type_error_message(expr.span,
                                               |actual| {
                                                format!("cannot index a value \
                                                      of type `{}`",
                                                     actual)
                                               },
                                               base_t,
                                               None);
                      };
                      let ret_ty = lookup_op_method(fcx,
                                                    callee_id,
                                                    expr,
                                                    base,
                                                    resolved,
                                                    index_ident.name,
                                                    &[idx],
                                                    DoDerefArgs,
                                                    AutoderefReceiver,
                                                    error_message,
                                                    expected);
                      fcx.write_ty(id, ret_ty);
                  }
              }
          }
       }
    }

    debug!("type of expr({}) {} is...", expr.id,
           syntax::print::pprust::expr_to_str(expr, tcx.sess.intr()));
    debug!("... {}, expected is {}",
           ppaux::ty_to_str(tcx, fcx.expr_ty(expr)),
           match expected {
               Some(t) => ppaux::ty_to_str(tcx, t),
               _ => ~"empty"
           });

    unifier();
}

pub fn require_integral(fcx: @FnCtxt, sp: Span, t: ty::t) {
    if !type_is_integral(fcx, sp, t) {
        fcx.type_error_message(sp, |actual| {
            format!("mismatched types: expected integral type but found `{}`",
                 actual)
        }, t, None);
    }
}

pub fn check_decl_initializer(fcx: @FnCtxt,
                              nid: ast::NodeId,
                              init: &ast::Expr)
                            {
    let local_ty = fcx.local_ty(init.span, nid);
    check_expr_coercable_to_type(fcx, init, local_ty)
}

pub fn check_decl_local(fcx: @FnCtxt, local: &ast::Local)  {
    let tcx = fcx.ccx.tcx;

    let t = fcx.local_ty(local.span, local.id);
    fcx.write_ty(local.id, t);

    match local.init {
        Some(init) => {
            check_decl_initializer(fcx, local.id, init);
            let init_ty = fcx.expr_ty(init);
            if ty::type_is_error(init_ty) || ty::type_is_bot(init_ty) {
                fcx.write_ty(local.id, init_ty);
            }
        }
        _ => {}
    }

    let pcx = pat_ctxt {
        fcx: fcx,
        map: pat_id_map(tcx.def_map, local.pat),
    };
    _match::check_pat(&pcx, local.pat, t);
    let pat_ty = fcx.node_ty(local.pat.id);
    if ty::type_is_error(pat_ty) || ty::type_is_bot(pat_ty) {
        fcx.write_ty(local.id, pat_ty);
    }
}

pub fn check_stmt(fcx: @FnCtxt, stmt: &ast::Stmt)  {
    let node_id;
    let mut saw_bot = false;
    let mut saw_err = false;
    match stmt.node {
      ast::StmtDecl(decl, id) => {
        node_id = id;
        match decl.node {
          ast::DeclLocal(ref l) => {
              check_decl_local(fcx, *l);
              let l_t = fcx.node_ty(l.id);
              saw_bot = saw_bot || ty::type_is_bot(l_t);
              saw_err = saw_err || ty::type_is_error(l_t);
          }
          ast::DeclItem(_) => {/* ignore for now */ }
        }
      }
      ast::StmtExpr(expr, id) => {
        node_id = id;
        // Check with expected type of ()
        check_expr_has_type(fcx, expr, ty::mk_nil());
        let expr_ty = fcx.expr_ty(expr);
        saw_bot = saw_bot || ty::type_is_bot(expr_ty);
        saw_err = saw_err || ty::type_is_error(expr_ty);
      }
      ast::StmtSemi(expr, id) => {
        node_id = id;
        check_expr(fcx, expr);
        let expr_ty = fcx.expr_ty(expr);
        saw_bot |= ty::type_is_bot(expr_ty);
        saw_err |= ty::type_is_error(expr_ty);
      }
      ast::StmtMac(..) => fcx.ccx.tcx.sess.bug("unexpanded macro")
    }
    if saw_bot {
        fcx.write_bot(node_id);
    }
    else if saw_err {
        fcx.write_error(node_id);
    }
    else {
        fcx.write_nil(node_id)
    }
}

pub fn check_block_no_value(fcx: @FnCtxt, blk: &ast::Block)  {
    check_block_with_expected(fcx, blk, Some(ty::mk_nil()));
    let blkty = fcx.node_ty(blk.id);
    if ty::type_is_error(blkty) {
        fcx.write_error(blk.id);
    }
    else if ty::type_is_bot(blkty) {
        fcx.write_bot(blk.id);
    }
    else {
        let nilty = ty::mk_nil();
        demand::suptype(fcx, blk.span, nilty, blkty);
    }
}

pub fn check_block(fcx0: @FnCtxt, blk: &ast::Block)  {
    check_block_with_expected(fcx0, blk, None)
}

pub fn check_block_with_expected(fcx: @FnCtxt,
                                 blk: &ast::Block,
                                 expected: Option<ty::t>) {
    let prev = {
        let mut fcx_ps = fcx.ps.borrow_mut();
        let purity_state = fcx_ps.get().recurse(blk);
        replace(fcx_ps.get(), purity_state)
    };

    fcx.with_region_lb(blk.id, || {
        let mut warned = false;
        let mut last_was_bot = false;
        let mut any_bot = false;
        let mut any_err = false;
        for s in blk.stmts.iter() {
            check_stmt(fcx, *s);
            let s_id = ast_util::stmt_id(*s);
            let s_ty = fcx.node_ty(s_id);
            if last_was_bot && !warned && match s.node {
                  ast::StmtDecl(decl, _) => {
                      match decl.node {
                          ast::DeclLocal(_) => true,
                          _ => false,
                      }
                  }
                  ast::StmtExpr(_, _) | ast::StmtSemi(_, _) => true,
                  _ => false
                } {
                fcx.ccx.tcx.sess.add_lint(UnreachableCode, s_id, s.span,
                                          ~"unreachable statement");
                warned = true;
            }
            if ty::type_is_bot(s_ty) {
                last_was_bot = true;
            }
            any_bot = any_bot || ty::type_is_bot(s_ty);
            any_err = any_err || ty::type_is_error(s_ty);
        }
        match blk.expr {
            None => if any_err {
                fcx.write_error(blk.id);
            }
            else if any_bot {
                fcx.write_bot(blk.id);
            }
            else  {
                fcx.write_nil(blk.id);
            },
          Some(e) => {
            if any_bot && !warned {
                fcx.ccx.tcx.sess.add_lint(UnreachableCode, e.id, e.span,
                                          ~"unreachable expression");
            }
            check_expr_with_opt_hint(fcx, e, expected);
              let ety = fcx.expr_ty(e);
              fcx.write_ty(blk.id, ety);
              if any_err {
                  fcx.write_error(blk.id);
              }
              else if any_bot {
                  fcx.write_bot(blk.id);
              }
          }
        };
    });

    fcx.ps.set(prev);
}

pub fn check_const(ccx: @CrateCtxt,
                   sp: Span,
                   e: &ast::Expr,
                   id: ast::NodeId) {
    let rty = ty::node_id_to_type(ccx.tcx, id);
    let fcx = blank_fn_ctxt(ccx, rty, e.id);
    let declty = {
        let tcache = fcx.ccx.tcx.tcache.borrow();
        tcache.get().get(&local_def(id)).ty
    };
    check_const_with_ty(fcx, sp, e, declty);
}

pub fn check_const_with_ty(fcx: @FnCtxt,
                           _: Span,
                           e: &ast::Expr,
                           declty: ty::t) {
    check_expr(fcx, e);
    let cty = fcx.expr_ty(e);
    demand::suptype(fcx, e.span, declty, cty);
    regionck::regionck_expr(fcx, e);
    writeback::resolve_type_vars_in_expr(fcx, e);
}

/// Checks whether a type can be created without an instance of itself.
/// This is similar but different from the question of whether a type
/// can be represented.  For example, the following type:
///
///     enum foo { None, Some(foo) }
///
/// is instantiable but is not representable.  Similarly, the type
///
///     enum foo { Some(@foo) }
///
/// is representable, but not instantiable.
pub fn check_instantiable(tcx: ty::ctxt,
                          sp: Span,
                          item_id: ast::NodeId) {
    let item_ty = ty::node_id_to_type(tcx, item_id);
    if !ty::is_instantiable(tcx, item_ty) {
        tcx.sess.span_err(sp, format!("this type cannot be instantiated \
                  without an instance of itself; \
                  consider using `Option<{}>`",
                                   ppaux::ty_to_str(tcx, item_ty)));
    }
}

pub fn check_simd(tcx: ty::ctxt, sp: Span, id: ast::NodeId) {
    let t = ty::node_id_to_type(tcx, id);
    if ty::type_needs_subst(t) {
        tcx.sess.span_err(sp, "SIMD vector cannot be generic");
        return;
    }
    match ty::get(t).sty {
        ty::ty_struct(did, ref substs) => {
            let fields = ty::lookup_struct_fields(tcx, did);
            if fields.is_empty() {
                tcx.sess.span_err(sp, "SIMD vector cannot be empty");
                return;
            }
            let e = ty::lookup_field_type(tcx, did, fields[0].id, substs);
            if !fields.iter().all(
                         |f| ty::lookup_field_type(tcx, did, f.id, substs) == e) {
                tcx.sess.span_err(sp, "SIMD vector should be homogeneous");
                return;
            }
            if !ty::type_is_machine(e) {
                tcx.sess.span_err(sp, "SIMD vector element type should be \
                                       machine type");
                return;
            }
        }
        _ => ()
    }
}

pub fn check_enum_variants(ccx: @CrateCtxt,
                           sp: Span,
                           vs: &[ast::P<ast::Variant>],
                           id: ast::NodeId) {

    fn disr_in_range(ccx: @CrateCtxt,
                     ty: attr::IntType,
                     disr: ty::Disr) -> bool {
        fn uint_in_range(ccx: @CrateCtxt, ty: ast::UintTy, disr: ty::Disr) -> bool {
            match ty {
                ast::TyU8 => disr as u8 as Disr == disr,
                ast::TyU16 => disr as u16 as Disr == disr,
                ast::TyU32 => disr as u32 as Disr == disr,
                ast::TyU64 => disr as u64 as Disr == disr,
                ast::TyU => uint_in_range(ccx, ccx.tcx.sess.targ_cfg.uint_type, disr)
            }
        }
        fn int_in_range(ccx: @CrateCtxt, ty: ast::IntTy, disr: ty::Disr) -> bool {
            match ty {
                ast::TyI8 => disr as i8 as Disr == disr,
                ast::TyI16 => disr as i16 as Disr == disr,
                ast::TyI32 => disr as i32 as Disr == disr,
                ast::TyI64 => disr as i64 as Disr == disr,
                ast::TyI => int_in_range(ccx, ccx.tcx.sess.targ_cfg.int_type, disr)
            }
        }
        match ty {
            attr::UnsignedInt(ty) => uint_in_range(ccx, ty, disr),
            attr::SignedInt(ty) => int_in_range(ccx, ty, disr)
        }
    }

    fn do_check(ccx: @CrateCtxt,
                vs: &[ast::P<ast::Variant>],
                id: ast::NodeId,
                hint: attr::ReprAttr)
                -> ~[@ty::VariantInfo] {

        let rty = ty::node_id_to_type(ccx.tcx, id);
        let mut variants: ~[@ty::VariantInfo] = ~[];
        let mut disr_vals: ~[ty::Disr] = ~[];
        let mut prev_disr_val: Option<ty::Disr> = None;

        for &v in vs.iter() {

            // If the discriminant value is specified explicitly in the enum check whether the
            // initialization expression is valid, otherwise use the last value plus one.
            let mut current_disr_val = match prev_disr_val {
                Some(prev_disr_val) => prev_disr_val + 1,
                None => ty::INITIAL_DISCRIMINANT_VALUE
            };

            match v.node.disr_expr {
                Some(e) => {
                    debug!("disr expr, checking {}", pprust::expr_to_str(e, ccx.tcx.sess.intr()));

                    let fcx = blank_fn_ctxt(ccx, rty, e.id);
                    let declty = ty::mk_int_var(ccx.tcx, fcx.infcx().next_int_var_id());
                    check_const_with_ty(fcx, e.span, e, declty);
                    // check_expr (from check_const pass) doesn't guarantee
                    // that the expression is in an form that eval_const_expr can
                    // handle, so we may still get an internal compiler error

                    match const_eval::eval_const_expr_partial(&ccx.tcx, e) {
                        Ok(const_eval::const_int(val)) => current_disr_val = val as Disr,
                        Ok(const_eval::const_uint(val)) => current_disr_val = val as Disr,
                        Ok(_) => {
                            ccx.tcx.sess.span_err(e.span, "expected signed integer constant");
                        }
                        Err(ref err) => {
                            ccx.tcx.sess.span_err(e.span, format!("expected constant: {}", (*err)));
                        }
                    }
                },
                None => ()
            };

            // Check for duplicate discriminant values
            if disr_vals.contains(&current_disr_val) {
                ccx.tcx.sess.span_err(v.span, "discriminant value already exists");
            }
            // Check for unrepresentable discriminant values
            match hint {
                attr::ReprAny | attr::ReprExtern => (),
                attr::ReprInt(sp, ity) => {
                    if !disr_in_range(ccx, ity, current_disr_val) {
                        ccx.tcx.sess.span_err(v.span,
                                              "discriminant value outside specified type");
                        ccx.tcx.sess.span_note(sp, "discriminant type specified here");
                    }
                }
            }
            disr_vals.push(current_disr_val);

            let variant_info = @VariantInfo::from_ast_variant(ccx.tcx, v, current_disr_val);
            prev_disr_val = Some(current_disr_val);

            variants.push(variant_info);
        }

        return variants;
    }

    let rty = ty::node_id_to_type(ccx.tcx, id);
    let hint = ty::lookup_repr_hint(ccx.tcx, ast::DefId { crate: ast::LOCAL_CRATE, node: id });
    if hint != attr::ReprAny && vs.len() <= 1 {
        ccx.tcx.sess.span_err(sp, format!("unsupported representation for {}variant enum",
                                          if vs.len() == 1 { "uni" } else { "zero-" }))
    }

    let variants = do_check(ccx, vs, id, hint);

    // cache so that ty::enum_variants won't repeat this work
    {
        let mut enum_var_cache = ccx.tcx.enum_var_cache.borrow_mut();
        enum_var_cache.get().insert(local_def(id), @variants);
    }

    // Check that it is possible to represent this enum:
    let mut outer = true;
    let did = local_def(id);
    if ty::type_structurally_contains(ccx.tcx, rty, |sty| {
        match *sty {
          ty::ty_enum(id, _) if id == did => {
            if outer { outer = false; false }
            else { true }
          }
          _ => false
        }
    }) {
        ccx.tcx.sess.span_err(sp,
                              "illegal recursive enum type; \
                               wrap the inner value in a box to make it representable");
    }

    // Check that it is possible to instantiate this enum:
    //
    // This *sounds* like the same that as representable, but it's
    // not.  See def'n of `check_instantiable()` for details.
    check_instantiable(ccx.tcx, sp, id);
}

pub fn lookup_def(fcx: @FnCtxt, sp: Span, id: ast::NodeId) -> ast::Def {
    lookup_def_ccx(fcx.ccx, sp, id)
}

// Returns the type parameter count and the type for the given definition.
pub fn ty_param_bounds_and_ty_for_def(fcx: @FnCtxt,
                                      sp: Span,
                                      defn: ast::Def)
                                   -> ty_param_bounds_and_ty {
    match defn {
      ast::DefArg(nid, _) | ast::DefLocal(nid, _) | ast::DefSelf(nid, _) |
      ast::DefBinding(nid, _) => {
          let typ = fcx.local_ty(sp, nid);
          return no_params(typ);
      }
      ast::DefFn(id, _) | ast::DefStaticMethod(id, _, _) |
      ast::DefStatic(id, _) | ast::DefVariant(_, id, _) |
      ast::DefStruct(id) => {
        return ty::lookup_item_type(fcx.ccx.tcx, id);
      }
      ast::DefUpvar(_, inner, _, _) => {
        return ty_param_bounds_and_ty_for_def(fcx, sp, *inner);
      }
      ast::DefTrait(_) |
      ast::DefTy(_) |
      ast::DefPrimTy(_) |
      ast::DefTyParam(..)=> {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value but found type");
      }
      ast::DefMod(..) | ast::DefForeignMod(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value but found module");
      }
      ast::DefUse(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value but found use");
      }
      ast::DefRegion(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value but found region");
      }
      ast::DefTyParamBinder(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value but found type parameter");
      }
      ast::DefLabel(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value but found label");
      }
      ast::DefSelfTy(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value but found self ty");
      }
      ast::DefMethod(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value but found method");
      }
    }
}

// Instantiates the given path, which must refer to an item with the given
// number of type parameters and type.
pub fn instantiate_path(fcx: @FnCtxt,
                        pth: &ast::Path,
                        tpt: ty_param_bounds_and_ty,
                        def: ast::Def,
                        span: Span,
                        node_id: ast::NodeId) {
    debug!(">>> instantiate_path");

    let ty_param_count = tpt.generics.type_param_defs.len();
    let mut ty_substs_len = 0;
    for segment in pth.segments.iter() {
        ty_substs_len += segment.types.len()
    }

    debug!("tpt={} ty_param_count={:?} ty_substs_len={:?}",
           tpt.repr(fcx.tcx()),
           ty_param_count,
           ty_substs_len);

    // determine the region parameters, using the value given by the user
    // (if any) and otherwise using a fresh region variable
    let num_expected_regions = tpt.generics.region_param_defs.len();
    let num_supplied_regions = pth.segments.last().lifetimes.len();
    let regions = if num_expected_regions == num_supplied_regions {
        pth.segments.last().lifetimes.map(
            |l| ast_region_to_region(fcx.tcx(), l))
    } else {
        if num_supplied_regions != 0 {
            fcx.ccx.tcx.sess.span_err(
                span,
                format!("expected {} lifetime parameter(s), \
                        found {} lifetime parameter(s)",
                        num_expected_regions, num_supplied_regions));
        }

        opt_vec::from(fcx.infcx().next_region_vars(
                infer::BoundRegionInTypeOrImpl(span),
                num_expected_regions))
    };

    // Special case: If there is a self parameter, omit it from the list of
    // type parameters.
    //
    // Here we calculate the "user type parameter count", which is the number
    // of type parameters actually manifest in the AST. This will differ from
    // the internal type parameter count when there are self types involved.
    let (user_type_parameter_count, self_parameter_index) = match def {
        ast::DefStaticMethod(_, provenance @ ast::FromTrait(_), _) => {
            let generics = generics_of_static_method_container(fcx.ccx.tcx,
                                                               provenance);
            (ty_param_count - 1, Some(generics.type_param_defs.len()))
        }
        _ => (ty_param_count, None),
    };

    // determine values for type parameters, using the values given by
    // the user (if any) and otherwise using fresh type variables
    let tps = if ty_substs_len == 0 {
        fcx.infcx().next_ty_vars(ty_param_count)
    } else if ty_param_count == 0 {
        fcx.ccx.tcx.sess.span_err
            (span, "this item does not take type parameters");
        fcx.infcx().next_ty_vars(ty_param_count)
    } else if ty_substs_len > user_type_parameter_count {
        fcx.ccx.tcx.sess.span_err
            (span,
             format!("too many type parameters provided: expected {}, found {}",
                  user_type_parameter_count, ty_substs_len));
        fcx.infcx().next_ty_vars(ty_param_count)
    } else if ty_substs_len < user_type_parameter_count {
        fcx.ccx.tcx.sess.span_err
            (span,
             format!("not enough type parameters provided: expected {}, found {}",
                  user_type_parameter_count, ty_substs_len));
        fcx.infcx().next_ty_vars(ty_param_count)
    } else {
        // Build up the list of type parameters, inserting the self parameter
        // at the appropriate position.
        let mut result = ~[];
        let mut pushed = false;
        for (i, &ast_type) in pth.segments
                                .iter()
                                .flat_map(|segment| segment.types.iter())
                                .enumerate() {
            match self_parameter_index {
                Some(index) if index == i => {
                    result.push(fcx.infcx().next_ty_vars(1)[0]);
                    pushed = true;
                }
                _ => {}
            }
            result.push(fcx.to_ty(ast_type))
        }

        // If the self parameter goes at the end, insert it there.
        if !pushed && self_parameter_index.is_some() {
            result.push(fcx.infcx().next_ty_vars(1)[0])
        }

        assert_eq!(result.len(), ty_param_count)
        result
    };

    let substs = substs {
        regions: ty::NonerasedRegions(regions),
        self_ty: None,
        tps: tps
    };
    fcx.write_ty_substs(node_id, tpt.ty, substs);

    debug!("<<<");
}

// Resolves `typ` by a single level if `typ` is a type variable.  If no
// resolution is possible, then an error is reported.
pub fn structurally_resolved_type(fcx: @FnCtxt, sp: Span, tp: ty::t)
                               -> ty::t {
    match infer::resolve_type(fcx.infcx(), tp, force_tvar) {
        Ok(t_s) if !ty::type_is_ty_var(t_s) => t_s,
        _ => {
            fcx.type_error_message(sp, |_actual| {
                ~"the type of this value must be known in this context"
            }, tp, None);
            demand::suptype(fcx, sp, ty::mk_err(), tp);
            tp
        }
    }
}

// Returns the one-level-deep structure of the given type.
pub fn structure_of<'a>(fcx: @FnCtxt, sp: Span, typ: ty::t)
                        -> &'a ty::sty {
    &ty::get(structurally_resolved_type(fcx, sp, typ)).sty
}

pub fn type_is_integral(fcx: @FnCtxt, sp: Span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    return ty::type_is_integral(typ_s);
}

pub fn type_is_scalar(fcx: @FnCtxt, sp: Span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    return ty::type_is_scalar(typ_s);
}

pub fn type_is_char(fcx: @FnCtxt, sp: Span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    return ty::type_is_char(typ_s);
}

pub fn type_is_bare_fn(fcx: @FnCtxt, sp: Span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    return ty::type_is_bare_fn(typ_s);
}

pub fn type_is_unsafe_ptr(fcx: @FnCtxt, sp: Span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    return ty::type_is_unsafe_ptr(typ_s);
}

pub fn type_is_region_ptr(fcx: @FnCtxt, sp: Span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    return ty::type_is_region_ptr(typ_s);
}

pub fn type_is_c_like_enum(fcx: @FnCtxt, sp: Span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    return ty::type_is_c_like_enum(fcx.ccx.tcx, typ_s);
}

pub fn ast_expr_vstore_to_vstore(fcx: @FnCtxt,
                                 e: &ast::Expr,
                                 v: ast::ExprVstore)
                              -> ty::vstore {
    match v {
        ast::ExprVstoreUniq => ty::vstore_uniq,
        ast::ExprVstoreBox => ty::vstore_box,
        ast::ExprVstoreSlice | ast::ExprVstoreMutSlice => {
            match e.node {
                ast::ExprLit(..) |
                ast::ExprVec([], _) => {
                    // string literals and *empty slices* live in static memory
                    ty::vstore_slice(ty::ReStatic)
                }
                ast::ExprRepeat(..) |
                ast::ExprVec(..) => {
                    // vector literals are temporaries on the stack
                    match fcx.tcx().region_maps.temporary_scope(e.id) {
                        Some(scope) => {
                            let r = ty::ReScope(scope);
                            ty::vstore_slice(r)
                        }
                        None => {
                            // this slice occurs in a static somewhere
                            ty::vstore_slice(ty::ReStatic)
                        }
                    }
                }
                _ => {
                    fcx.ccx.tcx.sess.span_bug(
                        e.span, format!("vstore with unexpected contents"))
                }
            }
        }
    }
}

// Returns true if b contains a break that can exit from b
pub fn may_break(cx: ty::ctxt, id: ast::NodeId, b: ast::P<ast::Block>) -> bool {
    // First: is there an unlabeled break immediately
    // inside the loop?
    (loop_query(b, |e| {
        match *e {
            ast::ExprBreak(_) => true,
            _ => false
        }
    })) ||
   // Second: is there a labeled break with label
   // <id> nested anywhere inside the loop?
    (block_query(b, |e| {
        match e.node {
            ast::ExprBreak(Some(_)) => {
                let def_map = cx.def_map.borrow();
                match def_map.get().find(&e.id) {
                    Some(&ast::DefLabel(loop_id)) if id == loop_id => true,
                    _ => false,
                }
            }
            _ => false
        }}))
}

pub fn check_bounds_are_used(ccx: @CrateCtxt,
                             span: Span,
                             tps: &OptVec<ast::TyParam>,
                             ty: ty::t) {
    debug!("check_bounds_are_used(n_tps={}, ty={})",
           tps.len(), ppaux::ty_to_str(ccx.tcx, ty));

    // make a vector of booleans initially false, set to true when used
    if tps.len() == 0u { return; }
    let mut tps_used = vec::from_elem(tps.len(), false);

    ty::walk_ty(ty, |t| {
            match ty::get(t).sty {
                ty::ty_param(param_ty {idx, ..}) => {
                    debug!("Found use of ty param \\#{}", idx);
                    tps_used[idx] = true;
                }
                _ => ()
            }
        });

    for (i, b) in tps_used.iter().enumerate() {
        if !*b {
            ccx.tcx.sess.span_err(
                span, format!("type parameter `{}` is unused",
                           ccx.tcx.sess.str_of(tps.get(i).ident)));
        }
    }
}

pub fn check_intrinsic_type(ccx: @CrateCtxt, it: &ast::ForeignItem) {
    fn param(ccx: @CrateCtxt, n: uint) -> ty::t {
        ty::mk_param(ccx.tcx, n, local_def(0))
    }

    let tcx = ccx.tcx;
    let nm = ccx.tcx.sess.str_of(it.ident);
    let name = nm.as_slice();
    let (n_tps, inputs, output) = if name.starts_with("atomic_") {
        let split : ~[&str] = name.split('_').collect();
        assert!(split.len() >= 2, "Atomic intrinsic not correct format");

        //We only care about the operation here
        match split[1] {
            "cxchg" => (0, ~[ty::mk_mut_rptr(tcx,
                                             ty::ReLateBound(it.id, ty::BrAnon(0)),
                                             ty::mk_int()),
                        ty::mk_int(),
                        ty::mk_int()
                        ], ty::mk_int()),
            "load" => (0,
               ~[
                  ty::mk_imm_rptr(tcx, ty::ReLateBound(it.id, ty::BrAnon(0)), ty::mk_int())
               ],
              ty::mk_int()),
            "store" => (0,
               ~[
                  ty::mk_mut_rptr(tcx, ty::ReLateBound(it.id, ty::BrAnon(0)), ty::mk_int()),
                  ty::mk_int()
               ],
               ty::mk_nil()),

            "xchg" | "xadd" | "xsub" | "and"  | "nand" | "or"   | "xor"  | "max"  |
            "min"  | "umax" | "umin" => {
                (0, ~[ty::mk_mut_rptr(tcx,
                                      ty::ReLateBound(it.id, ty::BrAnon(0)),
                                      ty::mk_int()), ty::mk_int() ], ty::mk_int())
            }
            "fence" => {
                (0, ~[], ty::mk_nil())
            }
            op => {
                tcx.sess.span_err(it.span,
                                  format!("unrecognized atomic operation function: `{}`",
                                       op));
                return;
            }
        }

    } else {
        match name {
            "abort" => (0, ~[], ty::mk_bot()),
            "breakpoint" => (0, ~[], ty::mk_nil()),
            "size_of" |
            "pref_align_of" | "min_align_of" => (1u, ~[], ty::mk_uint()),
            "init" => (1u, ~[], param(ccx, 0u)),
            "uninit" => (1u, ~[], param(ccx, 0u)),
            "forget" => (1u, ~[ param(ccx, 0) ], ty::mk_nil()),
            "transmute" => (2, ~[ param(ccx, 0) ], param(ccx, 1)),
            "move_val_init" => {
                (1u,
                 ~[
                    ty::mk_mut_rptr(tcx, ty::ReLateBound(it.id, ty::BrAnon(0)), param(ccx, 0)),
                    param(ccx, 0u)
                  ],
               ty::mk_nil())
            }
            "needs_drop" => (1u, ~[], ty::mk_bool()),
            "owns_managed" => (1u, ~[], ty::mk_bool()),
            "atomic_xchg"     | "atomic_xadd"     | "atomic_xsub"     |
            "atomic_xchg_acq" | "atomic_xadd_acq" | "atomic_xsub_acq" |
            "atomic_xchg_rel" | "atomic_xadd_rel" | "atomic_xsub_rel" => {
              (0,
               ~[
                  ty::mk_mut_rptr(tcx, ty::ReLateBound(it.id, ty::BrAnon(0)), ty::mk_int()),
                  ty::mk_int()
               ],
               ty::mk_int())
            }

            "get_tydesc" => {
              let tydesc_ty = match ty::get_tydesc_ty(ccx.tcx) {
                  Ok(t) => t,
                  Err(s) => { tcx.sess.span_fatal(it.span, s); }
              };
              let td_ptr = ty::mk_ptr(ccx.tcx, ty::mt {
                  ty: tydesc_ty,
                  mutbl: ast::MutImmutable
              });
              (1u, ~[], td_ptr)
            }
            "type_id" => {
                let langid = ccx.tcx.lang_items.require(TypeIdLangItem);
                match langid {
                    Ok(did) => (1u, ~[], ty::mk_struct(ccx.tcx, did, substs {
                                                 self_ty: None,
                                                 tps: ~[],
                                                 regions: ty::NonerasedRegions(opt_vec::Empty)
                                                 }) ),
                    Err(msg) => { tcx.sess.span_fatal(it.span, msg); }
                }
            },
            "visit_tydesc" => {
              let tydesc_ty = match ty::get_tydesc_ty(ccx.tcx) {
                  Ok(t) => t,
                  Err(s) => { tcx.sess.span_fatal(it.span, s); }
              };
              let region = ty::ReLateBound(it.id, ty::BrAnon(0));
              let visitor_object_ty = match ty::visitor_object_ty(tcx, region) {
                  Ok((_, vot)) => vot,
                  Err(s) => { tcx.sess.span_fatal(it.span, s); }
              };

              let td_ptr = ty::mk_ptr(ccx.tcx, ty::mt {
                  ty: tydesc_ty,
                  mutbl: ast::MutImmutable
              });
              (0, ~[ td_ptr, visitor_object_ty ], ty::mk_nil())
            }
            "morestack_addr" => {
              (0u, ~[], ty::mk_nil_ptr(ccx.tcx))
            }
            "offset" => {
              (1,
               ~[
                  ty::mk_ptr(tcx, ty::mt {
                      ty: param(ccx, 0),
                      mutbl: ast::MutImmutable
                  }),
                  ty::mk_int()
               ],
               ty::mk_ptr(tcx, ty::mt {
                   ty: param(ccx, 0),
                   mutbl: ast::MutImmutable
               }))
            }
            "copy_nonoverlapping_memory" => {
              (1,
               ~[
                  ty::mk_ptr(tcx, ty::mt {
                      ty: param(ccx, 0),
                      mutbl: ast::MutMutable
                  }),
                  ty::mk_ptr(tcx, ty::mt {
                      ty: param(ccx, 0),
                      mutbl: ast::MutImmutable
                  }),
                  ty::mk_uint()
               ],
               ty::mk_nil())
            }
            "copy_memory" => {
              (1,
               ~[
                  ty::mk_ptr(tcx, ty::mt {
                      ty: param(ccx, 0),
                      mutbl: ast::MutMutable
                  }),
                  ty::mk_ptr(tcx, ty::mt {
                      ty: param(ccx, 0),
                      mutbl: ast::MutImmutable
                  }),
                  ty::mk_uint()
               ],
               ty::mk_nil())
            }
            "set_memory" => {
              (1,
               ~[
                  ty::mk_ptr(tcx, ty::mt {
                      ty: param(ccx, 0),
                      mutbl: ast::MutMutable
                  }),
                  ty::mk_u8(),
                  ty::mk_uint()
               ],
               ty::mk_nil())
            }
            "sqrtf32" => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "sqrtf64" => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "powif32" => {
               (0,
                ~[ ty::mk_f32(), ty::mk_i32() ],
                ty::mk_f32())
            }
            "powif64" => {
               (0,
                ~[ ty::mk_f64(), ty::mk_i32() ],
                ty::mk_f64())
            }
            "sinf32" => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "sinf64" => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "cosf32" => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "cosf64" => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "powf32" => {
               (0,
                ~[ ty::mk_f32(), ty::mk_f32() ],
                ty::mk_f32())
            }
            "powf64" => {
               (0,
                ~[ ty::mk_f64(), ty::mk_f64() ],
                ty::mk_f64())
            }
            "expf32"   => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "expf64"   => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "exp2f32"  => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "exp2f64"  => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "logf32"   => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "logf64"   => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "log10f32" => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "log10f64" => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "log2f32"  => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "log2f64"  => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "fmaf32" => {
                (0,
                 ~[ ty::mk_f32(), ty::mk_f32(), ty::mk_f32() ],
                 ty::mk_f32())
            }
            "fmaf64" => {
                (0,
                 ~[ ty::mk_f64(), ty::mk_f64(), ty::mk_f64() ],
                 ty::mk_f64())
            }
            "fabsf32"      => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "fabsf64"      => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "copysignf32"  => (0, ~[ ty::mk_f32(), ty::mk_f32() ], ty::mk_f32()),
            "copysignf64"  => (0, ~[ ty::mk_f64(), ty::mk_f64() ], ty::mk_f64()),
            "floorf32"     => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "floorf64"     => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "ceilf32"      => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "ceilf64"      => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "truncf32"     => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "truncf64"     => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "rintf32"      => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "rintf64"      => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "nearbyintf32" => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "nearbyintf64" => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "roundf32"     => (0, ~[ ty::mk_f32() ], ty::mk_f32()),
            "roundf64"     => (0, ~[ ty::mk_f64() ], ty::mk_f64()),
            "ctpop8"       => (0, ~[ ty::mk_i8()  ], ty::mk_i8()),
            "ctpop16"      => (0, ~[ ty::mk_i16() ], ty::mk_i16()),
            "ctpop32"      => (0, ~[ ty::mk_i32() ], ty::mk_i32()),
            "ctpop64"      => (0, ~[ ty::mk_i64() ], ty::mk_i64()),
            "ctlz8"        => (0, ~[ ty::mk_i8()  ], ty::mk_i8()),
            "ctlz16"       => (0, ~[ ty::mk_i16() ], ty::mk_i16()),
            "ctlz32"       => (0, ~[ ty::mk_i32() ], ty::mk_i32()),
            "ctlz64"       => (0, ~[ ty::mk_i64() ], ty::mk_i64()),
            "cttz8"        => (0, ~[ ty::mk_i8()  ], ty::mk_i8()),
            "cttz16"       => (0, ~[ ty::mk_i16() ], ty::mk_i16()),
            "cttz32"       => (0, ~[ ty::mk_i32() ], ty::mk_i32()),
            "cttz64"       => (0, ~[ ty::mk_i64() ], ty::mk_i64()),
            "bswap16"      => (0, ~[ ty::mk_i16() ], ty::mk_i16()),
            "bswap32"      => (0, ~[ ty::mk_i32() ], ty::mk_i32()),
            "bswap64"      => (0, ~[ ty::mk_i64() ], ty::mk_i64()),

            "volatile_load" =>
                (1, ~[ ty::mk_imm_ptr(tcx, param(ccx, 0)) ], param(ccx, 0)),
            "volatile_store" =>
                (1, ~[ ty::mk_mut_ptr(tcx, param(ccx, 0)), param(ccx, 0) ], ty::mk_nil()),

            "i8_add_with_overflow" | "i8_sub_with_overflow" | "i8_mul_with_overflow" =>
                (0, ~[ty::mk_i8(), ty::mk_i8()],
                ty::mk_tup(tcx, ~[ty::mk_i8(), ty::mk_bool()])),

            "i16_add_with_overflow" | "i16_sub_with_overflow" | "i16_mul_with_overflow" =>
                (0, ~[ty::mk_i16(), ty::mk_i16()],
                ty::mk_tup(tcx, ~[ty::mk_i16(), ty::mk_bool()])),

            "i32_add_with_overflow" | "i32_sub_with_overflow" | "i32_mul_with_overflow" =>
                (0, ~[ty::mk_i32(), ty::mk_i32()],
                ty::mk_tup(tcx, ~[ty::mk_i32(), ty::mk_bool()])),

            "i64_add_with_overflow" | "i64_sub_with_overflow" | "i64_mul_with_overflow" =>
                (0, ~[ty::mk_i64(), ty::mk_i64()],
                ty::mk_tup(tcx, ~[ty::mk_i64(), ty::mk_bool()])),

            "u8_add_with_overflow" | "u8_sub_with_overflow" | "u8_mul_with_overflow" =>
                (0, ~[ty::mk_u8(), ty::mk_u8()],
                ty::mk_tup(tcx, ~[ty::mk_u8(), ty::mk_bool()])),

            "u16_add_with_overflow" | "u16_sub_with_overflow" | "u16_mul_with_overflow" =>
                (0, ~[ty::mk_u16(), ty::mk_u16()],
                ty::mk_tup(tcx, ~[ty::mk_u16(), ty::mk_bool()])),

            "u32_add_with_overflow" | "u32_sub_with_overflow" | "u32_mul_with_overflow"=>
                (0, ~[ty::mk_u32(), ty::mk_u32()],
                ty::mk_tup(tcx, ~[ty::mk_u32(), ty::mk_bool()])),

            "u64_add_with_overflow" | "u64_sub_with_overflow"  | "u64_mul_with_overflow" =>
                (0, ~[ty::mk_u64(), ty::mk_u64()],
                ty::mk_tup(tcx, ~[ty::mk_u64(), ty::mk_bool()])),

            ref other => {
                tcx.sess.span_err(it.span,
                                  format!("unrecognized intrinsic function: `{}`",
                                       *other));
                return;
            }
        }
    };
    let fty = ty::mk_bare_fn(tcx, ty::BareFnTy {
        purity: ast::UnsafeFn,
        abis: AbiSet::Intrinsic(),
        sig: FnSig {binder_id: it.id,
                    inputs: inputs,
                    output: output,
                    variadic: false}
    });
    let i_ty = ty::lookup_item_type(ccx.tcx, local_def(it.id));
    let i_n_tps = i_ty.generics.type_param_defs.len();
    if i_n_tps != n_tps {
        tcx.sess.span_err(it.span, format!("intrinsic has wrong number \
                                         of type parameters: found {}, \
                                         expected {}", i_n_tps, n_tps));
    } else {
        require_same_types(
            tcx, None, false, it.span, i_ty.ty, fty,
            || format!("intrinsic has wrong type: \
                      expected `{}`",
                     ppaux::ty_to_str(ccx.tcx, fty)));
    }
}

