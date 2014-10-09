// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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


use middle::const_eval;
use middle::def;
use middle::lang_items::IteratorItem;
use middle::mem_categorization::McResult;
use middle::mem_categorization;
use middle::pat_util::pat_id_map;
use middle::pat_util;
use middle::subst;
use middle::subst::{Subst, Substs, VecPerParamSpace, ParamSpace};
use middle::traits;
use middle::ty::{FnSig, VariantInfo};
use middle::ty::{Polytype};
use middle::ty::{Disr, ParamTy, ParameterEnvironment};
use middle::ty;
use middle::ty_fold::TypeFolder;
use middle::typeck::astconv::AstConv;
use middle::typeck::astconv::{ast_region_to_region, ast_ty_to_ty};
use middle::typeck::astconv;
use middle::typeck::check::_match::pat_ctxt;
use middle::typeck::check::method::{AutoderefReceiver};
use middle::typeck::check::method::{AutoderefReceiverFlag};
use middle::typeck::check::method::{CheckTraitsAndInherentMethods};
use middle::typeck::check::method::{DontAutoderefReceiver};
use middle::typeck::check::method::{IgnoreStaticMethods, ReportStaticMethods};
use middle::typeck::check::regionmanip::replace_late_bound_regions_in_fn_sig;
use middle::typeck::CrateCtxt;
use middle::typeck::infer::{resolve_type, force_tvar};
use middle::typeck::infer;
use middle::typeck::rscope::RegionScope;
use middle::typeck::{lookup_def_ccx};
use middle::typeck::no_params;
use middle::typeck::{require_same_types};
use middle::typeck::{MethodCall, MethodCallee, MethodMap, ObjectCastMap};
use middle::typeck::{TypeAndSubsts};
use middle::typeck;
use middle::lang_items::TypeIdLangItem;
use lint;
use util::common::{block_query, indenter, loop_query};
use util::ppaux;
use util::ppaux::{UserString, Repr};
use util::nodemap::{DefIdMap, FnvHashMap, NodeMap};

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::collections::hashmap::{Occupied, Vacant};
use std::mem::replace;
use std::rc::Rc;
use syntax::abi;
use syntax::ast::{ProvidedMethod, RequiredMethod, TypeTraitItem};
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::{local_def, PostExpansionMethod};
use syntax::ast_util;
use syntax::attr;
use syntax::codemap::Span;
use syntax::codemap;
use syntax::owned_slice::OwnedSlice;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::visit;
use syntax::visit::Visitor;
use syntax;

pub mod _match;
pub mod vtable2; // New trait code
pub mod writeback;
pub mod regionmanip;
pub mod regionck;
pub mod demand;
pub mod method;
pub mod wf;

/// Fields that are part of a `FnCtxt` which are inherited by
/// closures defined within the function.  For example:
///
///     fn foo() {
///         bar(proc() { ... })
///     }
///
/// Here, the function `foo()` and the closure passed to
/// `bar()` will each have their own `FnCtxt`, but they will
/// share the inherited fields.
pub struct Inherited<'a, 'tcx: 'a> {
    infcx: infer::InferCtxt<'a, 'tcx>,
    locals: RefCell<NodeMap<ty::t>>,
    param_env: ty::ParameterEnvironment,

    // Temporary tables:
    node_types: RefCell<NodeMap<ty::t>>,
    item_substs: RefCell<NodeMap<ty::ItemSubsts>>,
    adjustments: RefCell<NodeMap<ty::AutoAdjustment>>,
    method_map: MethodMap,
    upvar_borrow_map: RefCell<ty::UpvarBorrowMap>,
    unboxed_closures: RefCell<DefIdMap<ty::UnboxedClosure>>,
    object_cast_map: ObjectCastMap,

    // A mapping from each fn's id to its signature, with all bound
    // regions replaced with free ones. Unlike the other tables, this
    // one is never copied into the tcx: it is only used by regionck.
    fn_sig_map: RefCell<NodeMap<Vec<ty::t>>>,

    // A set of constraints that regionck must validate. Each
    // constraint has the form `T:'a`, meaning "some type `T` must
    // outlive the lifetime 'a". These constraints derive from
    // instantiated type parameters. So if you had a struct defined
    // like
    //
    //     struct Foo<T:'static> { ... }
    //
    // then in some expression `let x = Foo { ... }` it will
    // instantiate the type parameter `T` with a fresh type `$0`. At
    // the same time, it will record a region obligation of
    // `$0:'static`. This will get checked later by regionck. (We
    // can't generally check these things right away because we have
    // to wait until types are resolved.)
    //
    // These are stored in a map keyed to the id of the innermost
    // enclosing fn body / static initializer expression. This is
    // because the location where the obligation was incurred can be
    // relevant with respect to which sublifetime assumptions are in
    // place. The reason that we store under the fn-id, and not
    // something more fine-grained, is so that it is easier for
    // regionck to be sure that it has found *all* the region
    // obligations (otherwise, it's easy to fail to walk to a
    // particular node-id).
    region_obligations: RefCell<NodeMap<Vec<RegionObligation>>>,

    // Tracks trait obligations incurred during this function body.
    fulfillment_cx: RefCell<traits::FulfillmentContext>,
}

struct RegionObligation {
    sub_region: ty::Region,
    sup_type: ty::t,
    origin: infer::SubregionOrigin,
}

/// When type-checking an expression, we propagate downward
/// whatever type hint we are able in the form of an `Expectation`.
enum Expectation {
    /// We know nothing about what type this expression should have.
    NoExpectation,

    /// This expression should have the type given (or some subtype)
    ExpectHasType(ty::t),

    /// This expression will be cast to the `ty::t`
    ExpectCastableToType(ty::t),
}

#[deriving(Clone)]
pub struct FnStyleState {
    pub def: ast::NodeId,
    pub fn_style: ast::FnStyle,
    from_fn: bool
}

impl FnStyleState {
    pub fn function(fn_style: ast::FnStyle, def: ast::NodeId) -> FnStyleState {
        FnStyleState { def: def, fn_style: fn_style, from_fn: true }
    }

    pub fn recurse(&mut self, blk: &ast::Block) -> FnStyleState {
        match self.fn_style {
            // If this unsafe, then if the outer function was already marked as
            // unsafe we shouldn't attribute the unsafe'ness to the block. This
            // way the block can be warned about instead of ignoring this
            // extraneous block (functions are never warned about).
            ast::UnsafeFn if self.from_fn => *self,

            fn_style => {
                let (fn_style, def) = match blk.rules {
                    ast::UnsafeBlock(..) => (ast::UnsafeFn, blk.id),
                    ast::DefaultBlock => (fn_style, self.def),
                };
                FnStyleState{ def: def,
                             fn_style: fn_style,
                             from_fn: false }
            }
        }
    }
}

/// Whether `check_binop` is part of an assignment or not.
/// Used to know whether we allow user overloads and to print
/// better messages on error.
#[deriving(PartialEq)]
enum IsBinopAssignment{
    SimpleBinop,
    BinopAssignment,
}

#[deriving(Clone)]
pub struct FnCtxt<'a, 'tcx: 'a> {
    body_id: ast::NodeId,

    // This flag is set to true if, during the writeback phase, we encounter
    // a type error in this function.
    writeback_errors: Cell<bool>,

    // Number of errors that had been reported when we started
    // checking this function. On exit, if we find that *more* errors
    // have been reported, we will skip regionck and other work that
    // expects the types within the function to be consistent.
    err_count_on_creation: uint,

    ret_ty: ty::t,

    ps: RefCell<FnStyleState>,

    inh: &'a Inherited<'a, 'tcx>,

    ccx: &'a CrateCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> mem_categorization::Typer<'tcx> for FnCtxt<'a, 'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        self.ccx.tcx
    }
    fn node_ty(&self, id: ast::NodeId) -> McResult<ty::t> {
        Ok(self.node_ty(id))
    }
    fn node_method_ty(&self, method_call: typeck::MethodCall)
                      -> Option<ty::t> {
        self.inh.method_map.borrow().find(&method_call).map(|m| m.ty)
    }
    fn adjustments<'a>(&'a self) -> &'a RefCell<NodeMap<ty::AutoAdjustment>> {
        &self.inh.adjustments
    }
    fn is_method_call(&self, id: ast::NodeId) -> bool {
        self.inh.method_map.borrow().contains_key(&typeck::MethodCall::expr(id))
    }
    fn temporary_scope(&self, rvalue_id: ast::NodeId) -> Option<ast::NodeId> {
        self.tcx().temporary_scope(rvalue_id)
    }
    fn upvar_borrow(&self, upvar_id: ty::UpvarId) -> ty::UpvarBorrow {
        self.inh.upvar_borrow_map.borrow().get_copy(&upvar_id)
    }
    fn capture_mode(&self, closure_expr_id: ast::NodeId)
                    -> ast::CaptureClause {
        self.ccx.tcx.capture_mode(closure_expr_id)
    }
    fn unboxed_closures<'a>(&'a self) -> &'a RefCell<DefIdMap<ty::UnboxedClosure>> {
        &self.inh.unboxed_closures
    }
}

impl<'a, 'tcx> Inherited<'a, 'tcx> {
    fn new(tcx: &'a ty::ctxt<'tcx>,
           param_env: ty::ParameterEnvironment)
           -> Inherited<'a, 'tcx> {
        Inherited {
            infcx: infer::new_infer_ctxt(tcx),
            locals: RefCell::new(NodeMap::new()),
            param_env: param_env,
            node_types: RefCell::new(NodeMap::new()),
            item_substs: RefCell::new(NodeMap::new()),
            adjustments: RefCell::new(NodeMap::new()),
            method_map: RefCell::new(FnvHashMap::new()),
            object_cast_map: RefCell::new(NodeMap::new()),
            upvar_borrow_map: RefCell::new(HashMap::new()),
            unboxed_closures: RefCell::new(DefIdMap::new()),
            fn_sig_map: RefCell::new(NodeMap::new()),
            region_obligations: RefCell::new(NodeMap::new()),
            fulfillment_cx: RefCell::new(traits::FulfillmentContext::new()),
        }
    }
}

// Used by check_const and check_enum_variants
pub fn blank_fn_ctxt<'a, 'tcx>(ccx: &'a CrateCtxt<'a, 'tcx>,
                               inh: &'a Inherited<'a, 'tcx>,
                               rty: ty::t,
                               body_id: ast::NodeId)
                               -> FnCtxt<'a, 'tcx> {
    FnCtxt {
        body_id: body_id,
        writeback_errors: Cell::new(false),
        err_count_on_creation: ccx.tcx.sess.err_count(),
        ret_ty: rty,
        ps: RefCell::new(FnStyleState::function(ast::NormalFn, 0)),
        inh: inh,
        ccx: ccx
    }
}

fn static_inherited_fields<'a, 'tcx>(ccx: &'a CrateCtxt<'a, 'tcx>)
                                    -> Inherited<'a, 'tcx> {
    // It's kind of a kludge to manufacture a fake function context
    // and statement context, but we might as well do write the code only once
    let param_env = ty::empty_parameter_environment();
    Inherited::new(ccx.tcx, param_env)
}

struct CheckItemTypesVisitor<'a, 'tcx: 'a> { ccx: &'a CrateCtxt<'a, 'tcx> }

impl<'a, 'tcx, 'v> Visitor<'v> for CheckItemTypesVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &ast::Item) {
        check_item(self.ccx, i);
        visit::walk_item(self, i);
    }
}

pub fn check_item_types(ccx: &CrateCtxt) {
    let krate = ccx.tcx.map.krate();
    let mut visit = wf::CheckTypeWellFormedVisitor::new(ccx);
    visit::walk_crate(&mut visit, krate);

    // If types are not well-formed, it leads to all manner of errors
    // downstream, so stop reporting errors at this point.
    ccx.tcx.sess.abort_if_errors();

    let mut visit = CheckItemTypesVisitor { ccx: ccx };
    visit::walk_crate(&mut visit, krate);

    ccx.tcx.sess.abort_if_errors();
}

fn check_bare_fn(ccx: &CrateCtxt,
                 decl: &ast::FnDecl,
                 body: &ast::Block,
                 id: ast::NodeId,
                 fty: ty::t,
                 param_env: ty::ParameterEnvironment) {
    // Compute the fty from point of view of inside fn
    // (replace any type-scheme with a type)
    let fty = fty.subst(ccx.tcx, &param_env.free_substs);

    match ty::get(fty).sty {
        ty::ty_bare_fn(ref fn_ty) => {
            let inh = Inherited::new(ccx.tcx, param_env);
            let fcx = check_fn(ccx, fn_ty.fn_style, id, &fn_ty.sig,
                               decl, id, body, &inh);

            vtable2::select_all_fcx_obligations_or_error(&fcx);
            regionck::regionck_fn(&fcx, id, body);
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
    fn assign(&mut self, _span: Span, nid: ast::NodeId, ty_opt: Option<ty::t>) -> ty::t {
        match ty_opt {
            None => {
                // infer the variable's type
                let var_id = self.fcx.infcx().next_ty_var_id();
                let var_ty = ty::mk_var(self.fcx.tcx(), var_id);
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

impl<'a, 'tcx, 'v> Visitor<'v> for GatherLocalsVisitor<'a, 'tcx> {
    // Add explicitly-declared locals.
    fn visit_local(&mut self, local: &ast::Local) {
        let o_ty = match local.ty.node {
            ast::TyInfer => None,
            _ => Some(self.fcx.to_ty(&*local.ty))
        };
        self.assign(local.span, local.id, o_ty);
        debug!("Local variable {} is assigned type {}",
               self.fcx.pat_to_string(&*local.pat),
               self.fcx.infcx().ty_to_string(
                   self.fcx.inh.locals.borrow().get_copy(&local.id)));
        visit::walk_local(self, local);
    }

    // Add pattern bindings.
    fn visit_pat(&mut self, p: &ast::Pat) {
        match p.node {
            ast::PatIdent(_, ref path1, _)
                if pat_util::pat_is_binding(&self.fcx.ccx.tcx.def_map, p) => {
                    let var_ty = self.assign(p.span, p.id, None);

                    self.fcx.require_type_is_sized(var_ty, p.span,
                                                   traits::VariableType(p.id));

                    debug!("Pattern binding {} is assigned to {} with type {}",
                           token::get_ident(path1.node),
                           self.fcx.infcx().ty_to_string(
                               self.fcx.inh.locals.borrow().get_copy(&p.id)),
                           var_ty.repr(self.fcx.tcx()));
                }
            _ => {}
        }
        visit::walk_pat(self, p);
    }

    fn visit_block(&mut self, b: &ast::Block) {
        // non-obvious: the `blk` variable maps to region lb, so
        // we have to keep this up-to-date.  This
        // is... unfortunate.  It'd be nice to not need this.
        visit::walk_block(self, b);
    }

    // Since an expr occurs as part of the type fixed size arrays we
    // need to record the type for that node
    fn visit_ty(&mut self, t: &ast::Ty) {
        match t.node {
            ast::TyFixedLengthVec(ref ty, ref count_expr) => {
                self.visit_ty(&**ty);
                check_expr_with_hint(self.fcx, &**count_expr, ty::mk_uint());
            }
            _ => visit::walk_ty(self, t)
        }
    }

    // Don't descend into fns and items
    fn visit_fn(&mut self, _: visit::FnKind<'v>, _: &'v ast::FnDecl,
                _: &'v ast::Block, _: Span, _: ast::NodeId) { }
    fn visit_item(&mut self, _: &ast::Item) { }

}

fn check_fn<'a, 'tcx>(ccx: &'a CrateCtxt<'a, 'tcx>,
                      fn_style: ast::FnStyle,
                      fn_style_id: ast::NodeId,
                      fn_sig: &ty::FnSig,
                      decl: &ast::FnDecl,
                      fn_id: ast::NodeId,
                      body: &ast::Block,
                      inherited: &'a Inherited<'a, 'tcx>)
                      -> FnCtxt<'a, 'tcx> {
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

    // First, we have to replace any bound regions in the fn type with free ones.
    // The free region references will be bound the node_id of the body block.
    let (_, fn_sig) = replace_late_bound_regions_in_fn_sig(tcx, fn_sig, |br| {
        ty::ReFree(ty::FreeRegion {scope_id: body.id, bound_region: br})
    });

    let arg_tys = fn_sig.inputs.as_slice();
    let ret_ty = fn_sig.output;

    debug!("check_fn(arg_tys={}, ret_ty={}, fn_id={})",
           arg_tys.repr(tcx),
           ret_ty.repr(tcx),
           fn_id);

    // Create the function context.  This is either derived from scratch or,
    // in the case of function expressions, based on the outer context.
    let fcx = FnCtxt {
        body_id: body.id,
        writeback_errors: Cell::new(false),
        err_count_on_creation: err_count_on_creation,
        ret_ty: ret_ty,
        ps: RefCell::new(FnStyleState::function(fn_style, fn_style_id)),
        inh: inherited,
        ccx: ccx
    };

    // Remember return type so that regionck can access it later.
    let fn_sig_tys: Vec<ty::t> =
        arg_tys.iter()
        .chain([ret_ty].iter())
        .map(|&ty| ty)
        .collect();
    debug!("fn-sig-map: fn_id={} fn_sig_tys={}",
           fn_id,
           fn_sig_tys.repr(tcx));
    inherited.fn_sig_map
        .borrow_mut()
        .insert(fn_id, fn_sig_tys);

    {
        let mut visit = GatherLocalsVisitor { fcx: &fcx, };

        // Add formal parameters.
        for (arg_ty, input) in arg_tys.iter().zip(decl.inputs.iter()) {
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

    check_block_with_expected(&fcx, body, ExpectHasType(ret_ty));

    for (input, arg) in decl.inputs.iter().zip(arg_tys.iter()) {
        fcx.write_ty(input.id, *arg);
    }

    fcx
}

fn span_for_field(tcx: &ty::ctxt, field: &ty::field_ty, struct_id: ast::DefId) -> Span {
    assert!(field.id.krate == ast::LOCAL_CRATE);
    let item = match tcx.map.find(struct_id.node) {
        Some(ast_map::NodeItem(item)) => item,
        None => fail!("node not in ast map: {}", struct_id.node),
        _ => fail!("expected item, found {}", tcx.map.node_to_string(struct_id.node))
    };

    match item.node {
        ast::ItemStruct(ref struct_def, _) => {
            match struct_def.fields.iter().find(|f| match f.node.kind {
                ast::NamedField(ident, _) => ident.name == field.name,
                _ => false,
            }) {
                Some(f) => f.span,
                None => {
                    tcx.sess
                       .bug(format!("Could not find field {}",
                                    token::get_name(field.name)).as_slice())
                }
            }
        },
        _ => tcx.sess.bug("Field found outside of a struct?"),
    }
}

// Check struct fields are uniquely named wrt parents.
fn check_for_field_shadowing(tcx: &ty::ctxt,
                             id: ast::DefId) {
    let struct_fields = tcx.struct_fields.borrow();
    let fields = struct_fields.get(&id);

    let superstructs = tcx.superstructs.borrow();
    let super_struct = superstructs.get(&id);
    match *super_struct {
        Some(parent_id) => {
            let super_fields = ty::lookup_struct_fields(tcx, parent_id);
            for f in fields.iter() {
                match super_fields.iter().find(|sf| f.name == sf.name) {
                    Some(prev_field) => {
                        span_err!(tcx.sess, span_for_field(tcx, f, id), E0041,
                            "field `{}` hides field declared in super-struct",
                            token::get_name(f.name));
                        span_note!(tcx.sess, span_for_field(tcx, prev_field, parent_id),
                            "previously declared here");
                    },
                    None => {}
                }
            }
        },
        None => {}
    }
}

pub fn check_struct(ccx: &CrateCtxt, id: ast::NodeId, span: Span) {
    let tcx = ccx.tcx;

    check_representable(tcx, span, id, "struct");
    check_instantiable(tcx, span, id);

    // Check there are no overlapping fields in super-structs
    check_for_field_shadowing(tcx, local_def(id));

    if ty::lookup_simd(tcx, local_def(id)) {
        check_simd(tcx, span, id);
    }
}

pub fn check_item(ccx: &CrateCtxt, it: &ast::Item) {
    debug!("check_item(it.id={}, it.ident={})",
           it.id,
           ty::item_path_str(ccx.tcx, local_def(it.id)));
    let _indenter = indenter();

    match it.node {
      ast::ItemStatic(_, _, ref e) |
      ast::ItemConst(_, ref e) => check_const(ccx, it.span, &**e, it.id),
      ast::ItemEnum(ref enum_definition, _) => {
        check_enum_variants(ccx,
                            it.span,
                            enum_definition.variants.as_slice(),
                            it.id);
      }
      ast::ItemFn(ref decl, _, _, _, ref body) => {
        let fn_pty = ty::lookup_item_type(ccx.tcx, ast_util::local_def(it.id));
        let param_env = ParameterEnvironment::for_item(ccx.tcx, it.id);
        check_bare_fn(ccx, &**decl, &**body, it.id, fn_pty.ty, param_env);
      }
      ast::ItemImpl(_, ref opt_trait_ref, _, ref impl_items) => {
        debug!("ItemImpl {} with id {}", token::get_ident(it.ident), it.id);

        let impl_pty = ty::lookup_item_type(ccx.tcx, ast_util::local_def(it.id));

        match *opt_trait_ref {
            Some(ref ast_trait_ref) => {
                let impl_trait_ref =
                    ty::node_id_to_trait_ref(ccx.tcx, ast_trait_ref.ref_id);
                check_impl_items_against_trait(ccx,
                                               it.span,
                                               ast_trait_ref,
                                               &*impl_trait_ref,
                                               impl_items.as_slice());
            }
            None => { }
        }

        for impl_item in impl_items.iter() {
            match *impl_item {
                ast::MethodImplItem(ref m) => {
                    check_method_body(ccx, &impl_pty.generics, &**m);
                }
                ast::TypeImplItem(_) => {
                    // Nothing to do here.
                }
            }
        }

      }
      ast::ItemTrait(_, _, _, ref trait_methods) => {
        let trait_def = ty::lookup_trait_def(ccx.tcx, local_def(it.id));
        for trait_method in trait_methods.iter() {
            match *trait_method {
                RequiredMethod(..) => {
                    // Nothing to do, since required methods don't have
                    // bodies to check.
                }
                ProvidedMethod(ref m) => {
                    check_method_body(ccx, &trait_def.generics, &**m);
                }
                TypeTraitItem(_) => {
                    // Nothing to do.
                }
            }
        }
      }
      ast::ItemStruct(..) => {
        check_struct(ccx, it.id, it.span);
      }
      ast::ItemTy(ref t, ref generics) => {
        let pty_ty = ty::node_id_to_type(ccx.tcx, it.id);
        check_bounds_are_used(ccx, t.span, &generics.ty_params, pty_ty);
      }
      ast::ItemForeignMod(ref m) => {
        if m.abi == abi::RustIntrinsic {
            for item in m.items.iter() {
                check_intrinsic_type(ccx, &**item);
            }
        } else {
            for item in m.items.iter() {
                let pty = ty::lookup_item_type(ccx.tcx, local_def(item.id));
                if !pty.generics.types.is_empty() {
                    span_err!(ccx.tcx.sess, item.span, E0044,
                        "foreign items may not have type parameters");
                }

                match item.node {
                    ast::ForeignItemFn(ref fn_decl, _) => {
                        if fn_decl.variadic && m.abi != abi::C {
                            span_err!(ccx.tcx.sess, item.span, E0045,
                                "variadic function must have C calling convention");
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

fn check_method_body(ccx: &CrateCtxt,
                     item_generics: &ty::Generics,
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

    debug!("check_method_body(item_generics={}, method.id={})",
            item_generics.repr(ccx.tcx),
            method.id);
    let param_env = ParameterEnvironment::for_item(ccx.tcx, method.id);

    let fty = ty::node_id_to_type(ccx.tcx, method.id);

    check_bare_fn(ccx,
                  &*method.pe_fn_decl(),
                  &*method.pe_body(),
                  method.id,
                  fty,
                  param_env);
}

fn check_impl_items_against_trait(ccx: &CrateCtxt,
                                  impl_span: Span,
                                  ast_trait_ref: &ast::TraitRef,
                                  impl_trait_ref: &ty::TraitRef,
                                  impl_items: &[ast::ImplItem]) {
    // Locate trait methods
    let tcx = ccx.tcx;
    let trait_items = ty::trait_items(tcx, impl_trait_ref.def_id);

    // Check existing impl methods to see if they are both present in trait
    // and compatible with trait signature
    for impl_item in impl_items.iter() {
        match *impl_item {
            ast::MethodImplItem(ref impl_method) => {
                let impl_method_def_id = local_def(impl_method.id);
                let impl_item_ty = ty::impl_or_trait_item(ccx.tcx,
                                                          impl_method_def_id);

                // If this is an impl of a trait method, find the
                // corresponding method definition in the trait.
                let opt_trait_method_ty =
                    trait_items.iter()
                               .find(|ti| {
                                   ti.ident().name == impl_item_ty.ident()
                                                                  .name
                               });
                match opt_trait_method_ty {
                    Some(trait_method_ty) => {
                        match (trait_method_ty, &impl_item_ty) {
                            (&ty::MethodTraitItem(ref trait_method_ty),
                             &ty::MethodTraitItem(ref impl_method_ty)) => {
                                compare_impl_method(ccx.tcx,
                                                    &**impl_method_ty,
                                                    impl_method.span,
                                                    impl_method.pe_body().id,
                                                    &**trait_method_ty,
                                                    &impl_trait_ref.substs);
                            }
                            _ => {
                                // This is span_bug as it should have already been
                                // caught in resolve.
                                tcx.sess
                                   .span_bug(impl_method.span,
                                             format!("item `{}` is of a \
                                                      different kind from \
                                                      its trait `{}`",
                                                     token::get_ident(
                                                        impl_item_ty.ident()),
                                                     pprust::path_to_string(
                                                        &ast_trait_ref.path))
                                             .as_slice());
                            }
                        }
                    }
                    None => {
                        // This is span_bug as it should have already been
                        // caught in resolve.
                        tcx.sess.span_bug(
                            impl_method.span,
                            format!(
                                "method `{}` is not a member of trait `{}`",
                                token::get_ident(impl_item_ty.ident()),
                                pprust::path_to_string(
                                    &ast_trait_ref.path)).as_slice());
                    }
                }
            }
            ast::TypeImplItem(ref typedef) => {
                let typedef_def_id = local_def(typedef.id);
                let typedef_ty = ty::impl_or_trait_item(ccx.tcx,
                                                        typedef_def_id);

                // If this is an impl of an associated type, find the
                // corresponding type definition in the trait.
                let opt_associated_type =
                    trait_items.iter()
                               .find(|ti| {
                                   ti.ident().name == typedef_ty.ident().name
                               });
                match opt_associated_type {
                    Some(associated_type) => {
                        match (associated_type, &typedef_ty) {
                            (&ty::TypeTraitItem(_),
                             &ty::TypeTraitItem(_)) => {}
                            _ => {
                                // This is `span_bug` as it should have
                                // already been caught in resolve.
                                tcx.sess
                                   .span_bug(typedef.span,
                                             format!("item `{}` is of a \
                                                      different kind from \
                                                      its trait `{}`",
                                                     token::get_ident(
                                                        typedef_ty.ident()),
                                                     pprust::path_to_string(
                                                        &ast_trait_ref.path))
                                             .as_slice());
                            }
                        }
                    }
                    None => {
                        // This is `span_bug` as it should have already been
                        // caught in resolve.
                        tcx.sess.span_bug(
                            typedef.span,
                            format!(
                                "associated type `{}` is not a member of \
                                 trait `{}`",
                                token::get_ident(typedef_ty.ident()),
                                pprust::path_to_string(
                                    &ast_trait_ref.path)).as_slice());
                    }
                }
            }
        }
    }

    // Check for missing items from trait
    let provided_methods = ty::provided_trait_methods(tcx,
                                                      impl_trait_ref.def_id);
    let mut missing_methods = Vec::new();
    for trait_item in trait_items.iter() {
        match *trait_item {
            ty::MethodTraitItem(ref trait_method) => {
                let is_implemented =
                    impl_items.iter().any(|ii| {
                        match *ii {
                            ast::MethodImplItem(ref m) => {
                                m.pe_ident().name == trait_method.ident.name
                            }
                            ast::TypeImplItem(_) => false,
                        }
                    });
                let is_provided =
                    provided_methods.iter().any(
                        |m| m.ident.name == trait_method.ident.name);
                if !is_implemented && !is_provided {
                    missing_methods.push(
                        format!("`{}`",
                                token::get_ident(trait_method.ident)));
                }
            }
            ty::TypeTraitItem(ref associated_type) => {
                let is_implemented = impl_items.iter().any(|ii| {
                    match *ii {
                        ast::TypeImplItem(ref typedef) => {
                            typedef.ident.name == associated_type.ident.name
                        }
                        ast::MethodImplItem(_) => false,
                    }
                });
                if !is_implemented {
                    missing_methods.push(
                        format!("`{}`",
                                token::get_ident(associated_type.ident)));
                }
            }
        }
    }

    if !missing_methods.is_empty() {
        span_err!(tcx.sess, impl_span, E0046,
            "not all trait items implemented, missing: {}",
            missing_methods.connect(", "));
    }
}

/**
 * Checks that a method from an impl conforms to the signature of
 * the same method as declared in the trait.
 *
 * # Parameters
 *
 * - impl_generics: the generics declared on the impl itself (not the method!)
 * - impl_m: type of the method we are checking
 * - impl_m_span: span to use for reporting errors
 * - impl_m_body_id: id of the method body
 * - trait_m: the method in the trait
 * - trait_to_impl_substs: the substitutions used on the type of the trait
 */
fn compare_impl_method(tcx: &ty::ctxt,
                       impl_m: &ty::Method,
                       impl_m_span: Span,
                       impl_m_body_id: ast::NodeId,
                       trait_m: &ty::Method,
                       trait_to_impl_substs: &subst::Substs) {
    debug!("compare_impl_method(trait_to_impl_substs={})",
           trait_to_impl_substs.repr(tcx));
    let infcx = infer::new_infer_ctxt(tcx);

    // Try to give more informative error messages about self typing
    // mismatches.  Note that any mismatch will also be detected
    // below, where we construct a canonical function type that
    // includes the self parameter as a normal parameter.  It's just
    // that the error messages you get out of this code are a bit more
    // inscrutable, particularly for cases where one method has no
    // self.
    match (&trait_m.explicit_self, &impl_m.explicit_self) {
        (&ty::StaticExplicitSelfCategory,
         &ty::StaticExplicitSelfCategory) => {}
        (&ty::StaticExplicitSelfCategory, _) => {
            tcx.sess.span_err(
                impl_m_span,
                format!("method `{}` has a `{}` declaration in the impl, \
                        but not in the trait",
                        token::get_ident(trait_m.ident),
                        ppaux::explicit_self_category_to_str(
                            &impl_m.explicit_self)).as_slice());
            return;
        }
        (_, &ty::StaticExplicitSelfCategory) => {
            tcx.sess.span_err(
                impl_m_span,
                format!("method `{}` has a `{}` declaration in the trait, \
                        but not in the impl",
                        token::get_ident(trait_m.ident),
                        ppaux::explicit_self_category_to_str(
                            &trait_m.explicit_self)).as_slice());
            return;
        }
        _ => {
            // Let the type checker catch other errors below
        }
    }

    let num_impl_m_type_params = impl_m.generics.types.len(subst::FnSpace);
    let num_trait_m_type_params = trait_m.generics.types.len(subst::FnSpace);
    if num_impl_m_type_params != num_trait_m_type_params {
        span_err!(tcx.sess, impl_m_span, E0049,
            "method `{}` has {} type parameter{} \
             but its trait declaration has {} type parameter{}",
            token::get_ident(trait_m.ident),
            num_impl_m_type_params,
            if num_impl_m_type_params == 1 {""} else {"s"},
            num_trait_m_type_params,
            if num_trait_m_type_params == 1 {""} else {"s"});
        return;
    }

    if impl_m.fty.sig.inputs.len() != trait_m.fty.sig.inputs.len() {
        span_err!(tcx.sess, impl_m_span, E0050,
            "method `{}` has {} parameter{} \
             but the declaration in trait `{}` has {}",
            token::get_ident(trait_m.ident),
            impl_m.fty.sig.inputs.len(),
            if impl_m.fty.sig.inputs.len() == 1 {""} else {"s"},
            ty::item_path_str(tcx, trait_m.def_id),
            trait_m.fty.sig.inputs.len());
        return;
    }

    // This code is best explained by example. Consider a trait:
    //
    //     trait Trait<T> {
    //          fn method<'a,M>(t: T, m: &'a M) -> Self;
    //     }
    //
    // And an impl:
    //
    //     impl<'i, U> Trait<&'i U> for Foo {
    //          fn method<'b,N>(t: &'i U, m: &'b N) -> Foo;
    //     }
    //
    // We wish to decide if those two method types are compatible.
    //
    // We start out with trait_to_impl_substs, that maps the trait type
    // parameters to impl type parameters:
    //
    //     trait_to_impl_substs = {T => &'i U, Self => Foo}
    //
    // We create a mapping `dummy_substs` that maps from the impl type
    // parameters to fresh types and regions. For type parameters,
    // this is the identity transform, but we could as well use any
    // skolemized types. For regions, we convert from bound to free
    // regions (Note: but only early-bound regions, i.e., those
    // declared on the impl or used in type parameter bounds).
    //
    //     impl_to_skol_substs = {'i => 'i0, U => U0, N => N0 }
    //
    // Now we can apply skol_substs to the type of the impl method
    // to yield a new function type in terms of our fresh, skolemized
    // types:
    //
    //     <'b> fn(t: &'i0 U0, m: &'b) -> Foo
    //
    // We now want to extract and substitute the type of the *trait*
    // method and compare it. To do so, we must create a compound
    // substitution by combining trait_to_impl_substs and
    // impl_to_skol_substs, and also adding a mapping for the method
    // type parameters. We extend the mapping to also include
    // the method parameters.
    //
    //     trait_to_skol_substs = { T => &'i0 U0, Self => Foo, M => N0 }
    //
    // Applying this to the trait method type yields:
    //
    //     <'a> fn(t: &'i0 U0, m: &'a) -> Foo
    //
    // This type is also the same but the name of the bound region ('a
    // vs 'b).  However, the normal subtyping rules on fn types handle
    // this kind of equivalency just fine.

    // Create mapping from impl to skolemized.
    let skol_tps =
        impl_m.generics.types.map(
            |d| ty::mk_param_from_def(tcx, d));
    let skol_regions =
        impl_m.generics.regions.map(
            |l| ty::free_region_from_def(impl_m_body_id, l));
    let impl_to_skol_substs =
        subst::Substs::new(skol_tps.clone(), skol_regions.clone());

    // Create mapping from trait to skolemized.
    let trait_to_skol_substs =
        trait_to_impl_substs
        .subst(tcx, &impl_to_skol_substs)
        .with_method(Vec::from_slice(skol_tps.get_slice(subst::FnSpace)),
                     Vec::from_slice(skol_regions.get_slice(subst::FnSpace)));

    // Check region bounds.
    if !check_region_bounds_on_impl_method(tcx,
                                           impl_m_span,
                                           impl_m,
                                           &trait_m.generics,
                                           &impl_m.generics,
                                           &trait_to_skol_substs,
                                           &impl_to_skol_substs) {
        return;
    }

    // Check bounds.
    let it = trait_m.generics.types.get_slice(subst::FnSpace).iter()
        .zip(impl_m.generics.types.get_slice(subst::FnSpace).iter());
    for (i, (trait_param_def, impl_param_def)) in it.enumerate() {
        // Check that the impl does not require any builtin-bounds
        // that the trait does not guarantee:
        let extra_bounds =
            impl_param_def.bounds.builtin_bounds -
            trait_param_def.bounds.builtin_bounds;
        if !extra_bounds.is_empty() {
            span_err!(tcx.sess, impl_m_span, E0051,
                "in method `{}`, type parameter {} requires `{}`, \
                 which is not required by the corresponding type parameter \
                 in the trait declaration",
                token::get_ident(trait_m.ident),
                i,
                extra_bounds.user_string(tcx));
           return;
        }

        // Check that the trait bounds of the trait imply the bounds of its
        // implementation.
        //
        // FIXME(pcwalton): We could be laxer here regarding sub- and super-
        // traits, but I doubt that'll be wanted often, so meh.
        for impl_trait_bound in impl_param_def.bounds.trait_bounds.iter() {
            debug!("compare_impl_method(): impl-trait-bound subst");
            let impl_trait_bound =
                impl_trait_bound.subst(tcx, &impl_to_skol_substs);

            let mut ok = false;
            for trait_bound in trait_param_def.bounds.trait_bounds.iter() {
                debug!("compare_impl_method(): trait-bound subst");
                let trait_bound =
                    trait_bound.subst(tcx, &trait_to_skol_substs);
                let infcx = infer::new_infer_ctxt(tcx);
                match infer::mk_sub_trait_refs(&infcx,
                                               true,
                                               infer::Misc(impl_m_span),
                                               trait_bound,
                                               impl_trait_bound.clone()) {
                    Ok(_) => {
                        ok = true;
                        break
                    }
                    Err(_) => continue,
                }
            }

            if !ok {
                span_err!(tcx.sess, impl_m_span, E0052,
                    "in method `{}`, type parameter {} requires bound `{}`, which is not \
                     required by the corresponding type parameter in the trait declaration",
                    token::get_ident(trait_m.ident),
                    i,
                    ppaux::trait_ref_to_string(tcx, &*impl_trait_bound));
            }
        }
    }

    // Compute skolemized form of impl and trait method tys.
    let impl_fty = ty::mk_bare_fn(tcx, impl_m.fty.clone());
    let impl_fty = impl_fty.subst(tcx, &impl_to_skol_substs);
    let trait_fty = ty::mk_bare_fn(tcx, trait_m.fty.clone());
    let trait_fty = trait_fty.subst(tcx, &trait_to_skol_substs);

    // Check the impl method type IM is a subtype of the trait method
    // type TM. To see why this makes sense, think of a vtable. The
    // expected type of the function pointers in the vtable is the
    // type TM of the trait method.  The actual type will be the type
    // IM of the impl method. Because we know that IM <: TM, that
    // means that anywhere a TM is expected, a IM will do instead. In
    // other words, anyone expecting to call a method with the type
    // from the trait, can safely call a method with the type from the
    // impl instead.
    debug!("checking trait method for compatibility: impl ty {}, trait ty {}",
           impl_fty.repr(tcx),
           trait_fty.repr(tcx));
    match infer::mk_subty(&infcx, false, infer::MethodCompatCheck(impl_m_span),
                          impl_fty, trait_fty) {
        Ok(()) => {}
        Err(ref terr) => {
            span_err!(tcx.sess, impl_m_span, E0053,
                "method `{}` has an incompatible type for trait: {}",
                token::get_ident(trait_m.ident),
                ty::type_err_to_str(tcx, terr));
            ty::note_and_explain_type_err(tcx, terr);
        }
    }

    // Finally, resolve all regions. This catches wily misuses of lifetime
    // parameters.
    infcx.resolve_regions_and_report_errors();

    fn check_region_bounds_on_impl_method(tcx: &ty::ctxt,
                                          span: Span,
                                          impl_m: &ty::Method,
                                          trait_generics: &ty::Generics,
                                          impl_generics: &ty::Generics,
                                          trait_to_skol_substs: &Substs,
                                          impl_to_skol_substs: &Substs)
                                          -> bool
    {
        /*!

        Check that region bounds on impl method are the same as those
        on the trait. In principle, it could be ok for there to be
        fewer region bounds on the impl method, but this leads to an
        annoying corner case that is painful to handle (described
        below), so for now we can just forbid it.

        Example (see
        `src/test/compile-fail/regions-bound-missing-bound-in-impl.rs`):

            trait Foo<'a> {
                fn method1<'b>();
                fn method2<'b:'a>();
            }

            impl<'a> Foo<'a> for ... {
                fn method1<'b:'a>() { .. case 1, definitely bad .. }
                fn method2<'b>() { .. case 2, could be ok .. }
            }

        The "definitely bad" case is case #1. Here, the impl adds an
        extra constraint not present in the trait.

        The "maybe bad" case is case #2. Here, the impl adds an extra
        constraint not present in the trait. We could in principle
        allow this, but it interacts in a complex way with early/late
        bound resolution of lifetimes. Basically the presence or
        absence of a lifetime bound affects whether the lifetime is
        early/late bound, and right now the code breaks if the trait
        has an early bound lifetime parameter and the method does not.

        */

        let trait_params = trait_generics.regions.get_slice(subst::FnSpace);
        let impl_params = impl_generics.regions.get_slice(subst::FnSpace);

        debug!("check_region_bounds_on_impl_method: \
               trait_generics={} \
               impl_generics={}",
               trait_generics.repr(tcx),
               impl_generics.repr(tcx));

        // Must have same number of early-bound lifetime parameters.
        // Unfortunately, if the user screws up the bounds, then this
        // will change classification between early and late.  E.g.,
        // if in trait we have `<'a,'b:'a>`, and in impl we just have
        // `<'a,'b>`, then we have 2 early-bound lifetime parameters
        // in trait but 0 in the impl. But if we report "expected 2
        // but found 0" it's confusing, because it looks like there
        // are zero. Since I don't quite know how to phrase things at
        // the moment, give a kind of vague error message.
        if trait_params.len() != impl_params.len() {
            tcx.sess.span_err(
                span,
                format!("lifetime parameters or bounds on method `{}` do \
                         not match the trait declaration",
                        token::get_ident(impl_m.ident)).as_slice());
            return false;
        }

        // Each parameter `'a:'b+'c+'d` in trait should have the same
        // set of bounds in the impl, after subst.
        for (trait_param, impl_param) in
            trait_params.iter().zip(
                impl_params.iter())
        {
            let trait_bounds =
                trait_param.bounds.subst(tcx, trait_to_skol_substs);
            let impl_bounds =
                impl_param.bounds.subst(tcx, impl_to_skol_substs);

            debug!("check_region_bounds_on_impl_method: \
                   trait_param={} \
                   impl_param={} \
                   trait_bounds={} \
                   impl_bounds={}",
                   trait_param.repr(tcx),
                   impl_param.repr(tcx),
                   trait_bounds.repr(tcx),
                   impl_bounds.repr(tcx));

            // Collect the set of bounds present in trait but not in
            // impl.
            let missing: Vec<ty::Region> =
                trait_bounds.iter()
                .filter(|&b| !impl_bounds.contains(b))
                .map(|&b| b)
                .collect();

            // Collect set present in impl but not in trait.
            let extra: Vec<ty::Region> =
                impl_bounds.iter()
                .filter(|&b| !trait_bounds.contains(b))
                .map(|&b| b)
                .collect();

            debug!("missing={} extra={}",
                   missing.repr(tcx), extra.repr(tcx));

            let err = if missing.len() != 0 || extra.len() != 0 {
                tcx.sess.span_err(
                    span,
                    format!(
                        "the lifetime parameter `{}` declared in the impl \
                         has a distinct set of bounds \
                         from its counterpart `{}` \
                         declared in the trait",
                        impl_param.name.user_string(tcx),
                        trait_param.name.user_string(tcx)).as_slice());
                true
            } else {
                false
            };

            if missing.len() != 0 {
                tcx.sess.span_note(
                    span,
                    format!("the impl is missing the following bounds: `{}`",
                            missing.user_string(tcx)).as_slice());
            }

            if extra.len() != 0 {
                tcx.sess.span_note(
                    span,
                    format!("the impl has the following extra bounds: `{}`",
                            extra.user_string(tcx)).as_slice());
            }

            if err {
                return false;
            }
        }

        return true;
    }
}

fn check_cast(fcx: &FnCtxt,
              cast_expr: &ast::Expr,
              e: &ast::Expr,
              t: &ast::Ty) {
    let id = cast_expr.id;
    let span = cast_expr.span;

    // Find the type of `e`. Supply hints based on the type we are casting to,
    // if appropriate.
    let t_1 = fcx.to_ty(t);
    let t_1 = structurally_resolved_type(fcx, span, t_1);

    if ty::type_is_scalar(t_1) {
        // Supply the type as a hint so as to influence integer
        // literals and other things that might care.
        check_expr_with_expectation(fcx, e, ExpectCastableToType(t_1))
    } else {
        check_expr(fcx, e)
    }

    let t_e = fcx.expr_ty(e);

    debug!("t_1={}", fcx.infcx().ty_to_string(t_1));
    debug!("t_e={}", fcx.infcx().ty_to_string(t_e));

    if ty::type_is_error(t_e) {
        fcx.write_error(id);
        return
    }

    if ty::type_is_bot(t_e) {
        fcx.write_bot(id);
        return
    }

    if !ty::type_is_sized(fcx.tcx(), t_1) {
        let tstr = fcx.infcx().ty_to_string(t_1);
        fcx.type_error_message(span, |actual| {
            format!("cast to unsized type: `{}` as `{}`", actual, tstr)
        }, t_e, None);
        match ty::get(t_e).sty {
            ty::ty_rptr(_, ty::mt { mutbl: mt, .. }) => {
                let mtstr = match mt {
                    ast::MutMutable => "mut ",
                    ast::MutImmutable => ""
                };
                if ty::type_is_trait(t_1) {
                    span_note!(fcx.tcx().sess, t.span, "did you mean `&{}{}`?", mtstr, tstr);
                } else {
                    span_note!(fcx.tcx().sess, span,
                               "consider using an implicit coercion to `&{}{}` instead",
                               mtstr, tstr);
                }
            }
            ty::ty_uniq(..) => {
                span_note!(fcx.tcx().sess, t.span, "did you mean `Box<{}>`?", tstr);
            }
            _ => {
                span_note!(fcx.tcx().sess, e.span,
                           "consider using a box or reference as appropriate");
            }
        }
        fcx.write_error(id);
        return
    }

    if ty::type_is_trait(t_1) {
        // This will be looked up later on.
        vtable2::check_object_cast(fcx, cast_expr, e, t_1);
        fcx.write_ty(id, t_1);
        return
    }

    let t_1 = structurally_resolved_type(fcx, span, t_1);
    let t_e = structurally_resolved_type(fcx, span, t_e);

    if ty::type_is_nil(t_e) {
        fcx.type_error_message(span, |actual| {
            format!("cast from nil: `{}` as `{}`",
                    actual,
                    fcx.infcx().ty_to_string(t_1))
        }, t_e, None);
    } else if ty::type_is_nil(t_1) {
        fcx.type_error_message(span, |actual| {
            format!("cast to nil: `{}` as `{}`",
                    actual,
                    fcx.infcx().ty_to_string(t_1))
        }, t_e, None);
    }

    let t_1_is_scalar = ty::type_is_scalar(t_1);
    let t_1_is_char = ty::type_is_char(t_1);
    let t_1_is_bare_fn = ty::type_is_bare_fn(t_1);
    let t_1_is_float = ty::type_is_floating_point(t_1);

    // casts to scalars other than `char` and `bare fn` are trivial
    let t_1_is_trivial = t_1_is_scalar && !t_1_is_char && !t_1_is_bare_fn;
    if ty::type_is_c_like_enum(fcx.tcx(), t_e) && t_1_is_trivial {
        if t_1_is_float || ty::type_is_unsafe_ptr(t_1) {
            fcx.type_error_message(span, |actual| {
                format!("illegal cast; cast through an \
                         integer first: `{}` as `{}`",
                        actual,
                        fcx.infcx().ty_to_string(t_1))
            }, t_e, None);
        }
        // casts from C-like enums are allowed
    } else if t_1_is_char {
        let t_e = fcx.infcx().resolve_type_vars_if_possible(t_e);
        if ty::get(t_e).sty != ty::ty_uint(ast::TyU8) {
            fcx.type_error_message(span, |actual| {
                format!("only `u8` can be cast as \
                         `char`, not `{}`", actual)
            }, t_e, None);
        }
    } else if ty::get(t_1).sty == ty::ty_bool {
        span_err!(fcx.tcx().sess, span, E0054,
            "cannot cast as `bool`, compare with zero instead");
    } else if ty::type_is_region_ptr(t_e) && ty::type_is_unsafe_ptr(t_1) {
        fn types_compatible(fcx: &FnCtxt, sp: Span,
                            t1: ty::t, t2: ty::t) -> bool {
            match ty::get(t1).sty {
                ty::ty_vec(_, Some(_)) => {}
                _ => return false
            }
            if ty::type_needs_infer(t2) {
                // This prevents this special case from going off when casting
                // to a type that isn't fully specified; e.g. `as *_`. (Issue
                // #14893.)
                return false
            }

            let el = ty::sequence_element_type(fcx.tcx(), t1);
            infer::mk_eqty(fcx.infcx(),
                           false,
                           infer::Misc(sp),
                           el,
                           t2).is_ok()
        }

        // Due to the limitations of LLVM global constants,
        // region pointers end up pointing at copies of
        // vector elements instead of the original values.
        // To allow unsafe pointers to work correctly, we
        // need to special-case obtaining an unsafe pointer
        // from a region pointer to a vector.

        /* this cast is only allowed from &[T, ..n] to *T or
        &T to *T. */
        match (&ty::get(t_e).sty, &ty::get(t_1).sty) {
            (&ty::ty_rptr(_, ty::mt { ty: mt1, mutbl: ast::MutImmutable }),
             &ty::ty_ptr(ty::mt { ty: mt2, mutbl: ast::MutImmutable }))
            if types_compatible(fcx, e.span, mt1, mt2) => {
                /* this case is allowed */
            }
            _ => {
                demand::coerce(fcx, e.span, t_1, &*e);
            }
        }
    } else if !(ty::type_is_scalar(t_e) && t_1_is_trivial) {
        /*
        If more type combinations should be supported than are
        supported here, then file an enhancement issue and
        record the issue number in this comment.
        */
        fcx.type_error_message(span, |actual| {
            format!("non-scalar cast: `{}` as `{}`",
                    actual,
                    fcx.infcx().ty_to_string(t_1))
        }, t_e, None);
    } else if ty::type_is_unsafe_ptr(t_e) && t_1_is_float {
        fcx.type_error_message(span, |actual| {
            format!("cannot cast from pointer to float directly: `{}` as `{}`; cast through an \
                     integer first",
                    actual,
                    fcx.infcx().ty_to_string(t_1))
        }, t_e, None);
    }

    fcx.write_ty(id, t_1);
}

impl<'a, 'tcx> AstConv<'tcx> for FnCtxt<'a, 'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> { self.ccx.tcx }

    fn get_item_ty(&self, id: ast::DefId) -> ty::Polytype {
        ty::lookup_item_type(self.tcx(), id)
    }

    fn get_trait_def(&self, id: ast::DefId) -> Rc<ty::TraitDef> {
        ty::lookup_trait_def(self.tcx(), id)
    }

    fn ty_infer(&self, _span: Span) -> ty::t {
        self.infcx().next_ty_var()
    }

    fn associated_types_of_trait_are_valid(&self, _: ty::t, _: ast::DefId)
                                           -> bool {
        false
    }

    fn associated_type_binding(&self,
                               span: Span,
                               _: Option<ty::t>,
                               _: ast::DefId,
                               _: ast::DefId)
                               -> ty::t {
        self.tcx().sess.span_err(span, "unsupported associated type binding");
        ty::mk_err()
    }
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> { self.ccx.tcx }

    pub fn infcx<'b>(&'b self) -> &'b infer::InferCtxt<'a, 'tcx> {
        &self.inh.infcx
    }

    pub fn err_count_since_creation(&self) -> uint {
        self.ccx.tcx.sess.err_count() - self.err_count_on_creation
    }
}

impl<'a, 'tcx> RegionScope for infer::InferCtxt<'a, 'tcx> {
    fn default_region_bound(&self, span: Span) -> Option<ty::Region> {
        Some(self.next_region_var(infer::MiscVariable(span)))
    }

    fn anon_regions(&self, span: Span, count: uint)
                    -> Result<Vec<ty::Region>, Option<Vec<(String, uint)>>> {
        Ok(Vec::from_fn(count, |_| {
            self.next_region_var(infer::MiscVariable(span))
        }))
    }
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn tag(&self) -> String {
        format!("{}", self as *const FnCtxt)
    }

    pub fn local_ty(&self, span: Span, nid: ast::NodeId) -> ty::t {
        match self.inh.locals.borrow().find(&nid) {
            Some(&t) => t,
            None => {
                self.tcx().sess.span_bug(
                    span,
                    format!("no type for local variable {:?}",
                            nid).as_slice());
            }
        }
    }

    #[inline]
    pub fn write_ty(&self, node_id: ast::NodeId, ty: ty::t) {
        debug!("write_ty({}, {}) in fcx {}",
               node_id, ppaux::ty_to_string(self.tcx(), ty), self.tag());
        self.inh.node_types.borrow_mut().insert(node_id, ty);
    }

    pub fn write_object_cast(&self,
                             key: ast::NodeId,
                             trait_ref: Rc<ty::TraitRef>) {
        debug!("write_object_cast key={} trait_ref={}",
               key, trait_ref.repr(self.tcx()));
        self.inh.object_cast_map.borrow_mut().insert(key, trait_ref);
    }

    pub fn write_substs(&self, node_id: ast::NodeId, substs: ty::ItemSubsts) {
        if !substs.substs.is_noop() {
            debug!("write_substs({}, {}) in fcx {}",
                   node_id,
                   substs.repr(self.tcx()),
                   self.tag());

            self.inh.item_substs.borrow_mut().insert(node_id, substs);
        }
    }

    pub fn write_ty_substs(&self,
                           node_id: ast::NodeId,
                           ty: ty::t,
                           substs: ty::ItemSubsts) {
        let ty = ty.subst(self.tcx(), &substs.substs);
        self.write_ty(node_id, ty);
        self.write_substs(node_id, substs);
    }

    pub fn write_autoderef_adjustment(&self,
                                      node_id: ast::NodeId,
                                      span: Span,
                                      derefs: uint) {
        if derefs == 0 { return; }
        self.write_adjustment(
            node_id,
            span,
            ty::AdjustDerefRef(ty::AutoDerefRef {
                autoderefs: derefs,
                autoref: None })
        );
    }

    pub fn write_adjustment(&self,
                            node_id: ast::NodeId,
                            span: Span,
                            adj: ty::AutoAdjustment) {
        debug!("write_adjustment(node_id={:?}, adj={:?})", node_id, adj);

        // Careful: adjustments can imply trait obligations if we are
        // casting from a concrete type to an object type. I think
        // it'd probably be nicer to move the logic that creates the
        // obligation into the code that creates the adjustment, but
        // that's a bit awkward, so instead we go digging and pull the
        // obligation out here.
        self.register_adjustment_obligations(span, &adj);
        self.inh.adjustments.borrow_mut().insert(node_id, adj);
    }

    fn register_adjustment_obligations(&self,
                                       span: Span,
                                       adj: &ty::AutoAdjustment) {
        match *adj {
            ty::AdjustAddEnv(..) => { }
            ty::AdjustDerefRef(ref d_r) => {
                match d_r.autoref {
                    Some(ref a_r) => {
                        self.register_autoref_obligations(span, a_r);
                    }
                    None => {}
                }
            }
        }
    }

    fn register_autoref_obligations(&self,
                                    span: Span,
                                    autoref: &ty::AutoRef) {
        match *autoref {
            ty::AutoUnsize(ref unsize) => {
                self.register_unsize_obligations(span, unsize);
            }
            ty::AutoPtr(_, _, None) |
            ty::AutoUnsafe(_, None) => {
            }
            ty::AutoPtr(_, _, Some(ref a_r)) |
            ty::AutoUnsafe(_, Some(ref a_r)) => {
                self.register_autoref_obligations(span, &**a_r)
            }
            ty::AutoUnsizeUniq(ref unsize) => {
                self.register_unsize_obligations(span, unsize);
            }
        }
    }

    fn register_unsize_obligations(&self,
                                   span: Span,
                                   unsize: &ty::UnsizeKind) {
        debug!("register_unsize_obligations: unsize={:?}", unsize);

        match *unsize {
            ty::UnsizeLength(..) => {}
            ty::UnsizeStruct(ref u, _) => {
                self.register_unsize_obligations(span, &**u)
            }
            ty::UnsizeVtable(ref ty_trait, self_ty) => {
                vtable2::register_object_cast_obligations(self,
                                                          span,
                                                          ty_trait,
                                                          self_ty);
            }
        }
    }

    pub fn instantiate_item_type(&self,
                                 span: Span,
                                 def_id: ast::DefId)
                                 -> TypeAndSubsts
    {
        /*!
         * Returns the type of `def_id` with all generics replaced by
         * by fresh type/region variables. Also returns the
         * substitution from the type parameters on `def_id` to the
         * fresh variables.  Registers any trait obligations specified
         * on `def_id` at the same time.
         */

        let polytype =
            ty::lookup_item_type(self.tcx(), def_id);
        let substs =
            self.infcx().fresh_substs_for_generics(
                span,
                &polytype.generics);
        self.add_obligations_for_parameters(
            traits::ObligationCause::new(
                span,
                traits::ItemObligation(def_id)),
            &substs,
            &polytype.generics);
        let monotype =
            polytype.ty.subst(self.tcx(), &substs);

        TypeAndSubsts {
            ty: monotype,
            substs: substs
        }
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

    pub fn require_type_meets(&self,
                              ty: ty::t,
                              span: Span,
                              code: traits::ObligationCauseCode,
                              bound: ty::BuiltinBound)
    {
        let obligation = traits::obligation_for_builtin_bound(
            self.tcx(),
            traits::ObligationCause::new(span, code),
            ty,
            bound);
        match obligation {
            Ok(ob) => self.register_obligation(ob),
            _ => {}
        }
    }

    pub fn require_type_is_sized(&self,
                                 ty: ty::t,
                                 span: Span,
                                 code: traits::ObligationCauseCode)
    {
        self.require_type_meets(ty, span, code, ty::BoundSized);
    }

    pub fn require_expr_have_sized_type(&self,
                                        expr: &ast::Expr,
                                        code: traits::ObligationCauseCode)
    {
        self.require_type_is_sized(self.expr_ty(expr), expr.span, code);
    }

    pub fn register_obligation(&self,
                               obligation: traits::Obligation)
    {
        debug!("register_obligation({})",
               obligation.repr(self.tcx()));

        self.inh.fulfillment_cx
            .borrow_mut()
            .register_obligation(self.tcx(), obligation);
    }

    pub fn to_ty(&self, ast_t: &ast::Ty) -> ty::t {
        let t = ast_ty_to_ty(self, self.infcx(), ast_t);

        let mut bounds_checker = wf::BoundsChecker::new(self,
                                                        ast_t.span,
                                                        self.body_id,
                                                        None);
        bounds_checker.check_ty(t);

        t
    }

    pub fn pat_to_string(&self, pat: &ast::Pat) -> String {
        pat.repr(self.tcx())
    }

    pub fn expr_ty(&self, ex: &ast::Expr) -> ty::t {
        match self.inh.node_types.borrow().find(&ex.id) {
            Some(&t) => t,
            None => {
                self.tcx().sess.bug(format!("no type for expr in fcx {}",
                                            self.tag()).as_slice());
            }
        }
    }

    pub fn node_ty(&self, id: ast::NodeId) -> ty::t {
        match self.inh.node_types.borrow().find(&id) {
            Some(&t) => t,
            None => {
                self.tcx().sess.bug(
                    format!("no type for node {}: {} in fcx {}",
                            id, self.tcx().map.node_to_string(id),
                            self.tag()).as_slice());
            }
        }
    }

    pub fn opt_node_ty_substs(&self,
                              id: ast::NodeId,
                              f: |&ty::ItemSubsts|) {
        match self.inh.item_substs.borrow().find(&id) {
            Some(s) => { f(s) }
            None => { }
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

    pub fn can_mk_eqty(&self, sub: ty::t, sup: ty::t)
                       -> Result<(), ty::type_err> {
        infer::can_mk_eqty(self.infcx(), sub, sup)
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
            Ok(None) => Ok(()),
            Err(ref e) => Err((*e)),
            Ok(Some(adjustment)) => {
                self.write_adjustment(expr.id, expr.span, adjustment);
                Ok(())
            }
        }
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
                   origin: infer::SubregionOrigin,
                   sub: ty::Region,
                   sup: ty::Region) {
        infer::mk_subr(self.infcx(), origin, sub, sup)
    }

    pub fn type_error_message(&self,
                              sp: Span,
                              mk_msg: |String| -> String,
                              actual_ty: ty::t,
                              err: Option<&ty::type_err>) {
        self.infcx().type_error_message(sp, mk_msg, actual_ty, err);
    }

    pub fn report_mismatched_types(&self,
                                   sp: Span,
                                   e: ty::t,
                                   a: ty::t,
                                   err: &ty::type_err) {
        self.infcx().report_mismatched_types(sp, e, a, err)
    }

    pub fn register_region_obligation(&self,
                                      origin: infer::SubregionOrigin,
                                      ty: ty::t,
                                      r: ty::Region)
    {
        /*!
         * Registers an obligation for checking later, during
         * regionck, that the type `ty` must outlive the region `r`.
         */

        let mut region_obligations = self.inh.region_obligations.borrow_mut();
        let region_obligation = RegionObligation { sub_region: r,
                                  sup_type: ty,
                                  origin: origin };

        match region_obligations.entry(self.body_id) {
            Vacant(entry) => { entry.set(vec![region_obligation]); },
            Occupied(mut entry) => { entry.get_mut().push(region_obligation); },
        }
    }

    pub fn add_obligations_for_parameters(&self,
                                          cause: traits::ObligationCause,
                                          substs: &Substs,
                                          generics: &ty::Generics)
    {
        /*!
         * Given a set of generic parameter definitions (`generics`)
         * and the values provided for each of them (`substs`),
         * creates and registers suitable region obligations.
         *
         * For example, if there is a function:
         *
         *    fn foo<'a,T:'a>(...)
         *
         * and a reference:
         *
         *    let f = foo;
         *
         * Then we will create a fresh region variable `'$0` and a
         * fresh type variable `$1` for `'a` and `T`. This routine
         * will add a region obligation `$1:'$0` and register it
         * locally.
         */

        debug!("add_obligations_for_parameters(substs={}, generics={})",
               substs.repr(self.tcx()),
               generics.repr(self.tcx()));

        self.add_trait_obligations_for_generics(cause, substs, generics);
        self.add_region_obligations_for_generics(cause, substs, generics);
    }

    fn add_trait_obligations_for_generics(&self,
                                          cause: traits::ObligationCause,
                                          substs: &Substs,
                                          generics: &ty::Generics) {
        let obligations =
            traits::obligations_for_generics(self.tcx(),
                                             cause,
                                             generics,
                                             substs);
        obligations.map_move(|o| self.register_obligation(o));
    }

    fn add_region_obligations_for_generics(&self,
                                           cause: traits::ObligationCause,
                                           substs: &Substs,
                                           generics: &ty::Generics)
    {
        assert_eq!(generics.types.iter().len(),
                   substs.types.iter().len());
        for (type_def, &type_param) in
            generics.types.iter().zip(
                substs.types.iter())
        {
            let param_ty = ty::ParamTy { space: type_def.space,
                                         idx: type_def.index,
                                         def_id: type_def.def_id };
            let bounds = type_def.bounds.subst(self.tcx(), substs);
            self.add_region_obligations_for_type_parameter(
                cause.span, param_ty, &bounds, type_param);
        }

        assert_eq!(generics.regions.iter().len(),
                   substs.regions().iter().len());
        for (region_def, &region_param) in
            generics.regions.iter().zip(
                substs.regions().iter())
        {
            let bounds = region_def.bounds.subst(self.tcx(), substs);
            self.add_region_obligations_for_region_parameter(
                cause.span, bounds.as_slice(), region_param);
        }
    }

    fn add_region_obligations_for_type_parameter(&self,
                                                 span: Span,
                                                 param_ty: ty::ParamTy,
                                                 param_bound: &ty::ParamBounds,
                                                 ty: ty::t)
    {
        // For each declared region bound `T:r`, `T` must outlive `r`.
        let region_bounds =
            ty::required_region_bounds(
                self.tcx(),
                param_bound.region_bounds.as_slice(),
                param_bound.builtin_bounds,
                param_bound.trait_bounds.as_slice());
        for &r in region_bounds.iter() {
            let origin = infer::RelateParamBound(span, param_ty, ty);
            self.register_region_obligation(origin, ty, r);
        }
    }

    fn add_region_obligations_for_region_parameter(&self,
                                                   span: Span,
                                                   region_bounds: &[ty::Region],
                                                   region_param: ty::Region)
    {
        for &b in region_bounds.iter() {
            // For each bound `region:b`, `b <= region` must hold
            // (i.e., `region` must outlive `b`).
            let origin = infer::RelateRegionParamBound(span);
            self.mk_subr(origin, b, region_param);
        }
    }
}

#[deriving(Show)]
pub enum LvaluePreference {
    PreferMutLvalue,
    NoPreference
}

pub fn autoderef<T>(fcx: &FnCtxt, sp: Span, base_ty: ty::t,
                    expr_id: Option<ast::NodeId>,
                    mut lvalue_pref: LvaluePreference,
                    should_stop: |ty::t, uint| -> Option<T>)
                    -> (ty::t, uint, Option<T>) {
    /*!
     * Executes an autoderef loop for the type `t`. At each step, invokes
     * `should_stop` to decide whether to terminate the loop. Returns
     * the final type and number of derefs that it performed.
     *
     * Note: this method does not modify the adjustments table. The caller is
     * responsible for inserting an AutoAdjustment record into the `fcx`
     * using one of the suitable methods.
     */

    let mut t = base_ty;
    for autoderefs in range(0, fcx.tcx().sess.recursion_limit.get()) {
        let resolved_t = structurally_resolved_type(fcx, sp, t);

        if ty::type_is_bot(resolved_t) {
            return (resolved_t, autoderefs, None);
        }

        match should_stop(resolved_t, autoderefs) {
            Some(x) => return (resolved_t, autoderefs, Some(x)),
            None => {}
        }

        // Otherwise, deref if type is derefable:
        let mt = match ty::deref(resolved_t, false) {
            Some(mt) => Some(mt),
            None => {
                let method_call = expr_id.map(|id| MethodCall::autoderef(id, autoderefs));
                try_overloaded_deref(fcx, sp, method_call, None, resolved_t, lvalue_pref)
            }
        };
        match mt {
            Some(mt) => {
                t = mt.ty;
                if mt.mutbl == ast::MutImmutable {
                    lvalue_pref = NoPreference;
                }
            }
            None => return (resolved_t, autoderefs, None)
        }
    }

    // We've reached the recursion limit, error gracefully.
    span_err!(fcx.tcx().sess, sp, E0055,
        "reached the recursion limit while auto-dereferencing {}",
        base_ty.repr(fcx.tcx()));
    (ty::mk_err(), 0, None)
}

/// Attempts to resolve a call expression as an overloaded call.
fn try_overloaded_call<'a>(fcx: &FnCtxt,
                           call_expression: &ast::Expr,
                           callee: &ast::Expr,
                           callee_type: ty::t,
                           args: &[&'a P<ast::Expr>])
                           -> bool {
    // Bail out if the callee is a bare function or a closure. We check those
    // manually.
    match *structure_of(fcx, callee.span, callee_type) {
        ty::ty_bare_fn(_) | ty::ty_closure(_) => return false,
        _ => {}
    }

    // Try `FnOnce`, then `FnMut`, then `Fn`.
    for &(maybe_function_trait, method_name) in [
        (fcx.tcx().lang_items.fn_once_trait(), token::intern("call_once")),
        (fcx.tcx().lang_items.fn_mut_trait(), token::intern("call_mut")),
        (fcx.tcx().lang_items.fn_trait(), token::intern("call"))
    ].iter() {
        let function_trait = match maybe_function_trait {
            None => continue,
            Some(function_trait) => function_trait,
        };
        let method_callee = match method::lookup_in_trait(
                fcx,
                call_expression.span,
                Some(&*callee),
                method_name,
                function_trait,
                callee_type,
                [],
                DontAutoderefReceiver,
                IgnoreStaticMethods) {
            None => continue,
            Some(method_callee) => method_callee,
        };
        let method_call = MethodCall::expr(call_expression.id);
        let output_type = check_method_argument_types(fcx,
                                                      call_expression.span,
                                                      method_callee.ty,
                                                      call_expression,
                                                      args,
                                                      DontDerefArgs,
                                                      TupleArguments);
        fcx.inh.method_map.borrow_mut().insert(method_call, method_callee);
        write_call(fcx, call_expression, output_type);

        if !fcx.tcx().sess.features.borrow().overloaded_calls {
            span_err!(fcx.tcx().sess, call_expression.span, E0056,
                "overloaded calls are experimental");
            span_note!(fcx.tcx().sess, call_expression.span,
                "add `#![feature(overloaded_calls)]` to \
                the crate attributes to enable");
        }

        return true
    }

    false
}

fn try_overloaded_deref(fcx: &FnCtxt,
                        span: Span,
                        method_call: Option<MethodCall>,
                        base_expr: Option<&ast::Expr>,
                        base_ty: ty::t,
                        lvalue_pref: LvaluePreference)
                        -> Option<ty::mt> {
    // Try DerefMut first, if preferred.
    let method = match (lvalue_pref, fcx.tcx().lang_items.deref_mut_trait()) {
        (PreferMutLvalue, Some(trait_did)) => {
            method::lookup_in_trait(fcx, span, base_expr.map(|x| &*x),
                                    token::intern("deref_mut"), trait_did,
                                    base_ty, [], DontAutoderefReceiver, IgnoreStaticMethods)
        }
        _ => None
    };

    // Otherwise, fall back to Deref.
    let method = match (method, fcx.tcx().lang_items.deref_trait()) {
        (None, Some(trait_did)) => {
            method::lookup_in_trait(fcx, span, base_expr.map(|x| &*x),
                                    token::intern("deref"), trait_did,
                                    base_ty, [], DontAutoderefReceiver, IgnoreStaticMethods)
        }
        (method, _) => method
    };

    make_return_type(fcx, method_call, method)
}

fn get_method_ty(method: &Option<MethodCallee>) -> ty::t {
    match method {
        &Some(ref method) => method.ty,
        &None => ty::mk_err()
    }
}

fn make_return_type(fcx: &FnCtxt,
                    method_call: Option<MethodCall>,
                    method: Option<MethodCallee>)
                    -> Option<ty::mt> {
    match method {
        Some(method) => {
            let ref_ty = ty::ty_fn_ret(method.ty);
            match method_call {
                Some(method_call) => {
                    fcx.inh.method_map.borrow_mut().insert(method_call,
                                                           method);
                }
                None => {}
            }
            ty::deref(ref_ty, true)
        }
        None => None,
    }
}

fn try_overloaded_slice(fcx: &FnCtxt,
                        method_call: Option<MethodCall>,
                        expr: &ast::Expr,
                        base_expr: &ast::Expr,
                        base_ty: ty::t,
                        start_expr: &Option<P<ast::Expr>>,
                        end_expr: &Option<P<ast::Expr>>,
                        mutbl: &ast::Mutability)
                        -> Option<ty::mt> {
    let method = if mutbl == &ast::MutMutable {
        // Try `SliceMut` first, if preferred.
        match fcx.tcx().lang_items.slice_mut_trait() {
            Some(trait_did) => {
                let method_name = match (start_expr, end_expr) {
                    (&Some(_), &Some(_)) => "slice_or_fail_mut",
                    (&Some(_), &None) => "slice_from_or_fail_mut",
                    (&None, &Some(_)) => "slice_to_or_fail_mut",
                    (&None, &None) => "as_mut_slice_",
                };

                method::lookup_in_trait(fcx,
                                        expr.span,
                                        Some(&*base_expr),
                                        token::intern(method_name),
                                        trait_did,
                                        base_ty,
                                        [],
                                        DontAutoderefReceiver,
                                        IgnoreStaticMethods)
            }
            _ => None,
        }
    } else {
        // Otherwise, fall back to `Slice`.
        // FIXME(#17293) this will not coerce base_expr, so we miss the Slice
        // trait for `&mut [T]`.
        match fcx.tcx().lang_items.slice_trait() {
            Some(trait_did) => {
                let method_name = match (start_expr, end_expr) {
                    (&Some(_), &Some(_)) => "slice_or_fail",
                    (&Some(_), &None) => "slice_from_or_fail",
                    (&None, &Some(_)) => "slice_to_or_fail",
                    (&None, &None) => "as_slice_",
                };

                method::lookup_in_trait(fcx,
                                        expr.span,
                                        Some(&*base_expr),
                                        token::intern(method_name),
                                        trait_did,
                                        base_ty,
                                        [],
                                        DontAutoderefReceiver,
                                        IgnoreStaticMethods)
            }
            _ => None,
        }
    };


    // Regardless of whether the lookup succeeds, check the method arguments
    // so that we have *some* type for each argument.
    let method_type = get_method_ty(&method);

    let mut args = vec![];
    start_expr.as_ref().map(|x| args.push(x));
    end_expr.as_ref().map(|x| args.push(x));

    check_method_argument_types(fcx,
                                expr.span,
                                method_type,
                                expr,
                                args.as_slice(),
                                DoDerefArgs,
                                DontTupleArguments);

    match method {
        Some(method) => {
            let result_ty = ty::ty_fn_ret(method.ty);
            match method_call {
                Some(method_call) => {
                    fcx.inh.method_map.borrow_mut().insert(method_call,
                                                           method);
                }
                None => {}
            }
            Some(ty::mt { ty: result_ty, mutbl: ast::MutImmutable })
        }
        None => None,
    }
}

fn try_overloaded_index(fcx: &FnCtxt,
                        method_call: Option<MethodCall>,
                        expr: &ast::Expr,
                        base_expr: &ast::Expr,
                        base_ty: ty::t,
                        index_expr: &P<ast::Expr>,
                        lvalue_pref: LvaluePreference)
                        -> Option<ty::mt> {
    // Try `IndexMut` first, if preferred.
    let method = match (lvalue_pref, fcx.tcx().lang_items.index_mut_trait()) {
        (PreferMutLvalue, Some(trait_did)) => {
            method::lookup_in_trait(fcx,
                                    expr.span,
                                    Some(&*base_expr),
                                    token::intern("index_mut"),
                                    trait_did,
                                    base_ty,
                                    [],
                                    DontAutoderefReceiver,
                                    IgnoreStaticMethods)
        }
        _ => None,
    };

    // Otherwise, fall back to `Index`.
    let method = match (method, fcx.tcx().lang_items.index_trait()) {
        (None, Some(trait_did)) => {
            method::lookup_in_trait(fcx,
                                    expr.span,
                                    Some(&*base_expr),
                                    token::intern("index"),
                                    trait_did,
                                    base_ty,
                                    [],
                                    DontAutoderefReceiver,
                                    IgnoreStaticMethods)
        }
        (method, _) => method,
    };

    // Regardless of whether the lookup succeeds, check the method arguments
    // so that we have *some* type for each argument.
    let method_type = get_method_ty(&method);
    check_method_argument_types(fcx,
                                expr.span,
                                method_type,
                                expr,
                                &[index_expr],
                                DoDerefArgs,
                                DontTupleArguments);

    make_return_type(fcx, method_call, method)
}

/// Given the head of a `for` expression, looks up the `next` method in the
/// `Iterator` trait. Fails if the expression does not implement `next`.
///
/// The return type of this function represents the concrete element type
/// `A` in the type `Iterator<A>` that the method returns.
fn lookup_method_for_for_loop(fcx: &FnCtxt,
                              iterator_expr: &ast::Expr,
                              loop_id: ast::NodeId)
                              -> ty::t {
    let trait_did = match fcx.tcx().lang_items.require(IteratorItem) {
        Ok(trait_did) => trait_did,
        Err(ref err_string) => {
            fcx.tcx().sess.span_err(iterator_expr.span,
                                    err_string.as_slice());
            return ty::mk_err()
        }
    };

    let expr_type = fcx.expr_ty(&*iterator_expr);
    let method = method::lookup_in_trait(fcx,
                                         iterator_expr.span,
                                         Some(&*iterator_expr),
                                         token::intern("next"),
                                         trait_did,
                                         expr_type,
                                         [],
                                         DontAutoderefReceiver,
                                         IgnoreStaticMethods);

    // Regardless of whether the lookup succeeds, check the method arguments
    // so that we have *some* type for each argument.
    let method_type = match method {
        Some(ref method) => method.ty,
        None => {
            let true_expr_type = fcx.infcx().resolve_type_vars_if_possible(expr_type);

            if !ty::type_is_error(true_expr_type) {
                let ty_string = fcx.infcx().ty_to_string(true_expr_type);
                fcx.tcx().sess.span_err(iterator_expr.span,
                                        format!("`for` loop expression has type `{}` which does \
                                                 not implement the `Iterator` trait",
                                                ty_string).as_slice());
            }
            ty::mk_err()
        }
    };
    let return_type = check_method_argument_types(fcx,
                                                  iterator_expr.span,
                                                  method_type,
                                                  iterator_expr,
                                                  &[],
                                                  DontDerefArgs,
                                                  DontTupleArguments);

    match method {
        Some(method) => {
            fcx.inh.method_map.borrow_mut().insert(MethodCall::expr(loop_id),
                                                   method);

            // We expect the return type to be `Option` or something like it.
            // Grab the first parameter of its type substitution.
            let return_type = structurally_resolved_type(fcx,
                                                         iterator_expr.span,
                                                         return_type);
            match ty::get(return_type).sty {
                ty::ty_enum(_, ref substs)
                        if !substs.types.is_empty_in(subst::TypeSpace) => {
                    *substs.types.get(subst::TypeSpace, 0)
                }
                _ => {
                    fcx.tcx().sess.span_err(iterator_expr.span,
                                            "`next` method of the `Iterator` \
                                             trait has an unexpected type");
                    ty::mk_err()
                }
            }
        }
        None => ty::mk_err()
    }
}

fn check_method_argument_types<'a>(fcx: &FnCtxt,
                                   sp: Span,
                                   method_fn_ty: ty::t,
                                   callee_expr: &ast::Expr,
                                   args_no_rcvr: &[&'a P<ast::Expr>],
                                   deref_args: DerefArgs,
                                   tuple_arguments: TupleArgumentsFlag)
                                   -> ty::t {
    if ty::type_is_error(method_fn_ty) {
       let err_inputs = err_args(args_no_rcvr.len());
        check_argument_types(fcx,
                             sp,
                             err_inputs.as_slice(),
                             callee_expr,
                             args_no_rcvr,
                             deref_args,
                             false,
                             tuple_arguments);
        method_fn_ty
    } else {
        match ty::get(method_fn_ty).sty {
            ty::ty_bare_fn(ref fty) => {
                // HACK(eddyb) ignore self in the definition (see above).
                check_argument_types(fcx,
                                     sp,
                                     fty.sig.inputs.slice_from(1),
                                     callee_expr,
                                     args_no_rcvr,
                                     deref_args,
                                     fty.sig.variadic,
                                     tuple_arguments);
                fty.sig.output
            }
            _ => {
                fcx.tcx().sess.span_bug(callee_expr.span,
                                        "method without bare fn type");
            }
        }
    }
}

fn check_argument_types<'a>(fcx: &FnCtxt,
                            sp: Span,
                            fn_inputs: &[ty::t],
                            _callee_expr: &ast::Expr,
                            args: &[&'a P<ast::Expr>],
                            deref_args: DerefArgs,
                            variadic: bool,
                            tuple_arguments: TupleArgumentsFlag) {
    /*!
     *
     * Generic function that factors out common logic from
     * function calls, method calls and overloaded operators.
     */

    let tcx = fcx.ccx.tcx;

    // Grab the argument types, supplying fresh type variables
    // if the wrong number of arguments were supplied
    let supplied_arg_count = if tuple_arguments == DontTupleArguments {
        args.len()
    } else {
        1
    };

    let expected_arg_count = fn_inputs.len();
    let formal_tys = if tuple_arguments == TupleArguments {
        let tuple_type = structurally_resolved_type(fcx, sp, fn_inputs[0]);
        match ty::get(tuple_type).sty {
            ty::ty_tup(ref arg_types) => {
                if arg_types.len() != args.len() {
                    span_err!(tcx.sess, sp, E0057,
                        "this function takes {} parameter{} but {} parameter{} supplied",
                        arg_types.len(),
                        if arg_types.len() == 1 {""} else {"s"},
                        args.len(),
                        if args.len() == 1 {" was"} else {"s were"});
                    err_args(args.len())
                } else {
                    (*arg_types).clone()
                }
            }
            ty::ty_nil => {
                if args.len() != 0 {
                    span_err!(tcx.sess, sp, E0058,
                        "this function takes 0 parameters but {} parameter{} supplied",
                        args.len(),
                        if args.len() == 1 {" was"} else {"s were"});
                }
                Vec::new()
            }
            _ => {
                span_err!(tcx.sess, sp, E0059,
                    "cannot use call notation; the first type parameter \
                     for the function trait is neither a tuple nor unit");
                err_args(supplied_arg_count)
            }
        }
    } else if expected_arg_count == supplied_arg_count {
        fn_inputs.iter().map(|a| *a).collect()
    } else if variadic {
        if supplied_arg_count >= expected_arg_count {
            fn_inputs.iter().map(|a| *a).collect()
        } else {
            span_err!(tcx.sess, sp, E0060,
                "this function takes at least {} parameter{} \
                 but {} parameter{} supplied",
                expected_arg_count,
                if expected_arg_count == 1 {""} else {"s"},
                supplied_arg_count,
                if supplied_arg_count == 1 {" was"} else {"s were"});
            err_args(supplied_arg_count)
        }
    } else {
        span_err!(tcx.sess, sp, E0061,
            "this function takes {} parameter{} but {} parameter{} supplied",
            expected_arg_count,
            if expected_arg_count == 1 {""} else {"s"},
            supplied_arg_count,
            if supplied_arg_count == 1 {" was"} else {"s were"});
        err_args(supplied_arg_count)
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
    for check_blocks in xs.iter() {
        let check_blocks = *check_blocks;
        debug!("check_blocks={}", check_blocks);

        // More awful hacks: before we check the blocks, try to do
        // an "opportunistic" vtable resolution of any trait
        // bounds on the call.
        if check_blocks {
            vtable2::select_fcx_obligations_where_possible(fcx);
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
            let is_block = match arg.node {
                ast::ExprFnBlock(..) |
                ast::ExprProc(..) |
                ast::ExprUnboxedFn(..) => true,
                _ => false
            };

            if is_block == check_blocks {
                debug!("checking the argument");
                let mut formal_ty = *formal_tys.get(i);

                match deref_args {
                    DoDerefArgs => {
                        match ty::get(formal_ty).sty {
                            ty::ty_rptr(_, mt) => formal_ty = mt.ty,
                            ty::ty_err => (),
                            _ => {
                                // So we hit this case when one implements the
                                // operator traits but leaves an argument as
                                // just T instead of &T. We'll catch it in the
                                // mismatch impl/trait method phase no need to
                                // ICE here.
                                // See: #11450
                                formal_ty = ty::mk_err();
                            }
                        }
                    }
                    DontDerefArgs => {}
                }

                check_expr_coercable_to_type(fcx, &***arg, formal_ty);
            }
        }
    }

    // We also need to make sure we at least write the ty of the other
    // arguments which we skipped above.
    if variadic {
        for arg in args.iter().skip(expected_arg_count) {
            check_expr(fcx, &***arg);

            // There are a few types which get autopromoted when passed via varargs
            // in C but we just error out instead and require explicit casts.
            let arg_ty = structurally_resolved_type(fcx, arg.span,
                                                    fcx.expr_ty(&***arg));
            match ty::get(arg_ty).sty {
                ty::ty_float(ast::TyF32) => {
                    fcx.type_error_message(arg.span,
                                           |t| {
                        format!("can't pass an {} to variadic \
                                 function, cast to c_double", t)
                    }, arg_ty, None);
                }
                ty::ty_int(ast::TyI8) | ty::ty_int(ast::TyI16) | ty::ty_bool => {
                    fcx.type_error_message(arg.span, |t| {
                        format!("can't pass {} to variadic \
                                 function, cast to c_int",
                                       t)
                    }, arg_ty, None);
                }
                ty::ty_uint(ast::TyU8) | ty::ty_uint(ast::TyU16) => {
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

fn err_args(len: uint) -> Vec<ty::t> {
    Vec::from_fn(len, |_| ty::mk_err())
}

fn write_call(fcx: &FnCtxt, call_expr: &ast::Expr, output: ty::t) {
    fcx.write_ty(call_expr.id, output);
}

// AST fragment checking
fn check_lit(fcx: &FnCtxt,
             lit: &ast::Lit,
             expected: Expectation)
             -> ty::t
{
    let tcx = fcx.ccx.tcx;

    match lit.node {
        ast::LitStr(..) => ty::mk_str_slice(tcx, ty::ReStatic, ast::MutImmutable),
        ast::LitBinary(..) => {
            ty::mk_slice(tcx, ty::ReStatic, ty::mt{ ty: ty::mk_u8(), mutbl: ast::MutImmutable })
        }
        ast::LitByte(_) => ty::mk_u8(),
        ast::LitChar(_) => ty::mk_char(),
        ast::LitInt(_, ast::SignedIntLit(t, _)) => ty::mk_mach_int(t),
        ast::LitInt(_, ast::UnsignedIntLit(t)) => ty::mk_mach_uint(t),
        ast::LitInt(_, ast::UnsuffixedIntLit(_)) => {
            let opt_ty = expected.map_to_option(fcx, |sty| {
                match *sty {
                    ty::ty_int(i) => Some(ty::mk_mach_int(i)),
                    ty::ty_uint(i) => Some(ty::mk_mach_uint(i)),
                    ty::ty_char => Some(ty::mk_mach_uint(ast::TyU8)),
                    ty::ty_ptr(..) => Some(ty::mk_mach_uint(ast::TyU)),
                    ty::ty_bare_fn(..) => Some(ty::mk_mach_uint(ast::TyU)),
                    _ => None
                }
            });
            opt_ty.unwrap_or_else(
                || ty::mk_int_var(tcx, fcx.infcx().next_int_var_id()))
        }
        ast::LitFloat(_, t) => ty::mk_mach_float(t),
        ast::LitFloatUnsuffixed(_) => {
            let opt_ty = expected.map_to_option(fcx, |sty| {
                match *sty {
                    ty::ty_float(i) => Some(ty::mk_mach_float(i)),
                    _ => None
                }
            });
            opt_ty.unwrap_or_else(
                || ty::mk_float_var(tcx, fcx.infcx().next_float_var_id()))
        }
        ast::LitNil => ty::mk_nil(),
        ast::LitBool(_) => ty::mk_bool()
    }
}

pub fn valid_range_bounds(ccx: &CrateCtxt,
                          from: &ast::Expr,
                          to: &ast::Expr)
                       -> Option<bool> {
    match const_eval::compare_lit_exprs(ccx.tcx, from, to) {
        Some(val) => Some(val <= 0),
        None => None
    }
}

pub fn check_expr_has_type(fcx: &FnCtxt,
                           expr: &ast::Expr,
                           expected: ty::t) {
    check_expr_with_unifier(
        fcx, expr, ExpectHasType(expected), NoPreference,
        || demand::suptype(fcx, expr.span, expected, fcx.expr_ty(expr)));
}

fn check_expr_coercable_to_type(fcx: &FnCtxt,
                                expr: &ast::Expr,
                                expected: ty::t) {
    check_expr_with_unifier(
        fcx, expr, ExpectHasType(expected), NoPreference,
        || demand::coerce(fcx, expr.span, expected, expr));
}

fn check_expr_with_hint(fcx: &FnCtxt, expr: &ast::Expr, expected: ty::t) {
    check_expr_with_unifier(
        fcx, expr, ExpectHasType(expected), NoPreference,
        || ())
}

fn check_expr_with_expectation(fcx: &FnCtxt,
                               expr: &ast::Expr,
                               expected: Expectation) {
    check_expr_with_unifier(
        fcx, expr, expected, NoPreference,
        || ())
}

fn check_expr_with_expectation_and_lvalue_pref(fcx: &FnCtxt,
                                            expr: &ast::Expr,
                                            expected: Expectation,
                                            lvalue_pref: LvaluePreference)
{
    check_expr_with_unifier(fcx, expr, expected, lvalue_pref, || ())
}

fn check_expr(fcx: &FnCtxt, expr: &ast::Expr)  {
    check_expr_with_unifier(fcx, expr, NoExpectation, NoPreference, || ())
}

fn check_expr_with_lvalue_pref(fcx: &FnCtxt, expr: &ast::Expr,
                               lvalue_pref: LvaluePreference)  {
    check_expr_with_unifier(fcx, expr, NoExpectation, lvalue_pref, || ())
}

// determine the `self` type, using fresh variables for all variables
// declared on the impl declaration e.g., `impl<A,B> for ~[(A,B)]`
// would return ($0, $1) where $0 and $1 are freshly instantiated type
// variables.
pub fn impl_self_ty(fcx: &FnCtxt,
                    span: Span, // (potential) receiver for this impl
                    did: ast::DefId)
                    -> TypeAndSubsts {
    let tcx = fcx.tcx();

    let ity = ty::lookup_item_type(tcx, did);
    let (n_tps, rps, raw_ty) =
        (ity.generics.types.len(subst::TypeSpace),
         ity.generics.regions.get_slice(subst::TypeSpace),
         ity.ty);

    let rps = fcx.inh.infcx.region_vars_for_defs(span, rps);
    let tps = fcx.inh.infcx.next_ty_vars(n_tps);
    let substs = subst::Substs::new_type(tps, rps);
    let substd_ty = raw_ty.subst(tcx, &substs);

    TypeAndSubsts { substs: substs, ty: substd_ty }
}

// Only for fields! Returns <none> for methods>
// Indifferent to privacy flags
pub fn lookup_field_ty(tcx: &ty::ctxt,
                       class_id: ast::DefId,
                       items: &[ty::field_ty],
                       fieldname: ast::Name,
                       substs: &subst::Substs) -> Option<ty::t> {

    let o_field = items.iter().find(|f| f.name == fieldname);
    o_field.map(|f| ty::lookup_field_type(tcx, class_id, f.id, substs))
}

pub fn lookup_tup_field_ty(tcx: &ty::ctxt,
                           class_id: ast::DefId,
                           items: &[ty::field_ty],
                           idx: uint,
                           substs: &subst::Substs) -> Option<ty::t> {

    let o_field = if idx < items.len() { Some(&items[idx]) } else { None };
    o_field.map(|f| ty::lookup_field_type(tcx, class_id, f.id, substs))
}

// Controls whether the arguments are automatically referenced. This is useful
// for overloaded binary and unary operators.
pub enum DerefArgs {
    DontDerefArgs,
    DoDerefArgs
}

/// Controls whether the arguments are tupled. This is used for the call
/// operator.
///
/// Tupling means that all call-side arguments are packed into a tuple and
/// passed as a single parameter. For example, if tupling is enabled, this
/// function:
///
///     fn f(x: (int, int))
///
/// Can be called as:
///
///     f(1, 2);
///
/// Instead of:
///
///     f((1, 2));
#[deriving(Clone, Eq, PartialEq)]
enum TupleArgumentsFlag {
    DontTupleArguments,
    TupleArguments,
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
fn check_expr_with_unifier(fcx: &FnCtxt,
                           expr: &ast::Expr,
                           expected: Expectation,
                           lvalue_pref: LvaluePreference,
                           unifier: ||)
{
    debug!(">> typechecking: expr={} expected={}",
           expr.repr(fcx.tcx()), expected.repr(fcx.tcx()));

    // A generic function for doing all of the checking for call expressions
    fn check_call<'a>(fcx: &FnCtxt,
                      call_expr: &ast::Expr,
                      f: &ast::Expr,
                      args: &[&'a P<ast::Expr>]) {
        // Store the type of `f` as the type of the callee
        let fn_ty = fcx.expr_ty(f);

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
            ty::ty_closure(box ty::ClosureTy {sig: ref sig, ..}) => sig,
            _ => {
                fcx.type_error_message(call_expr.span, |actual| {
                    format!("expected function, found `{}`", actual)
                }, fn_ty, None);
                &error_fn_sig
            }
        };

        // Replace any bound regions that appear in the function
        // signature with region variables
        let (_, fn_sig) = replace_late_bound_regions_in_fn_sig(fcx.tcx(), fn_sig, |br| {
            fcx.infcx().next_region_var(infer::LateBoundRegion(call_expr.span, br))
        });

        // Call the generic checker.
        check_argument_types(fcx,
                             call_expr.span,
                             fn_sig.inputs.as_slice(),
                             f,
                             args,
                             DontDerefArgs,
                             fn_sig.variadic,
                             DontTupleArguments);

        write_call(fcx, call_expr, fn_sig.output);
    }

    // Checks a method call.
    fn check_method_call(fcx: &FnCtxt,
                         expr: &ast::Expr,
                         method_name: ast::SpannedIdent,
                         args: &[P<ast::Expr>],
                         tps: &[P<ast::Ty>]) {
        let rcvr = &*args[0];
        // We can't know if we need &mut self before we look up the method,
        // so treat the receiver as mutable just in case - only explicit
        // overloaded dereferences care about the distinction.
        check_expr_with_lvalue_pref(fcx, &*rcvr, PreferMutLvalue);

        // no need to check for bot/err -- callee does that
        let expr_t = structurally_resolved_type(fcx,
                                                expr.span,
                                                fcx.expr_ty(&*rcvr));

        let tps = tps.iter().map(|ast_ty| fcx.to_ty(&**ast_ty)).collect::<Vec<_>>();
        let fn_ty = match method::lookup(fcx, expr, &*rcvr,
                                         method_name.node.name,
                                         expr_t, tps.as_slice(),
                                         DontDerefArgs,
                                         CheckTraitsAndInherentMethods,
                                         AutoderefReceiver, IgnoreStaticMethods) {
            Some(method) => {
                let method_ty = method.ty;
                let method_call = MethodCall::expr(expr.id);
                fcx.inh.method_map.borrow_mut().insert(method_call, method);
                method_ty
            }
            None => {
                debug!("(checking method call) failing expr is {}", expr.id);

                fcx.type_error_message(method_name.span,
                  |actual| {
                      format!("type `{}` does not implement any \
                               method in scope named `{}`",
                              actual,
                              token::get_ident(method_name.node))
                  },
                  expr_t,
                  None);

                // Add error type for the result
                fcx.write_error(expr.id);

                // Check for potential static matches (missing self parameters)
                method::lookup(fcx,
                               expr,
                               &*rcvr,
                               method_name.node.name,
                               expr_t,
                               tps.as_slice(),
                               DontDerefArgs,
                               CheckTraitsAndInherentMethods,
                               DontAutoderefReceiver,
                               ReportStaticMethods);

                ty::mk_err()
            }
        };

        // Call the generic checker.
        let args: Vec<_> = args[1..].iter().map(|x| x).collect();
        let ret_ty = check_method_argument_types(fcx,
                                                 method_name.span,
                                                 fn_ty,
                                                 expr,
                                                 args.as_slice(),
                                                 DontDerefArgs,
                                                 DontTupleArguments);

        write_call(fcx, expr, ret_ty);
    }

    // A generic function for checking the then and else in an if
    // or if-check
    fn check_then_else(fcx: &FnCtxt,
                       cond_expr: &ast::Expr,
                       then_blk: &ast::Block,
                       opt_else_expr: Option<&ast::Expr>,
                       id: ast::NodeId,
                       sp: Span,
                       expected: Expectation) {
        check_expr_has_type(fcx, cond_expr, ty::mk_bool());

        let branches_ty = match opt_else_expr {
            Some(ref else_expr) => {
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
                let expected = match expected.only_has_type() {
                    ExpectHasType(ety) => {
                        match infer::resolve_type(fcx.infcx(), Some(sp), ety, force_tvar) {
                            Ok(rty) if !ty::type_is_ty_var(rty) => ExpectHasType(rty),
                            _ => NoExpectation
                        }
                    }
                    _ => NoExpectation
                };
                check_block_with_expected(fcx, then_blk, expected);
                let then_ty = fcx.node_ty(then_blk.id);
                check_expr_with_expectation(fcx, &**else_expr, expected);
                let else_ty = fcx.expr_ty(&**else_expr);
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

    fn lookup_op_method<'a, 'tcx>(fcx: &'a FnCtxt<'a, 'tcx>,
                                  op_ex: &ast::Expr,
                                  lhs_ty: ty::t,
                                  opname: ast::Name,
                                  trait_did: Option<ast::DefId>,
                                  lhs: &'a ast::Expr,
                                  rhs: Option<&P<ast::Expr>>,
                                  autoderef_receiver: AutoderefReceiverFlag,
                                  unbound_method: ||) -> ty::t {
        let method = match trait_did {
            Some(trait_did) => {
                method::lookup_in_trait(fcx, op_ex.span, Some(lhs), opname,
                                        trait_did, lhs_ty, &[], autoderef_receiver,
                                        IgnoreStaticMethods)
            }
            None => None
        };
        let args = match rhs {
            Some(rhs) => vec![rhs],
            None => vec![]
        };
        match method {
            Some(method) => {
                let method_ty = method.ty;
                // HACK(eddyb) Fully qualified path to work around a resolve bug.
                let method_call = ::middle::typeck::MethodCall::expr(op_ex.id);
                fcx.inh.method_map.borrow_mut().insert(method_call, method);
                check_method_argument_types(fcx,
                                            op_ex.span,
                                            method_ty,
                                            op_ex,
                                            args.as_slice(),
                                            DoDerefArgs,
                                            DontTupleArguments)
            }
            None => {
                unbound_method();
                // Check the args anyway
                // so we get all the error messages
                let expected_ty = ty::mk_err();
                check_method_argument_types(fcx,
                                            op_ex.span,
                                            expected_ty,
                                            op_ex,
                                            args.as_slice(),
                                            DoDerefArgs,
                                            DontTupleArguments);
                ty::mk_err()
            }
        }
    }

    // could be either an expr_binop or an expr_assign_binop
    fn check_binop(fcx: &FnCtxt,
                   expr: &ast::Expr,
                   op: ast::BinOp,
                   lhs: &ast::Expr,
                   rhs: &P<ast::Expr>,
                   is_binop_assignment: IsBinopAssignment) {
        let tcx = fcx.ccx.tcx;

        let lvalue_pref = match is_binop_assignment {
            BinopAssignment => PreferMutLvalue,
            SimpleBinop => NoPreference
        };
        check_expr_with_lvalue_pref(fcx, &*lhs, lvalue_pref);

        // Callee does bot / err checking
        let lhs_t = structurally_resolved_type(fcx, lhs.span,
                                               fcx.expr_ty(&*lhs));

        if ty::type_is_integral(lhs_t) && ast_util::is_shift_binop(op) {
            // Shift is a special case: rhs must be uint, no matter what lhs is
            check_expr_has_type(fcx, &**rhs, ty::mk_uint());
            fcx.write_ty(expr.id, lhs_t);
            return;
        }

        if ty::is_binopable(tcx, lhs_t, op) {
            let tvar = fcx.infcx().next_ty_var();
            demand::suptype(fcx, expr.span, tvar, lhs_t);
            check_expr_has_type(fcx, &**rhs, tvar);

            let result_t = match op {
                ast::BiEq | ast::BiNe | ast::BiLt | ast::BiLe | ast::BiGe |
                ast::BiGt => {
                    if ty::type_is_simd(tcx, lhs_t) {
                        if ty::type_is_fp(ty::simd_type(tcx, lhs_t)) {
                            fcx.type_error_message(expr.span,
                                |actual| {
                                    format!("binary comparison \
                                             operation `{}` not \
                                             supported for floating \
                                             point SIMD vector `{}`",
                                            ast_util::binop_to_string(op),
                                            actual)
                                },
                                lhs_t,
                                None
                            );
                            ty::mk_err()
                        } else {
                            lhs_t
                        }
                    } else {
                        ty::mk_bool()
                    }
                },
                _ => lhs_t,
            };

            fcx.write_ty(expr.id, result_t);
            return;
        }

        if op == ast::BiOr || op == ast::BiAnd {
            // This is an error; one of the operands must have the wrong
            // type
            fcx.write_error(expr.id);
            fcx.write_error(rhs.id);
            fcx.type_error_message(expr.span,
                                   |actual| {
                    format!("binary operation `{}` cannot be applied \
                             to type `{}`",
                            ast_util::binop_to_string(op),
                            actual)
                },
                lhs_t,
                None)
        }

        // Check for overloaded operators if not an assignment.
        let result_t = if is_binop_assignment == SimpleBinop {
            check_user_binop(fcx, expr, lhs, lhs_t, op, rhs)
        } else {
            fcx.type_error_message(expr.span,
                                   |actual| {
                                        format!("binary assignment \
                                                 operation `{}=` \
                                                 cannot be applied to \
                                                 type `{}`",
                                                ast_util::binop_to_string(op),
                                                actual)
                                   },
                                   lhs_t,
                                   None);
            check_expr(fcx, &**rhs);
            ty::mk_err()
        };

        fcx.write_ty(expr.id, result_t);
        if ty::type_is_error(result_t) {
            fcx.write_ty(rhs.id, result_t);
        }
    }

    fn check_user_binop(fcx: &FnCtxt,
                        ex: &ast::Expr,
                        lhs_expr: &ast::Expr,
                        lhs_resolved_t: ty::t,
                        op: ast::BinOp,
                        rhs: &P<ast::Expr>) -> ty::t {
        let tcx = fcx.ccx.tcx;
        let lang = &tcx.lang_items;
        let (name, trait_did) = match op {
            ast::BiAdd => ("add", lang.add_trait()),
            ast::BiSub => ("sub", lang.sub_trait()),
            ast::BiMul => ("mul", lang.mul_trait()),
            ast::BiDiv => ("div", lang.div_trait()),
            ast::BiRem => ("rem", lang.rem_trait()),
            ast::BiBitXor => ("bitxor", lang.bitxor_trait()),
            ast::BiBitAnd => ("bitand", lang.bitand_trait()),
            ast::BiBitOr => ("bitor", lang.bitor_trait()),
            ast::BiShl => ("shl", lang.shl_trait()),
            ast::BiShr => ("shr", lang.shr_trait()),
            ast::BiLt => ("lt", lang.ord_trait()),
            ast::BiLe => ("le", lang.ord_trait()),
            ast::BiGe => ("ge", lang.ord_trait()),
            ast::BiGt => ("gt", lang.ord_trait()),
            ast::BiEq => ("eq", lang.eq_trait()),
            ast::BiNe => ("ne", lang.eq_trait()),
            ast::BiAnd | ast::BiOr => {
                check_expr(fcx, &**rhs);
                return ty::mk_err();
            }
        };
        lookup_op_method(fcx, ex, lhs_resolved_t, token::intern(name),
                         trait_did, lhs_expr, Some(rhs), DontAutoderefReceiver, || {
            fcx.type_error_message(ex.span, |actual| {
                format!("binary operation `{}` cannot be applied to type `{}`",
                        ast_util::binop_to_string(op),
                        actual)
            }, lhs_resolved_t, None)
        })
    }

    fn check_user_unop(fcx: &FnCtxt,
                       op_str: &str,
                       mname: &str,
                       trait_did: Option<ast::DefId>,
                       ex: &ast::Expr,
                       rhs_expr: &ast::Expr,
                       rhs_t: ty::t) -> ty::t {
       lookup_op_method(fcx, ex, rhs_t, token::intern(mname),
                        trait_did, rhs_expr, None, DontAutoderefReceiver, || {
            fcx.type_error_message(ex.span, |actual| {
                format!("cannot apply unary operator `{}` to type `{}`",
                        op_str, actual)
            }, rhs_t, None);
        })
    }

    fn check_unboxed_closure(fcx: &FnCtxt,
                             expr: &ast::Expr,
                             kind: ast::UnboxedClosureKind,
                             decl: &ast::FnDecl,
                             body: &ast::Block) {
        let mut fn_ty = astconv::ty_of_closure(
            fcx,
            expr.id,
            ast::NormalFn,
            ast::Many,

            // The `RegionTraitStore` and region_existential_bounds
            // are lies, but we ignore them so it doesn't matter.
            //
            // FIXME(pcwalton): Refactor this API.
            ty::region_existential_bound(ty::ReStatic),
            ty::RegionTraitStore(ty::ReStatic, ast::MutImmutable),

            decl,
            abi::RustCall,
            None);

        let region = match fcx.infcx().anon_regions(expr.span, 1) {
            Err(_) => {
                fcx.ccx.tcx.sess.span_bug(expr.span,
                                          "can't make anon regions here?!")
            }
            Ok(regions) => *regions.get(0),
        };
        let closure_type = ty::mk_unboxed_closure(fcx.ccx.tcx,
                                                  local_def(expr.id),
                                                  region);
        fcx.write_ty(expr.id, closure_type);

        check_fn(fcx.ccx,
                 ast::NormalFn,
                 expr.id,
                 &fn_ty.sig,
                 decl,
                 expr.id,
                 &*body,
                 fcx.inh);

        // Tuple up the arguments and insert the resulting function type into
        // the `unboxed_closures` table.
        fn_ty.sig.inputs = if fn_ty.sig.inputs.len() == 0 {
            vec![ty::mk_nil()]
        } else {
            vec![ty::mk_tup(fcx.tcx(), fn_ty.sig.inputs)]
        };

        let kind = match kind {
            ast::FnUnboxedClosureKind => ty::FnUnboxedClosureKind,
            ast::FnMutUnboxedClosureKind => ty::FnMutUnboxedClosureKind,
            ast::FnOnceUnboxedClosureKind => ty::FnOnceUnboxedClosureKind,
        };

        let unboxed_closure = ty::UnboxedClosure {
            closure_type: fn_ty,
            kind: kind,
        };

        fcx.inh
           .unboxed_closures
           .borrow_mut()
           .insert(local_def(expr.id), unboxed_closure);
    }

    fn check_expr_fn(fcx: &FnCtxt,
                     expr: &ast::Expr,
                     store: ty::TraitStore,
                     decl: &ast::FnDecl,
                     body: &ast::Block,
                     expected: Expectation) {
        let tcx = fcx.ccx.tcx;

        // Find the expected input/output types (if any). Substitute
        // fresh bound regions for any bound regions we find in the
        // expected types so as to avoid capture.
        let expected_sty = expected.map_to_option(fcx, |x| Some((*x).clone()));
        let (expected_sig,
             expected_onceness,
             expected_bounds) = {
            match expected_sty {
                Some(ty::ty_closure(ref cenv)) => {
                    let (_, sig) =
                        replace_late_bound_regions_in_fn_sig(
                            tcx, &cenv.sig,
                            |_| fcx.inh.infcx.fresh_bound_region(expr.id));
                    let onceness = match (&store, &cenv.store) {
                        // As the closure type and onceness go, only three
                        // combinations are legit:
                        //      once closure
                        //      many closure
                        //      once proc
                        // If the actual and expected closure type disagree with
                        // each other, set expected onceness to be always Once or
                        // Many according to the actual type. Otherwise, it will
                        // yield either an illegal "many proc" or a less known
                        // "once closure" in the error message.
                        (&ty::UniqTraitStore, &ty::UniqTraitStore) |
                        (&ty::RegionTraitStore(..), &ty::RegionTraitStore(..)) =>
                            cenv.onceness,
                        (&ty::UniqTraitStore, _) => ast::Once,
                        (&ty::RegionTraitStore(..), _) => ast::Many,
                    };
                    (Some(sig), onceness, cenv.bounds)
                }
                _ => {
                    // Not an error! Means we're inferring the closure type
                    let (bounds, onceness) = match expr.node {
                        ast::ExprProc(..) => {
                            let mut bounds = ty::region_existential_bound(ty::ReStatic);
                            bounds.builtin_bounds.add(ty::BoundSend); // FIXME
                            (bounds, ast::Once)
                        }
                        _ => {
                            let region = fcx.infcx().next_region_var(
                                infer::AddrOfRegion(expr.span));
                            (ty::region_existential_bound(region), ast::Many)
                        }
                    };
                    (None, onceness, bounds)
                }
            }
        };

        // construct the function type
        let fn_ty = astconv::ty_of_closure(fcx,
                                           expr.id,
                                           ast::NormalFn,
                                           expected_onceness,
                                           expected_bounds,
                                           store,
                                           decl,
                                           abi::Rust,
                                           expected_sig);
        let fty_sig = fn_ty.sig.clone();
        let fty = ty::mk_closure(tcx, fn_ty);
        debug!("check_expr_fn fty={}", fcx.infcx().ty_to_string(fty));

        fcx.write_ty(expr.id, fty);

        // If the closure is a stack closure and hasn't had some non-standard
        // style inferred for it, then check it under its parent's style.
        // Otherwise, use its own
        let (inherited_style, inherited_style_id) = match store {
            ty::RegionTraitStore(..) => (fcx.ps.borrow().fn_style,
                                         fcx.ps.borrow().def),
            ty::UniqTraitStore => (ast::NormalFn, expr.id)
        };

        check_fn(fcx.ccx,
                 inherited_style,
                 inherited_style_id,
                 &fty_sig,
                 &*decl,
                 expr.id,
                 &*body,
                 fcx.inh);
    }


    // Check field access expressions
    fn check_field(fcx: &FnCtxt,
                   expr: &ast::Expr,
                   lvalue_pref: LvaluePreference,
                   base: &ast::Expr,
                   field: &ast::SpannedIdent,
                   tys: &[P<ast::Ty>]) {
        let tcx = fcx.ccx.tcx;
        check_expr_with_lvalue_pref(fcx, base, lvalue_pref);
        let expr_t = structurally_resolved_type(fcx, expr.span,
                                                fcx.expr_ty(base));
        // FIXME(eddyb) #12808 Integrate privacy into this auto-deref loop.
        let (_, autoderefs, field_ty) =
            autoderef(fcx, expr.span, expr_t, Some(base.id), lvalue_pref, |base_t, _| {
                match ty::get(base_t).sty {
                    ty::ty_struct(base_id, ref substs) => {
                        debug!("struct named {}", ppaux::ty_to_string(tcx, base_t));
                        let fields = ty::lookup_struct_fields(tcx, base_id);
                        lookup_field_ty(tcx, base_id, fields.as_slice(),
                                        field.node.name, &(*substs))
                    }
                    _ => None
                }
            });
        match field_ty {
            Some(field_ty) => {
                fcx.write_ty(expr.id, field_ty);
                fcx.write_autoderef_adjustment(base.id, base.span, autoderefs);
                return;
            }
            None => {}
        }

        let tps: Vec<ty::t> = tys.iter().map(|ty| fcx.to_ty(&**ty)).collect();
        match method::lookup(fcx,
                             expr,
                             base,
                             field.node.name,
                             expr_t,
                             tps.as_slice(),
                             DontDerefArgs,
                             CheckTraitsAndInherentMethods,
                             AutoderefReceiver,
                             IgnoreStaticMethods) {
            Some(_) => {
                fcx.type_error_message(
                    field.span,
                    |actual| {
                        format!("attempted to take value of method `{}` on type \
                                 `{}`", token::get_ident(field.node), actual)
                    },
                    expr_t, None);

                tcx.sess.span_note(field.span,
                    "maybe a missing `()` to call it? If not, try an anonymous function.");
            }

            None => {
                fcx.type_error_message(
                    expr.span,
                    |actual| {
                        format!("attempted access of field `{}` on \
                                        type `{}`, but no field with that \
                                        name was found",
                                       token::get_ident(field.node),
                                       actual)
                    },
                    expr_t, None);
            }
        }

        fcx.write_error(expr.id);
    }

    // Check tuple index expressions
    fn check_tup_field(fcx: &FnCtxt,
                       expr: &ast::Expr,
                       lvalue_pref: LvaluePreference,
                       base: &ast::Expr,
                       idx: codemap::Spanned<uint>,
                       _tys: &[P<ast::Ty>]) {
        let tcx = fcx.ccx.tcx;
        check_expr_with_lvalue_pref(fcx, base, lvalue_pref);
        let expr_t = structurally_resolved_type(fcx, expr.span,
                                                fcx.expr_ty(base));
        let mut tuple_like = false;
        // FIXME(eddyb) #12808 Integrate privacy into this auto-deref loop.
        let (_, autoderefs, field_ty) =
            autoderef(fcx, expr.span, expr_t, Some(base.id), lvalue_pref, |base_t, _| {
                match ty::get(base_t).sty {
                    ty::ty_struct(base_id, ref substs) => {
                        tuple_like = ty::is_tuple_struct(tcx, base_id);
                        if tuple_like {
                            debug!("tuple struct named {}", ppaux::ty_to_string(tcx, base_t));
                            let fields = ty::lookup_struct_fields(tcx, base_id);
                            lookup_tup_field_ty(tcx, base_id, fields.as_slice(),
                                                idx.node, &(*substs))
                        } else {
                            None
                        }
                    }
                    ty::ty_tup(ref v) => {
                        tuple_like = true;
                        if idx.node < v.len() { Some(v[idx.node]) } else { None }
                    }
                    _ => None
                }
            });
        match field_ty {
            Some(field_ty) => {
                fcx.write_ty(expr.id, field_ty);
                fcx.write_autoderef_adjustment(base.id, base.span, autoderefs);
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

    fn check_struct_or_variant_fields(fcx: &FnCtxt,
                                      struct_ty: ty::t,
                                      span: Span,
                                      class_id: ast::DefId,
                                      node_id: ast::NodeId,
                                      substitutions: subst::Substs,
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
                                  actual, token::get_ident(field.ident.node))
                      },
                      struct_ty,
                      None);
                    error_happened = true;
                }
                Some((_, true)) => {
                    span_err!(fcx.tcx().sess, field.ident.span, E0062,
                        "field `{}` specified more than once",
                        token::get_ident(field.ident.node));
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
                    &*field.expr,
                    expected_field_type);
        }

        if error_happened {
            fcx.write_error(node_id);
        }

        if check_completeness && !error_happened {
            // Make sure the programmer specified all the fields.
            assert!(fields_found <= field_types.len());
            if fields_found < field_types.len() {
                let mut missing_fields = Vec::new();
                for class_field in field_types.iter() {
                    let name = class_field.name;
                    let (_, seen) = *class_field_map.get(&name);
                    if !seen {
                        missing_fields.push(
                            format!("`{}`", token::get_name(name).get()))
                    }
                }

                span_err!(tcx.sess, span, E0063,
                    "missing field{}: {}",
                    if missing_fields.len() == 1 {""} else {"s"},
                    missing_fields.connect(", "));
             }
        }

        if !error_happened {
            fcx.write_ty(node_id, ty::mk_struct(fcx.ccx.tcx,
                                class_id, substitutions));
        }
    }

    fn check_struct_constructor(fcx: &FnCtxt,
                                id: ast::NodeId,
                                span: codemap::Span,
                                class_id: ast::DefId,
                                fields: &[ast::Field],
                                base_expr: Option<&ast::Expr>) {
        let tcx = fcx.ccx.tcx;

        // Generate the struct type.
        let TypeAndSubsts {
            ty: mut struct_type,
            substs: struct_substs
        } = fcx.instantiate_item_type(span, class_id);

        // Look up and check the fields.
        let class_fields = ty::lookup_struct_fields(tcx, class_id);
        check_struct_or_variant_fields(fcx,
                                       struct_type,
                                       span,
                                       class_id,
                                       id,
                                       struct_substs,
                                       class_fields.as_slice(),
                                       fields,
                                       base_expr.is_none());
        if ty::type_is_error(fcx.node_ty(id)) {
            struct_type = ty::mk_err();
        }

        // Check the base expression if necessary.
        match base_expr {
            None => {}
            Some(base_expr) => {
                check_expr_has_type(fcx, &*base_expr, struct_type);
                if ty::type_is_bot(fcx.node_ty(base_expr.id)) {
                    struct_type = ty::mk_bot();
                }
            }
        }

        // Write in the resulting type.
        fcx.write_ty(id, struct_type);
    }

    fn check_struct_enum_variant(fcx: &FnCtxt,
                                 id: ast::NodeId,
                                 span: codemap::Span,
                                 enum_id: ast::DefId,
                                 variant_id: ast::DefId,
                                 fields: &[ast::Field]) {
        let tcx = fcx.ccx.tcx;

        // Look up the number of type parameters and the raw type, and
        // determine whether the enum is region-parameterized.
        let TypeAndSubsts {
            ty: enum_type,
            substs: substitutions
        } = fcx.instantiate_item_type(span, enum_id);

        // Look up and check the enum variant fields.
        let variant_fields = ty::lookup_struct_fields(tcx, variant_id);
        check_struct_or_variant_fields(fcx,
                                       enum_type,
                                       span,
                                       variant_id,
                                       id,
                                       substitutions,
                                       variant_fields.as_slice(),
                                       fields,
                                       true);
        fcx.write_ty(id, enum_type);
    }

    fn check_struct_fields_on_error(fcx: &FnCtxt,
                                    id: ast::NodeId,
                                    fields: &[ast::Field],
                                    base_expr: &Option<P<ast::Expr>>) {
        // Make sure to still write the types
        // otherwise we might ICE
        fcx.write_error(id);
        for field in fields.iter() {
            check_expr(fcx, &*field.expr);
        }
        match *base_expr {
            Some(ref base) => check_expr(fcx, &**base),
            None => {}
        }
    }

    type ExprCheckerWithTy = fn(&FnCtxt, &ast::Expr, ty::t);

    let tcx = fcx.ccx.tcx;
    let id = expr.id;
    match expr.node {
      ast::ExprBox(ref place, ref subexpr) => {
          check_expr(fcx, &**place);
          check_expr(fcx, &**subexpr);

          let mut checked = false;
          match place.node {
              ast::ExprPath(ref path) => {
                  // FIXME(pcwalton): For now we hardcode the two permissible
                  // places: the exchange heap and the managed heap.
                  let definition = lookup_def(fcx, path.span, place.id);
                  let def_id = definition.def_id();
                  let referent_ty = fcx.expr_ty(&**subexpr);
                  if tcx.lang_items.exchange_heap() == Some(def_id) {
                      fcx.write_ty(id, ty::mk_uniq(tcx, referent_ty));
                      checked = true
                  }
              }
              _ => {}
          }

          if !checked {
              span_err!(tcx.sess, expr.span, E0066,
                  "only the managed heap and exchange heap are currently supported");
              fcx.write_ty(id, ty::mk_err());
          }
      }

      ast::ExprLit(ref lit) => {
        let typ = check_lit(fcx, &**lit, expected);
        fcx.write_ty(id, typ);
      }
      ast::ExprBinary(op, ref lhs, ref rhs) => {
        check_binop(fcx, expr, op, &**lhs, rhs, SimpleBinop);

        let lhs_ty = fcx.expr_ty(&**lhs);
        let rhs_ty = fcx.expr_ty(&**rhs);
        if ty::type_is_error(lhs_ty) ||
            ty::type_is_error(rhs_ty) {
            fcx.write_error(id);
        }
        else if ty::type_is_bot(lhs_ty) ||
          (ty::type_is_bot(rhs_ty) && !ast_util::lazy_binop(op)) {
            fcx.write_bot(id);
        }
      }
      ast::ExprAssignOp(op, ref lhs, ref rhs) => {
        check_binop(fcx, expr, op, &**lhs, rhs, BinopAssignment);

        let lhs_t = fcx.expr_ty(&**lhs);
        let result_t = fcx.expr_ty(expr);
        demand::suptype(fcx, expr.span, result_t, lhs_t);

        let tcx = fcx.tcx();
        if !ty::expr_is_lval(tcx, &**lhs) {
            span_err!(tcx.sess, lhs.span, E0067, "illegal left-hand side expression");
        }

        fcx.require_expr_have_sized_type(&**lhs, traits::AssignmentLhsSized);

        // Overwrite result of check_binop...this preserves existing behavior
        // but seems quite dubious with regard to user-defined methods
        // and so forth. - Niko
        if !ty::type_is_error(result_t)
            && !ty::type_is_bot(result_t) {
            fcx.write_nil(expr.id);
        }
      }
      ast::ExprUnary(unop, ref oprnd) => {
        let expected_inner = expected.map(fcx, |sty| {
            match unop {
                ast::UnUniq => match *sty {
                    ty::ty_uniq(ty) => {
                        ExpectHasType(ty)
                    }
                    _ => {
                        NoExpectation
                    }
                },
                ast::UnNot | ast::UnNeg => {
                    expected
                }
                ast::UnDeref => {
                    NoExpectation
                }
            }
        });
        let lvalue_pref = match unop {
            ast::UnDeref => lvalue_pref,
            _ => NoPreference
        };
        check_expr_with_expectation_and_lvalue_pref(
            fcx, &**oprnd, expected_inner, lvalue_pref);
        let mut oprnd_t = fcx.expr_ty(&**oprnd);

        if !ty::type_is_error(oprnd_t) {
            match unop {
                ast::UnUniq => {
                    if !ty::type_is_bot(oprnd_t) {
                        oprnd_t = ty::mk_uniq(tcx, oprnd_t);
                    }
                }
                ast::UnDeref => {
                    oprnd_t = structurally_resolved_type(fcx, expr.span, oprnd_t);
                    oprnd_t = match ty::deref(oprnd_t, true) {
                        Some(mt) => mt.ty,
                        None => match try_overloaded_deref(fcx, expr.span,
                                                           Some(MethodCall::expr(expr.id)),
                                                           Some(&**oprnd), oprnd_t, lvalue_pref) {
                            Some(mt) => mt.ty,
                            None => {
                                let is_newtype = match ty::get(oprnd_t).sty {
                                    ty::ty_struct(did, ref substs) => {
                                        let fields = ty::struct_fields(fcx.tcx(), did, substs);
                                        fields.len() == 1
                                        && fields.get(0).ident ==
                                        token::special_idents::unnamed_field
                                    }
                                    _ => false
                                };
                                if is_newtype {
                                    // This is an obsolete struct deref
                                    span_err!(tcx.sess, expr.span, E0068,
                                        "single-field tuple-structs can \
                                         no longer be dereferenced");
                                } else {
                                    fcx.type_error_message(expr.span, |actual| {
                                        format!("type `{}` cannot be \
                                                dereferenced", actual)
                                    }, oprnd_t, None);
                                }
                                ty::mk_err()
                            }
                        }
                    };
                }
                ast::UnNot => {
                    if !ty::type_is_bot(oprnd_t) {
                        oprnd_t = structurally_resolved_type(fcx, oprnd.span,
                                                             oprnd_t);
                        if !(ty::type_is_integral(oprnd_t) ||
                             ty::get(oprnd_t).sty == ty::ty_bool) {
                            oprnd_t = check_user_unop(fcx, "!", "not",
                                                      tcx.lang_items.not_trait(),
                                                      expr, &**oprnd, oprnd_t);
                        }
                    }
                }
                ast::UnNeg => {
                    if !ty::type_is_bot(oprnd_t) {
                        oprnd_t = structurally_resolved_type(fcx, oprnd.span,
                                                             oprnd_t);
                        if !(ty::type_is_integral(oprnd_t) ||
                             ty::type_is_fp(oprnd_t)) {
                            oprnd_t = check_user_unop(fcx, "-", "neg",
                                                      tcx.lang_items.neg_trait(),
                                                      expr, &**oprnd, oprnd_t);
                        }
                    }
                }
            }
        }
        fcx.write_ty(id, oprnd_t);
      }
      ast::ExprAddrOf(mutbl, ref oprnd) => {
        let expected = expected.only_has_type();
        let hint = expected.map(fcx, |sty| {
            match *sty { ty::ty_rptr(_, ref mt) | ty::ty_ptr(ref mt) => ExpectHasType(mt.ty),
                         _ => NoExpectation }
        });
        let lvalue_pref = match mutbl {
            ast::MutMutable => PreferMutLvalue,
            ast::MutImmutable => NoPreference
        };
        check_expr_with_expectation_and_lvalue_pref(fcx,
                                                    &**oprnd,
                                                    hint,
                                                    lvalue_pref);

        let tm = ty::mt { ty: fcx.expr_ty(&**oprnd), mutbl: mutbl };
        let oprnd_t = if ty::type_is_error(tm.ty) {
            ty::mk_err()
        } else if ty::type_is_bot(tm.ty) {
            ty::mk_bot()
        }
        else {
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
            match oprnd.node {
                // String literals are already, implicitly converted to slices.
                //ast::ExprLit(lit) if ast_util::lit_is_str(lit) => fcx.expr_ty(oprnd),
                // Empty slices live in static memory.
                ast::ExprVec(ref elements) if elements.len() == 0 => {
                    // Note: we do not assign a lifetime of
                    // static. This is because the resulting type
                    // `&'static [T]` would require that T outlives
                    // `'static`!
                    let region = fcx.infcx().next_region_var(
                        infer::AddrOfSlice(expr.span));
                    ty::mk_rptr(tcx, region, tm)
                }
                _ => {
                    let region = fcx.infcx().next_region_var(infer::AddrOfRegion(expr.span));
                    ty::mk_rptr(tcx, region, tm)
                }
            }
        };
        fcx.write_ty(id, oprnd_t);
      }
      ast::ExprPath(ref pth) => {
          let defn = lookup_def(fcx, pth.span, id);
          let pty = polytype_for_def(fcx, expr.span, defn);
          instantiate_path(fcx, pth, pty, defn, expr.span, expr.id);

          // We always require that the type provided as the value for
          // a type parameter outlives the moment of instantiation.
          constrain_path_type_parameters(fcx, expr);
      }
      ast::ExprInlineAsm(ref ia) => {
          for &(_, ref input) in ia.inputs.iter() {
              check_expr(fcx, &**input);
          }
          for &(_, ref out, _) in ia.outputs.iter() {
              check_expr(fcx, &**out);
          }
          fcx.write_nil(id);
      }
      ast::ExprMac(_) => tcx.sess.bug("unexpanded macro"),
      ast::ExprBreak(_) => { fcx.write_bot(id); }
      ast::ExprAgain(_) => { fcx.write_bot(id); }
      ast::ExprRet(ref expr_opt) => {
        let ret_ty = fcx.ret_ty;
        match *expr_opt {
          None => match fcx.mk_eqty(false, infer::Misc(expr.span),
                                    ret_ty, ty::mk_nil()) {
            Ok(_) => { /* fall through */ }
            Err(_) => {
                span_err!(tcx.sess, expr.span, E0069,
                    "`return;` in function returning non-nil");
            }
          },
          Some(ref e) => {
              check_expr_coercable_to_type(fcx, &**e, ret_ty);
          }
        }
        fcx.write_bot(id);
      }
      ast::ExprParen(ref a) => {
        check_expr_with_expectation_and_lvalue_pref(fcx,
                                                    &**a,
                                                    expected,
                                                    lvalue_pref);
        fcx.write_ty(id, fcx.expr_ty(&**a));
      }
      ast::ExprAssign(ref lhs, ref rhs) => {
        check_expr_with_lvalue_pref(fcx, &**lhs, PreferMutLvalue);

        let tcx = fcx.tcx();
        if !ty::expr_is_lval(tcx, &**lhs) {
            span_err!(tcx.sess, expr.span, E0070,
                "illegal left-hand side expression");
        }

        let lhs_ty = fcx.expr_ty(&**lhs);
        check_expr_has_type(fcx, &**rhs, lhs_ty);
        let rhs_ty = fcx.expr_ty(&**rhs);

        fcx.require_expr_have_sized_type(&**lhs, traits::AssignmentLhsSized);

        if ty::type_is_error(lhs_ty) || ty::type_is_error(rhs_ty) {
            fcx.write_error(id);
        } else if ty::type_is_bot(lhs_ty) || ty::type_is_bot(rhs_ty) {
            fcx.write_bot(id);
        } else {
            fcx.write_nil(id);
        }
      }
      ast::ExprIf(ref cond, ref then_blk, ref opt_else_expr) => {
        check_then_else(fcx, &**cond, &**then_blk, opt_else_expr.as_ref().map(|e| &**e),
                        id, expr.span, expected);
      }
      ast::ExprIfLet(..) => {
        tcx.sess.span_bug(expr.span, "non-desugared ExprIfLet");
      }
      ast::ExprWhile(ref cond, ref body, _) => {
        check_expr_has_type(fcx, &**cond, ty::mk_bool());
        check_block_no_value(fcx, &**body);
        let cond_ty = fcx.expr_ty(&**cond);
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
      ast::ExprForLoop(ref pat, ref head, ref block, _) => {
        check_expr(fcx, &**head);
        let typ = lookup_method_for_for_loop(fcx, &**head, expr.id);
        vtable2::select_fcx_obligations_where_possible(fcx);

        let pcx = pat_ctxt {
            fcx: fcx,
            map: pat_id_map(&tcx.def_map, &**pat),
        };
        _match::check_pat(&pcx, &**pat, typ);

        check_block_no_value(fcx, &**block);
        fcx.write_nil(id);
      }
      ast::ExprLoop(ref body, _) => {
        check_block_no_value(fcx, &**body);
        if !may_break(tcx, expr.id, &**body) {
            fcx.write_bot(id);
        } else {
            fcx.write_nil(id);
        }
      }
      ast::ExprMatch(ref discrim, ref arms, _) => {
        _match::check_match(fcx, expr, &**discrim, arms.as_slice());
      }
      ast::ExprFnBlock(_, ref decl, ref body) => {
        let region = astconv::opt_ast_region_to_region(fcx,
                                                       fcx.infcx(),
                                                       expr.span,
                                                       &None);
        check_expr_fn(fcx,
                      expr,
                      ty::RegionTraitStore(region, ast::MutMutable),
                      &**decl,
                      &**body,
                      expected);
      }
      ast::ExprUnboxedFn(_, kind, ref decl, ref body) => {
        check_unboxed_closure(fcx,
                              expr,
                              kind,
                              &**decl,
                              &**body);
      }
      ast::ExprProc(ref decl, ref body) => {
        check_expr_fn(fcx,
                      expr,
                      ty::UniqTraitStore,
                      &**decl,
                      &**body,
                      expected);
      }
      ast::ExprBlock(ref b) => {
        check_block_with_expected(fcx, &**b, expected);
        fcx.write_ty(id, fcx.node_ty(b.id));
      }
      ast::ExprCall(ref f, ref args) => {
          // Index expressions need to be handled separately, to inform them
          // that they appear in call position.
          check_expr(fcx, &**f);
          let f_ty = fcx.expr_ty(&**f);

          let args: Vec<_> = args.iter().map(|x| x).collect();
          if !try_overloaded_call(fcx, expr, &**f, f_ty, args.as_slice()) {
              check_call(fcx, expr, &**f, args.as_slice());
              let (args_bot, args_err) = args.iter().fold((false, false),
                 |(rest_bot, rest_err), a| {
                     // is this not working?
                     let a_ty = fcx.expr_ty(&***a);
                     (rest_bot || ty::type_is_bot(a_ty),
                      rest_err || ty::type_is_error(a_ty))});
              if ty::type_is_error(f_ty) || args_err {
                  fcx.write_error(id);
              }
              else if ty::type_is_bot(f_ty) || args_bot {
                  fcx.write_bot(id);
              }
          }
      }
      ast::ExprMethodCall(ident, ref tps, ref args) => {
        check_method_call(fcx, expr, ident, args.as_slice(), tps.as_slice());
        let mut arg_tys = args.iter().map(|a| fcx.expr_ty(&**a));
        let (args_bot, args_err) = arg_tys.fold((false, false),
             |(rest_bot, rest_err), a| {
              (rest_bot || ty::type_is_bot(a),
               rest_err || ty::type_is_error(a))});
        if args_err {
            fcx.write_error(id);
        } else if args_bot {
            fcx.write_bot(id);
        }
      }
      ast::ExprCast(ref e, ref t) => {
        match t.node {
            ast::TyFixedLengthVec(_, ref count_expr) => {
                check_expr_with_hint(fcx, &**count_expr, ty::mk_uint());
            }
            _ => {}
        }
        check_cast(fcx, expr, &**e, &**t);
      }
      ast::ExprVec(ref args) => {
        let uty = match expected {
            ExpectHasType(uty) => {
                match ty::get(uty).sty {
                        ty::ty_vec(ty, _) => Some(ty),
                        _ => None
                }
            }
            _ => None
        };

        let typ = match uty {
            Some(uty) => {
                for e in args.iter() {
                    check_expr_coercable_to_type(fcx, &**e, uty);
                }
                uty
            }
            None => {
                let t: ty::t = fcx.infcx().next_ty_var();
                for e in args.iter() {
                    check_expr_has_type(fcx, &**e, t);
                }
                t
            }
        };
        let typ = ty::mk_vec(tcx, typ, Some(args.len()));
        fcx.write_ty(id, typ);
      }
      ast::ExprRepeat(ref element, ref count_expr) => {
        check_expr_has_type(fcx, &**count_expr, ty::mk_uint());
        let count = ty::eval_repeat_count(fcx.tcx(), &**count_expr);

        let uty = match expected {
            ExpectHasType(uty) => {
                match ty::get(uty).sty {
                        ty::ty_vec(ty, _) => Some(ty),
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
                let t: ty::t = fcx.infcx().next_ty_var();
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

        if ty::type_is_error(element_ty) {
            fcx.write_error(id);
        } else if ty::type_is_bot(element_ty) {
            fcx.write_bot(id);
        } else {
            let t = ty::mk_vec(tcx, t, Some(count));
            fcx.write_ty(id, t);
        }
      }
      ast::ExprTup(ref elts) => {
        let expected = expected.only_has_type();
        let flds = expected.map_to_option(fcx, |sty| {
            match *sty {
                ty::ty_tup(ref flds) => Some((*flds).clone()),
                _ => None
            }
        });
        let mut bot_field = false;
        let mut err_field = false;

        let elt_ts = elts.iter().enumerate().map(|(i, e)| {
            let t = match flds {
                Some(ref fs) if i < fs.len() => {
                    let ety = *fs.get(i);
                    check_expr_coercable_to_type(fcx, &**e, ety);
                    ety
                }
                _ => {
                    check_expr_with_expectation(fcx, &**e, NoExpectation);
                    fcx.expr_ty(&**e)
                }
            };
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
      ast::ExprStruct(ref path, ref fields, ref base_expr) => {
        // Resolve the path.
        let def = tcx.def_map.borrow().find(&id).map(|i| *i);
        let struct_id = match def {
            Some(def::DefVariant(enum_id, variant_id, true)) => {
                check_struct_enum_variant(fcx, id, expr.span, enum_id,
                                          variant_id, fields.as_slice());
                enum_id
            }
            Some(def::DefTrait(def_id)) => {
                span_err!(tcx.sess, path.span, E0159,
                    "use of trait `{}` as a struct constructor",
                    pprust::path_to_string(path));
                check_struct_fields_on_error(fcx,
                                             id,
                                             fields.as_slice(),
                                             base_expr);
                def_id
            },
            Some(def) => {
                // Verify that this was actually a struct.
                let typ = ty::lookup_item_type(fcx.ccx.tcx, def.def_id());
                match ty::get(typ.ty).sty {
                    ty::ty_struct(struct_did, _) => {
                        check_struct_constructor(fcx,
                                                 id,
                                                 expr.span,
                                                 struct_did,
                                                 fields.as_slice(),
                                                 base_expr.as_ref().map(|e| &**e));
                    }
                    _ => {
                        span_err!(tcx.sess, path.span, E0071,
                            "`{}` does not name a structure",
                            pprust::path_to_string(path));
                        check_struct_fields_on_error(fcx,
                                                     id,
                                                     fields.as_slice(),
                                                     base_expr);
                    }
                }

                def.def_id()
            }
            _ => {
                tcx.sess.span_bug(path.span,
                                  "structure constructor wasn't resolved")
            }
        };

        // Turn the path into a type and verify that that type unifies with
        // the resulting structure type. This is needed to handle type
        // parameters correctly.
        let actual_structure_type = fcx.expr_ty(&*expr);
        if !ty::type_is_error(actual_structure_type) {
            let type_and_substs = astconv::ast_path_to_ty_relaxed(fcx,
                                                                  fcx.infcx(),
                                                                  struct_id,
                                                                  path);
            match fcx.mk_subty(false,
                               infer::Misc(path.span),
                               actual_structure_type,
                               type_and_substs.ty) {
                Ok(()) => {}
                Err(type_error) => {
                    let type_error_description =
                        ty::type_err_to_str(tcx, &type_error);
                    fcx.tcx()
                       .sess
                       .span_err(path.span,
                                 format!("structure constructor specifies a \
                                         structure of type `{}`, but this \
                                         structure has type `{}`: {}",
                                         fcx.infcx()
                                            .ty_to_string(type_and_substs.ty),
                                         fcx.infcx()
                                            .ty_to_string(
                                                actual_structure_type),
                                         type_error_description).as_slice());
                    ty::note_and_explain_type_err(tcx, &type_error);
                }
            }
        }

        fcx.require_expr_have_sized_type(expr, traits::StructInitializerSized);
      }
      ast::ExprField(ref base, ref field, ref tys) => {
        check_field(fcx, expr, lvalue_pref, &**base, field, tys.as_slice());
      }
      ast::ExprTupField(ref base, idx, ref tys) => {
        check_tup_field(fcx, expr, lvalue_pref, &**base, idx, tys.as_slice());
      }
      ast::ExprIndex(ref base, ref idx) => {
          check_expr_with_lvalue_pref(fcx, &**base, lvalue_pref);
          check_expr(fcx, &**idx);
          let raw_base_t = fcx.expr_ty(&**base);
          let idx_t = fcx.expr_ty(&**idx);
          if ty::type_is_error(raw_base_t) {
              fcx.write_ty(id, raw_base_t);
          } else if ty::type_is_error(idx_t) {
              fcx.write_ty(id, idx_t);
          } else {
              let (_, autoderefs, field_ty) =
                autoderef(fcx, expr.span, raw_base_t, Some(base.id),
                          lvalue_pref, |base_t, _| ty::index(base_t));
              match field_ty {
                  Some(ty) if !ty::type_is_bot(ty) => {
                      check_expr_has_type(fcx, &**idx, ty::mk_uint());
                      fcx.write_ty(id, ty);
                      fcx.write_autoderef_adjustment(base.id, base.span, autoderefs);
                  }
                  _ => {
                      // This is an overloaded method.
                      let base_t = structurally_resolved_type(fcx,
                                                              expr.span,
                                                              raw_base_t);
                      let method_call = MethodCall::expr(expr.id);
                      match try_overloaded_index(fcx,
                                                 Some(method_call),
                                                 expr,
                                                 &**base,
                                                 base_t,
                                                 idx,
                                                 lvalue_pref) {
                          Some(mt) => fcx.write_ty(id, mt.ty),
                          None => {
                                fcx.type_error_message(expr.span,
                                                       |actual| {
                                                        format!("cannot \
                                                                 index a \
                                                                 value of \
                                                                 type `{}`",
                                                                actual)
                                                       },
                                                       base_t,
                                                       None);
                                fcx.write_ty(id, ty::mk_err())
                          }
                      }
                  }
              }
          }
       }
       ast::ExprSlice(ref base, ref start, ref end, ref mutbl) => {
          check_expr_with_lvalue_pref(fcx, &**base, lvalue_pref);
          let raw_base_t = fcx.expr_ty(&**base);

          let mut some_err = false;
          if ty::type_is_error(raw_base_t) || ty::type_is_bot(raw_base_t) {
              fcx.write_ty(id, raw_base_t);
              some_err = true;
          }

          {
              let check_slice_idx = |e: &ast::Expr| {
                  check_expr(fcx, e);
                  let e_t = fcx.expr_ty(e);
                  if ty::type_is_error(e_t) || ty::type_is_bot(e_t) {
                    fcx.write_ty(id, e_t);
                    some_err = true;
                  }
              };
              start.as_ref().map(|e| check_slice_idx(&**e));
              end.as_ref().map(|e| check_slice_idx(&**e));
          }

          if !some_err {
              let base_t = structurally_resolved_type(fcx,
                                                      expr.span,
                                                      raw_base_t);
              let method_call = MethodCall::expr(expr.id);
              match try_overloaded_slice(fcx,
                                         Some(method_call),
                                         expr,
                                         &**base,
                                         base_t,
                                         start,
                                         end,
                                         mutbl) {
                  Some(mt) => fcx.write_ty(id, mt.ty),
                  None => {
                        fcx.type_error_message(expr.span,
                           |actual| {
                                format!("cannot take a {}slice of a value with type `{}`",
                                        if mutbl == &ast::MutMutable {
                                            "mutable "
                                        } else {
                                            ""
                                        },
                                        actual)
                           },
                           base_t,
                           None);
                        fcx.write_ty(id, ty::mk_err())
                  }
              }
          }
       }
    }

    debug!("type of expr({}) {} is...", expr.id,
           syntax::print::pprust::expr_to_string(expr));
    debug!("... {}, expected is {}",
           ppaux::ty_to_string(tcx, fcx.expr_ty(expr)),
           expected.repr(tcx));

    unifier();
}

fn constrain_path_type_parameters(fcx: &FnCtxt,
                                  expr: &ast::Expr)
{
    fcx.opt_node_ty_substs(expr.id, |item_substs| {
        for &ty in item_substs.substs.types.iter() {
            let default_bound = ty::ReScope(expr.id);
            let origin = infer::RelateDefaultParamBound(expr.span, ty);
            fcx.register_region_obligation(origin, ty, default_bound);
        }
    });
}

impl Expectation {
    fn only_has_type(self) -> Expectation {
        match self {
            NoExpectation | ExpectCastableToType(..) => NoExpectation,
            ExpectHasType(t) => ExpectHasType(t)
        }
    }

    // Resolves `expected` by a single level if it is a variable. If
    // there is no expected type or resolution is not possible (e.g.,
    // no constraints yet present), just returns `None`.
    fn resolve(self, fcx: &FnCtxt) -> Expectation {
        match self {
            NoExpectation => {
                NoExpectation
            }
            ExpectCastableToType(t) => {
                ExpectCastableToType(
                    fcx.infcx().resolve_type_vars_if_possible(t))
            }
            ExpectHasType(t) => {
                ExpectHasType(
                    fcx.infcx().resolve_type_vars_if_possible(t))
            }
        }
    }

    fn map(self, fcx: &FnCtxt, unpack: |&ty::sty| -> Expectation) -> Expectation {
        match self.resolve(fcx) {
            NoExpectation => NoExpectation,
            ExpectCastableToType(t) | ExpectHasType(t) => unpack(&ty::get(t).sty),
        }
    }

    fn map_to_option<O>(self,
                        fcx: &FnCtxt,
                        unpack: |&ty::sty| -> Option<O>)
                        -> Option<O>
    {
        match self.resolve(fcx) {
            NoExpectation => None,
            ExpectCastableToType(t) | ExpectHasType(t) => unpack(&ty::get(t).sty),
        }
    }
}

impl Repr for Expectation {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            NoExpectation => format!("NoExpectation"),
            ExpectHasType(t) => format!("ExpectHasType({})",
                                        t.repr(tcx)),
            ExpectCastableToType(t) => format!("ExpectCastableToType({})",
                                               t.repr(tcx)),
        }
    }
}

pub fn check_decl_initializer(fcx: &FnCtxt,
                              nid: ast::NodeId,
                              init: &ast::Expr)
                            {
    let local_ty = fcx.local_ty(init.span, nid);
    check_expr_coercable_to_type(fcx, init, local_ty)
}

pub fn check_decl_local(fcx: &FnCtxt, local: &ast::Local)  {
    let tcx = fcx.ccx.tcx;

    let t = fcx.local_ty(local.span, local.id);
    fcx.write_ty(local.id, t);

    match local.init {
        Some(ref init) => {
            check_decl_initializer(fcx, local.id, &**init);
            let init_ty = fcx.expr_ty(&**init);
            if ty::type_is_error(init_ty) || ty::type_is_bot(init_ty) {
                fcx.write_ty(local.id, init_ty);
            }
        }
        _ => {}
    }

    let pcx = pat_ctxt {
        fcx: fcx,
        map: pat_id_map(&tcx.def_map, &*local.pat),
    };
    _match::check_pat(&pcx, &*local.pat, t);
    let pat_ty = fcx.node_ty(local.pat.id);
    if ty::type_is_error(pat_ty) || ty::type_is_bot(pat_ty) {
        fcx.write_ty(local.id, pat_ty);
    }
}

pub fn check_stmt(fcx: &FnCtxt, stmt: &ast::Stmt)  {
    let node_id;
    let mut saw_bot = false;
    let mut saw_err = false;
    match stmt.node {
      ast::StmtDecl(ref decl, id) => {
        node_id = id;
        match decl.node {
          ast::DeclLocal(ref l) => {
              check_decl_local(fcx, &**l);
              let l_t = fcx.node_ty(l.id);
              saw_bot = saw_bot || ty::type_is_bot(l_t);
              saw_err = saw_err || ty::type_is_error(l_t);
          }
          ast::DeclItem(_) => {/* ignore for now */ }
        }
      }
      ast::StmtExpr(ref expr, id) => {
        node_id = id;
        // Check with expected type of ()
        check_expr_has_type(fcx, &**expr, ty::mk_nil());
        let expr_ty = fcx.expr_ty(&**expr);
        saw_bot = saw_bot || ty::type_is_bot(expr_ty);
        saw_err = saw_err || ty::type_is_error(expr_ty);
      }
      ast::StmtSemi(ref expr, id) => {
        node_id = id;
        check_expr(fcx, &**expr);
        let expr_ty = fcx.expr_ty(&**expr);
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

pub fn check_block_no_value(fcx: &FnCtxt, blk: &ast::Block)  {
    check_block_with_expected(fcx, blk, ExpectHasType(ty::mk_nil()));
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

fn check_block_with_expected(fcx: &FnCtxt,
                             blk: &ast::Block,
                             expected: Expectation) {
    let prev = {
        let mut fcx_ps = fcx.ps.borrow_mut();
        let fn_style_state = fcx_ps.recurse(blk);
        replace(&mut *fcx_ps, fn_style_state)
    };

    let mut warned = false;
    let mut last_was_bot = false;
    let mut any_bot = false;
    let mut any_err = false;
    for s in blk.stmts.iter() {
        check_stmt(fcx, &**s);
        let s_id = ast_util::stmt_id(&**s);
        let s_ty = fcx.node_ty(s_id);
        if last_was_bot && !warned && match s.node {
            ast::StmtDecl(ref decl, _) => {
                match decl.node {
                    ast::DeclLocal(_) => true,
                    _ => false,
                }
            }
            ast::StmtExpr(_, _) | ast::StmtSemi(_, _) => true,
            _ => false
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
        if ty::type_is_bot(s_ty) {
            last_was_bot = true;
        }
        any_bot = any_bot || ty::type_is_bot(s_ty);
        any_err = any_err || ty::type_is_error(s_ty);
    }
    match blk.expr {
        None => if any_err {
            fcx.write_error(blk.id);
        } else if any_bot {
            fcx.write_bot(blk.id);
        } else {
            fcx.write_nil(blk.id);
        },
        Some(ref e) => {
            if any_bot && !warned {
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

            fcx.write_ty(blk.id, ety);
            if any_err {
                fcx.write_error(blk.id);
            } else if any_bot {
                fcx.write_bot(blk.id);
            }
        }
    };

    *fcx.ps.borrow_mut() = prev;
}

/// Checks a constant appearing in a type. At the moment this is just the
/// length expression in a fixed-length vector, but someday it might be
/// extended to type-level numeric literals.
pub fn check_const_in_type(tcx: &ty::ctxt,
                           expr: &ast::Expr,
                           expected_type: ty::t) {
    // Synthesize a crate context. The trait map is not needed here (though I
    // imagine it will be if we have associated statics --pcwalton), so we
    // leave it blank.
    let ccx = CrateCtxt {
        trait_map: NodeMap::new(),
        tcx: tcx,
    };
    let inh = static_inherited_fields(&ccx);
    let fcx = blank_fn_ctxt(&ccx, &inh, expected_type, expr.id);
    check_const_with_ty(&fcx, expr.span, expr, expected_type);
}

pub fn check_const(ccx: &CrateCtxt,
                   sp: Span,
                   e: &ast::Expr,
                   id: ast::NodeId) {
    let inh = static_inherited_fields(ccx);
    let rty = ty::node_id_to_type(ccx.tcx, id);
    let fcx = blank_fn_ctxt(ccx, &inh, rty, e.id);
    let declty = fcx.ccx.tcx.tcache.borrow().get(&local_def(id)).ty;
    check_const_with_ty(&fcx, sp, e, declty);
}

pub fn check_const_with_ty(fcx: &FnCtxt,
                           _: Span,
                           e: &ast::Expr,
                           declty: ty::t) {
    // Gather locals in statics (because of block expressions).
    // This is technically unnecessary because locals in static items are forbidden,
    // but prevents type checking from blowing up before const checking can properly
    // emit a error.
    GatherLocalsVisitor { fcx: fcx }.visit_expr(e);

    check_expr_with_hint(fcx, e, declty);
    demand::coerce(fcx, e.span, declty, e);
    vtable2::select_all_fcx_obligations_or_error(fcx);
    regionck::regionck_expr(fcx, e);
    writeback::resolve_type_vars_in_expr(fcx, e);
}

/// Checks whether a type can be represented in memory. In particular, it
/// identifies types that contain themselves without indirection through a
/// pointer, which would mean their size is unbounded. This is different from
/// the question of whether a type can be instantiated. See the definition of
/// `check_instantiable`.
pub fn check_representable(tcx: &ty::ctxt,
                           sp: Span,
                           item_id: ast::NodeId,
                           designation: &str) -> bool {
    let rty = ty::node_id_to_type(tcx, item_id);

    // Check that it is possible to represent this type. This call identifies
    // (1) types that contain themselves and (2) types that contain a different
    // recursive type. It is only necessary to throw an error on those that
    // contain themselves. For case 2, there must be an inner type that will be
    // caught by case 1.
    match ty::is_type_representable(tcx, sp, rty) {
      ty::SelfRecursive => {
        span_err!(tcx.sess, sp, E0072,
            "illegal recursive {} type; \
             wrap the inner value in a box to make it representable",
            designation);
        return false
      }
      ty::Representable | ty::ContainsRecursive => (),
    }
    return true
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
pub fn check_instantiable(tcx: &ty::ctxt,
                          sp: Span,
                          item_id: ast::NodeId)
                          -> bool {
    let item_ty = ty::node_id_to_type(tcx, item_id);
    if !ty::is_instantiable(tcx, item_ty) {
        span_err!(tcx.sess, sp, E0073,
            "this type cannot be instantiated without an \
             instance of itself; consider using `Option<{}>`",
            ppaux::ty_to_string(tcx, item_ty));
        false
    } else {
        true
    }
}

pub fn check_simd(tcx: &ty::ctxt, sp: Span, id: ast::NodeId) {
    let t = ty::node_id_to_type(tcx, id);
    if ty::type_needs_subst(t) {
        span_err!(tcx.sess, sp, E0074, "SIMD vector cannot be generic");
        return;
    }
    match ty::get(t).sty {
        ty::ty_struct(did, ref substs) => {
            let fields = ty::lookup_struct_fields(tcx, did);
            if fields.is_empty() {
                span_err!(tcx.sess, sp, E0075, "SIMD vector cannot be empty");
                return;
            }
            let e = ty::lookup_field_type(tcx, did, fields.get(0).id, substs);
            if !fields.iter().all(
                         |f| ty::lookup_field_type(tcx, did, f.id, substs) == e) {
                span_err!(tcx.sess, sp, E0076, "SIMD vector should be homogeneous");
                return;
            }
            if !ty::type_is_machine(e) {
                span_err!(tcx.sess, sp, E0077,
                    "SIMD vector element type should be machine type");
                return;
            }
        }
        _ => ()
    }
}

pub fn check_enum_variants(ccx: &CrateCtxt,
                           sp: Span,
                           vs: &[P<ast::Variant>],
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
                ast::TyU => uint_in_range(ccx, ccx.tcx.sess.targ_cfg.uint_type, disr)
            }
        }
        fn int_in_range(ccx: &CrateCtxt, ty: ast::IntTy, disr: ty::Disr) -> bool {
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

    fn do_check(ccx: &CrateCtxt,
                vs: &[P<ast::Variant>],
                id: ast::NodeId,
                hint: attr::ReprAttr)
                -> Vec<Rc<ty::VariantInfo>> {

        let rty = ty::node_id_to_type(ccx.tcx, id);
        let mut variants: Vec<Rc<ty::VariantInfo>> = Vec::new();
        let mut disr_vals: Vec<ty::Disr> = Vec::new();
        let mut prev_disr_val: Option<ty::Disr> = None;

        for v in vs.iter() {

            // If the discriminant value is specified explicitly in the enum check whether the
            // initialization expression is valid, otherwise use the last value plus one.
            let mut current_disr_val = match prev_disr_val {
                Some(prev_disr_val) => prev_disr_val + 1,
                None => ty::INITIAL_DISCRIMINANT_VALUE
            };

            match v.node.disr_expr {
                Some(ref e) => {
                    debug!("disr expr, checking {}", pprust::expr_to_string(&**e));

                    let inh = static_inherited_fields(ccx);
                    let fcx = blank_fn_ctxt(ccx, &inh, rty, e.id);
                    let declty = match hint {
                        attr::ReprAny | attr::ReprPacked | attr::ReprExtern => ty::mk_int(),
                        attr::ReprInt(_, attr::SignedInt(ity)) => {
                            ty::mk_mach_int(ity)
                        }
                        attr::ReprInt(_, attr::UnsignedInt(ity)) => {
                            ty::mk_mach_uint(ity)
                        },
                    };
                    check_const_with_ty(&fcx, e.span, &**e, declty);
                    // check_expr (from check_const pass) doesn't guarantee
                    // that the expression is in a form that eval_const_expr can
                    // handle, so we may still get an internal compiler error

                    match const_eval::eval_const_expr_partial(ccx.tcx, &**e) {
                        Ok(const_eval::const_int(val)) => current_disr_val = val as Disr,
                        Ok(const_eval::const_uint(val)) => current_disr_val = val as Disr,
                        Ok(_) => {
                            span_err!(ccx.tcx.sess, e.span, E0079,
                                "expected signed integer constant");
                        }
                        Err(ref err) => {
                            span_err!(ccx.tcx.sess, e.span, E0080,
                                "expected constant: {}", *err);
                        }
                    }
                },
                None => ()
            };

            // Check for duplicate discriminant values
            match disr_vals.iter().position(|&x| x == current_disr_val) {
                Some(i) => {
                    span_err!(ccx.tcx.sess, v.span, E0081,
                        "discriminant value `{}` already exists", disr_vals[i]);
                    span_note!(ccx.tcx.sess, ccx.tcx().map.span(variants[i].id.node),
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
                attr::ReprPacked => {
                    ccx.tcx.sess.bug("range_to_inttype: found ReprPacked on an enum");
                }
            }
            disr_vals.push(current_disr_val);

            let variant_info = Rc::new(VariantInfo::from_ast_variant(ccx.tcx, &**v,
                                                                     current_disr_val));
            prev_disr_val = Some(current_disr_val);

            variants.push(variant_info);
        }

        return variants;
    }

    let hint = *ty::lookup_repr_hints(ccx.tcx, ast::DefId { krate: ast::LOCAL_CRATE, node: id })
                    .as_slice().get(0).unwrap_or(&attr::ReprAny);

    if hint != attr::ReprAny && vs.len() <= 1 {
        if vs.len() == 1 {
            span_err!(ccx.tcx.sess, sp, E0083,
                "unsupported representation for univariant enum");
        } else {
            span_err!(ccx.tcx.sess, sp, E0084,
                "unsupported representation for zero-variant enum");
        };
    }

    let variants = do_check(ccx, vs, id, hint);

    // cache so that ty::enum_variants won't repeat this work
    ccx.tcx.enum_var_cache.borrow_mut().insert(local_def(id), Rc::new(variants));

    check_representable(ccx.tcx, sp, id, "enum");

    // Check that it is possible to instantiate this enum:
    //
    // This *sounds* like the same that as representable, but it's
    // not.  See def'n of `check_instantiable()` for details.
    check_instantiable(ccx.tcx, sp, id);
}

pub fn lookup_def(fcx: &FnCtxt, sp: Span, id: ast::NodeId) -> def::Def {
    lookup_def_ccx(fcx.ccx, sp, id)
}

// Returns the type parameter count and the type for the given definition.
pub fn polytype_for_def(fcx: &FnCtxt,
                        sp: Span,
                        defn: def::Def)
                        -> Polytype {
    match defn {
      def::DefLocal(nid) | def::DefUpvar(nid, _, _) => {
          let typ = fcx.local_ty(sp, nid);
          return no_params(typ);
      }
      def::DefFn(id, _, _) | def::DefStaticMethod(id, _, _) |
      def::DefStatic(id, _) | def::DefVariant(_, id, _) |
      def::DefStruct(id) | def::DefConst(id) => {
        return ty::lookup_item_type(fcx.ccx.tcx, id);
      }
      def::DefTrait(_) |
      def::DefTy(..) |
      def::DefAssociatedTy(..) |
      def::DefPrimTy(_) |
      def::DefTyParam(..)=> {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value, found type");
      }
      def::DefMod(..) | def::DefForeignMod(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value, found module");
      }
      def::DefUse(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value, found use");
      }
      def::DefRegion(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value, found region");
      }
      def::DefTyParamBinder(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value, found type parameter");
      }
      def::DefLabel(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value, found label");
      }
      def::DefSelfTy(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value, found self ty");
      }
      def::DefMethod(..) => {
        fcx.ccx.tcx.sess.span_bug(sp, "expected value, found method");
      }
    }
}

// Instantiates the given path, which must refer to an item with the given
// number of type parameters and type.
pub fn instantiate_path(fcx: &FnCtxt,
                        path: &ast::Path,
                        polytype: Polytype,
                        def: def::Def,
                        span: Span,
                        node_id: ast::NodeId) {
    debug!("instantiate_path(path={}, def={}, node_id={}, polytype={})",
           path.repr(fcx.tcx()),
           def.repr(fcx.tcx()),
           node_id,
           polytype.repr(fcx.tcx()));

    // We need to extract the type parameters supplied by the user in
    // the path `path`. Due to the current setup, this is a bit of a
    // tricky-process; the problem is that resolve only tells us the
    // end-point of the path resolution, and not the intermediate steps.
    // Luckily, we can (at least for now) deduce the intermediate steps
    // just from the end-point.
    //
    // There are basically three cases to consider:
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
    // 1b. Reference to a enum variant or tuple-like struct:
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
    // The first step then is to categorize the segments appropriately.

    assert!(path.segments.len() >= 1);
    let mut segment_spaces;
    match def {
        // Case 1 and 1b. Reference to a *type* or *enum variant*.
        def::DefSelfTy(..) |
        def::DefStruct(..) |
        def::DefVariant(..) |
        def::DefTyParamBinder(..) |
        def::DefTy(..) |
        def::DefAssociatedTy(..) |
        def::DefTrait(..) |
        def::DefPrimTy(..) |
        def::DefTyParam(..) => {
            // Everything but the final segment should have no
            // parameters at all.
            segment_spaces = Vec::from_elem(path.segments.len() - 1, None);
            segment_spaces.push(Some(subst::TypeSpace));
        }

        // Case 2. Reference to a top-level value.
        def::DefFn(..) |
        def::DefConst(..) |
        def::DefStatic(..) => {
            segment_spaces = Vec::from_elem(path.segments.len() - 1, None);
            segment_spaces.push(Some(subst::FnSpace));
        }

        // Case 3. Reference to a method.
        def::DefStaticMethod(..) => {
            assert!(path.segments.len() >= 2);
            segment_spaces = Vec::from_elem(path.segments.len() - 2, None);
            segment_spaces.push(Some(subst::TypeSpace));
            segment_spaces.push(Some(subst::FnSpace));
        }

        // Other cases. Various nonsense that really shouldn't show up
        // here. If they do, an error will have been reported
        // elsewhere. (I hope)
        def::DefMod(..) |
        def::DefForeignMod(..) |
        def::DefLocal(..) |
        def::DefMethod(..) |
        def::DefUse(..) |
        def::DefRegion(..) |
        def::DefLabel(..) |
        def::DefUpvar(..) => {
            segment_spaces = Vec::from_elem(path.segments.len(), None);
        }
    }
    assert_eq!(segment_spaces.len(), path.segments.len());

    debug!("segment_spaces={}", segment_spaces);

    // Next, examine the definition, and determine how many type
    // parameters we expect from each space.
    let type_defs = &polytype.generics.types;
    let region_defs = &polytype.generics.regions;

    // Now that we have categorized what space the parameters for each
    // segment belong to, let's sort out the parameters that the user
    // provided (if any) into their appropriate spaces. We'll also report
    // errors if type parameters are provided in an inappropriate place.
    let mut substs = Substs::empty();
    for (opt_space, segment) in segment_spaces.iter().zip(path.segments.iter()) {
        match *opt_space {
            None => {
                report_error_if_segment_contains_type_parameters(fcx, segment);
            }

            Some(space) => {
                push_explicit_parameters_from_segment_to_substs(fcx,
                                                                space,
                                                                type_defs,
                                                                region_defs,
                                                                segment,
                                                                &mut substs);
            }
        }
    }

    // Now we have to compare the types that the user *actually*
    // provided against the types that were *expected*. If the user
    // did not provide any types, then we want to substitute inference
    // variables. If the user provided some types, we may still need
    // to add defaults. If the user provided *too many* types, that's
    // a problem.
    for &space in ParamSpace::all().iter() {
        adjust_type_parameters(fcx, span, space, type_defs, &mut substs);
        assert_eq!(substs.types.len(space), type_defs.len(space));

        adjust_region_parameters(fcx, span, space, region_defs, &mut substs);
        assert_eq!(substs.regions().len(space), region_defs.len(space));
    }

    fcx.add_obligations_for_parameters(
        traits::ObligationCause::new(span,
                                     traits::ItemObligation(def.def_id())),
        &substs,
        &polytype.generics);

    fcx.write_ty_substs(node_id, polytype.ty, ty::ItemSubsts {
        substs: substs,
    });

    fn report_error_if_segment_contains_type_parameters(
        fcx: &FnCtxt,
        segment: &ast::PathSegment)
    {
        for typ in segment.types.iter() {
            span_err!(fcx.tcx().sess, typ.span, E0085,
                "type parameters may not appear here");
            break;
        }

        for lifetime in segment.lifetimes.iter() {
            span_err!(fcx.tcx().sess, lifetime.span, E0086,
                "lifetime parameters may not appear here");
            break;
        }
    }

    fn push_explicit_parameters_from_segment_to_substs(
        fcx: &FnCtxt,
        space: subst::ParamSpace,
        type_defs: &VecPerParamSpace<ty::TypeParameterDef>,
        region_defs: &VecPerParamSpace<ty::RegionParameterDef>,
        segment: &ast::PathSegment,
        substs: &mut Substs)
    {
        /*!
         * Finds the parameters that the user provided and adds them
         * to `substs`. If too many parameters are provided, then
         * reports an error and clears the output vector.
         *
         * We clear the output vector because that will cause the
         * `adjust_XXX_parameters()` later to use inference
         * variables. This seems less likely to lead to derived
         * errors.
         *
         * Note that we *do not* check for *too few* parameters here.
         * Due to the presence of defaults etc that is more
         * complicated. I wanted however to do the reporting of *too
         * many* parameters here because we can easily use the precise
         * span of the N+1'th parameter.
         */

        {
            let type_count = type_defs.len(space);
            assert_eq!(substs.types.len(space), 0);
            for (i, typ) in segment.types.iter().enumerate() {
                let t = fcx.to_ty(&**typ);
                if i < type_count {
                    substs.types.push(space, t);
                } else if i == type_count {
                    span_err!(fcx.tcx().sess, typ.span, E0087,
                        "too many type parameters provided: \
                         expected at most {} parameter(s), \
                         found {} parameter(s)",
                         type_count, segment.types.len());
                    substs.types.truncate(space, 0);
                }
            }
        }

        {
            let region_count = region_defs.len(space);
            assert_eq!(substs.regions().len(space), 0);
            for (i, lifetime) in segment.lifetimes.iter().enumerate() {
                let r = ast_region_to_region(fcx.tcx(), lifetime);
                if i < region_count {
                    substs.mut_regions().push(space, r);
                } else if i == region_count {
                    span_err!(fcx.tcx().sess, lifetime.span, E0088,
                        "too many lifetime parameters provided: \
                         expected {} parameter(s), found {} parameter(s)",
                        region_count,
                        segment.lifetimes.len());
                    substs.mut_regions().truncate(space, 0);
                }
            }
        }
    }

    fn adjust_type_parameters(
        fcx: &FnCtxt,
        span: Span,
        space: ParamSpace,
        defs: &VecPerParamSpace<ty::TypeParameterDef>,
        substs: &mut Substs)
    {
        let provided_len = substs.types.len(space);
        let desired = defs.get_slice(space);
        let required_len = desired.iter()
                              .take_while(|d| d.default.is_none())
                              .count();

        debug!("adjust_type_parameters(space={}, \
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
        if provided_len == 0 {
            substs.types.replace(space,
                                 fcx.infcx().next_ty_vars(desired.len()));
            return;
        }

        // Too few parameters specified: report an error and use Err
        // for everything.
        if provided_len < required_len {
            let qualifier =
                if desired.len() != required_len { "at least " } else { "" };
            span_err!(fcx.tcx().sess, span, E0089,
                "too few type parameters provided: expected {}{} parameter(s) \
                , found {} parameter(s)",
                qualifier, required_len, provided_len);
            substs.types.replace(space,
                                 Vec::from_elem(desired.len(), ty::mk_err()));
            return;
        }

        // Otherwise, add in any optional parameters that the user
        // omitted. The case of *too many* parameters is handled
        // already by
        // push_explicit_parameters_from_segment_to_substs(). Note
        // that the *default* type are expressed in terms of all prior
        // parameters, so we have to substitute as we go with the
        // partial substitution that we have built up.
        for i in range(provided_len, desired.len()) {
            let default = desired[i].default.unwrap();
            let default = default.subst_spanned(fcx.tcx(), substs, Some(span));
            substs.types.push(space, default);
        }
        assert_eq!(substs.types.len(space), desired.len());

        debug!("Final substs: {}", substs.repr(fcx.tcx()));
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
            "too few lifetime parameters provided: expected {} parameter(s), \
             found {} parameter(s)",
            desired.len(), provided_len);

        substs.mut_regions().replace(
            space,
            fcx.infcx().region_vars_for_defs(span, desired));
    }
}

// Resolves `typ` by a single level if `typ` is a type variable.  If no
// resolution is possible, then an error is reported.
pub fn structurally_resolved_type(fcx: &FnCtxt, sp: Span, tp: ty::t) -> ty::t {
    match infer::resolve_type(fcx.infcx(), Some(sp), tp, force_tvar) {
        Ok(t_s) if !ty::type_is_ty_var(t_s) => t_s,
        _ => {
            fcx.type_error_message(sp, |_actual| {
                "the type of this value must be known in this \
                 context".to_string()
            }, tp, None);
            demand::suptype(fcx, sp, ty::mk_err(), tp);
            tp
        }
    }
}

// Returns the one-level-deep structure of the given type.
pub fn structure_of<'a>(fcx: &FnCtxt, sp: Span, typ: ty::t)
                        -> &'a ty::sty {
    &ty::get(structurally_resolved_type(fcx, sp, typ)).sty
}

// Returns true if b contains a break that can exit from b
pub fn may_break(cx: &ty::ctxt, id: ast::NodeId, b: &ast::Block) -> bool {
    // First: is there an unlabeled break immediately
    // inside the loop?
    (loop_query(&*b, |e| {
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
                match cx.def_map.borrow().find(&e.id) {
                    Some(&def::DefLabel(loop_id)) if id == loop_id => true,
                    _ => false,
                }
            }
            _ => false
        }}))
}

pub fn check_bounds_are_used(ccx: &CrateCtxt,
                             span: Span,
                             tps: &OwnedSlice<ast::TyParam>,
                             ty: ty::t) {
    debug!("check_bounds_are_used(n_tps={}, ty={})",
           tps.len(), ppaux::ty_to_string(ccx.tcx, ty));

    // make a vector of booleans initially false, set to true when used
    if tps.len() == 0u { return; }
    let mut tps_used = Vec::from_elem(tps.len(), false);

    ty::walk_ty(ty, |t| {
            match ty::get(t).sty {
                ty::ty_param(ParamTy {idx, ..}) => {
                    debug!("Found use of ty param num {}", idx);
                    *tps_used.get_mut(idx) = true;
                }
                _ => ()
            }
        });

    for (i, b) in tps_used.iter().enumerate() {
        if !*b {
            span_err!(ccx.tcx.sess, span, E0091,
                "type parameter `{}` is unused",
                token::get_ident(tps.get(i).ident));
        }
    }
}

pub fn check_intrinsic_type(ccx: &CrateCtxt, it: &ast::ForeignItem) {
    fn param(ccx: &CrateCtxt, n: uint) -> ty::t {
        ty::mk_param(ccx.tcx, subst::FnSpace, n, local_def(0))
    }

    let tcx = ccx.tcx;
    let name = token::get_ident(it.ident);
    let (n_tps, inputs, output) = if name.get().starts_with("atomic_") {
        let split : Vec<&str> = name.get().split('_').collect();
        assert!(split.len() >= 2, "Atomic intrinsic not correct format");

        //We only care about the operation here
        match *split.get(1) {
            "cxchg" => (1, vec!(ty::mk_mut_ptr(tcx, param(ccx, 0)),
                                param(ccx, 0),
                                param(ccx, 0)),
                        param(ccx, 0)),
            "load" => (1, vec!(ty::mk_imm_ptr(tcx, param(ccx, 0))),
                       param(ccx, 0)),
            "store" => (1, vec!(ty::mk_mut_ptr(tcx, param(ccx, 0)), param(ccx, 0)),
                        ty::mk_nil()),

            "xchg" | "xadd" | "xsub" | "and"  | "nand" | "or" | "xor" | "max" |
            "min"  | "umax" | "umin" => {
                (1, vec!(ty::mk_mut_ptr(tcx, param(ccx, 0)), param(ccx, 0)),
                 param(ccx, 0))
            }
            "fence" => {
                (0, Vec::new(), ty::mk_nil())
            }
            op => {
                span_err!(tcx.sess, it.span, E0092,
                    "unrecognized atomic operation function: `{}`", op);
                return;
            }
        }

    } else {
        match name.get() {
            "abort" => (0, Vec::new(), ty::mk_bot()),
            "unreachable" => (0, Vec::new(), ty::mk_bot()),
            "breakpoint" => (0, Vec::new(), ty::mk_nil()),
            "size_of" |
            "pref_align_of" | "min_align_of" => (1u, Vec::new(), ty::mk_uint()),
            "init" => (1u, Vec::new(), param(ccx, 0u)),
            "uninit" => (1u, Vec::new(), param(ccx, 0u)),
            "forget" => (1u, vec!( param(ccx, 0) ), ty::mk_nil()),
            "transmute" => (2, vec!( param(ccx, 0) ), param(ccx, 1)),
            "move_val_init" => {
                (1u,
                 vec!(
                    ty::mk_mut_rptr(tcx, ty::ReLateBound(it.id, ty::BrAnon(0)), param(ccx, 0)),
                    param(ccx, 0u)
                  ),
               ty::mk_nil())
            }
            "needs_drop" => (1u, Vec::new(), ty::mk_bool()),
            "owns_managed" => (1u, Vec::new(), ty::mk_bool()),

            "get_tydesc" => {
              let tydesc_ty = match ty::get_tydesc_ty(ccx.tcx) {
                  Ok(t) => t,
                  Err(s) => { tcx.sess.span_fatal(it.span, s.as_slice()); }
              };
              let td_ptr = ty::mk_ptr(ccx.tcx, ty::mt {
                  ty: tydesc_ty,
                  mutbl: ast::MutImmutable
              });
              (1u, Vec::new(), td_ptr)
            }
            "type_id" => {
                let langid = ccx.tcx.lang_items.require(TypeIdLangItem);
                match langid {
                    Ok(did) => (1u,
                                Vec::new(),
                                ty::mk_struct(ccx.tcx, did,
                                              subst::Substs::empty())),
                    Err(msg) => {
                        tcx.sess.span_fatal(it.span, msg.as_slice());
                    }
                }
            },
            "visit_tydesc" => {
              let tydesc_ty = match ty::get_tydesc_ty(ccx.tcx) {
                  Ok(t) => t,
                  Err(s) => { tcx.sess.span_fatal(it.span, s.as_slice()); }
              };
              let region0 = ty::ReLateBound(it.id, ty::BrAnon(0));
              let region1 = ty::ReLateBound(it.id, ty::BrAnon(1));
              let visitor_object_ty =
                    match ty::visitor_object_ty(tcx, region0, region1) {
                        Ok((_, vot)) => vot,
                        Err(s) => { tcx.sess.span_fatal(it.span, s.as_slice()); }
                    };

              let td_ptr = ty::mk_ptr(ccx.tcx, ty::mt {
                  ty: tydesc_ty,
                  mutbl: ast::MutImmutable
              });
              (0, vec!( td_ptr, visitor_object_ty ), ty::mk_nil())
            }
            "offset" => {
              (1,
               vec!(
                  ty::mk_ptr(tcx, ty::mt {
                      ty: param(ccx, 0),
                      mutbl: ast::MutImmutable
                  }),
                  ty::mk_int()
               ),
               ty::mk_ptr(tcx, ty::mt {
                   ty: param(ccx, 0),
                   mutbl: ast::MutImmutable
               }))
            }
            "copy_memory" | "copy_nonoverlapping_memory" |
            "volatile_copy_memory" | "volatile_copy_nonoverlapping_memory" => {
              (1,
               vec!(
                  ty::mk_ptr(tcx, ty::mt {
                      ty: param(ccx, 0),
                      mutbl: ast::MutMutable
                  }),
                  ty::mk_ptr(tcx, ty::mt {
                      ty: param(ccx, 0),
                      mutbl: ast::MutImmutable
                  }),
                  ty::mk_uint()
               ),
               ty::mk_nil())
            }
            "set_memory" | "volatile_set_memory" => {
              (1,
               vec!(
                  ty::mk_ptr(tcx, ty::mt {
                      ty: param(ccx, 0),
                      mutbl: ast::MutMutable
                  }),
                  ty::mk_u8(),
                  ty::mk_uint()
               ),
               ty::mk_nil())
            }
            "sqrtf32" => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "sqrtf64" => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "powif32" => {
               (0,
                vec!( ty::mk_f32(), ty::mk_i32() ),
                ty::mk_f32())
            }
            "powif64" => {
               (0,
                vec!( ty::mk_f64(), ty::mk_i32() ),
                ty::mk_f64())
            }
            "sinf32" => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "sinf64" => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "cosf32" => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "cosf64" => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "powf32" => {
               (0,
                vec!( ty::mk_f32(), ty::mk_f32() ),
                ty::mk_f32())
            }
            "powf64" => {
               (0,
                vec!( ty::mk_f64(), ty::mk_f64() ),
                ty::mk_f64())
            }
            "expf32"   => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "expf64"   => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "exp2f32"  => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "exp2f64"  => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "logf32"   => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "logf64"   => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "log10f32" => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "log10f64" => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "log2f32"  => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "log2f64"  => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "fmaf32" => {
                (0,
                 vec!( ty::mk_f32(), ty::mk_f32(), ty::mk_f32() ),
                 ty::mk_f32())
            }
            "fmaf64" => {
                (0,
                 vec!( ty::mk_f64(), ty::mk_f64(), ty::mk_f64() ),
                 ty::mk_f64())
            }
            "fabsf32"      => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "fabsf64"      => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "copysignf32"  => (0, vec!( ty::mk_f32(), ty::mk_f32() ), ty::mk_f32()),
            "copysignf64"  => (0, vec!( ty::mk_f64(), ty::mk_f64() ), ty::mk_f64()),
            "floorf32"     => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "floorf64"     => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "ceilf32"      => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "ceilf64"      => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "truncf32"     => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "truncf64"     => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "rintf32"      => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "rintf64"      => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "nearbyintf32" => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "nearbyintf64" => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "roundf32"     => (0, vec!( ty::mk_f32() ), ty::mk_f32()),
            "roundf64"     => (0, vec!( ty::mk_f64() ), ty::mk_f64()),
            "ctpop8"       => (0, vec!( ty::mk_u8()  ), ty::mk_u8()),
            "ctpop16"      => (0, vec!( ty::mk_u16() ), ty::mk_u16()),
            "ctpop32"      => (0, vec!( ty::mk_u32() ), ty::mk_u32()),
            "ctpop64"      => (0, vec!( ty::mk_u64() ), ty::mk_u64()),
            "ctlz8"        => (0, vec!( ty::mk_u8()  ), ty::mk_u8()),
            "ctlz16"       => (0, vec!( ty::mk_u16() ), ty::mk_u16()),
            "ctlz32"       => (0, vec!( ty::mk_u32() ), ty::mk_u32()),
            "ctlz64"       => (0, vec!( ty::mk_u64() ), ty::mk_u64()),
            "cttz8"        => (0, vec!( ty::mk_u8()  ), ty::mk_u8()),
            "cttz16"       => (0, vec!( ty::mk_u16() ), ty::mk_u16()),
            "cttz32"       => (0, vec!( ty::mk_u32() ), ty::mk_u32()),
            "cttz64"       => (0, vec!( ty::mk_u64() ), ty::mk_u64()),
            "bswap16"      => (0, vec!( ty::mk_u16() ), ty::mk_u16()),
            "bswap32"      => (0, vec!( ty::mk_u32() ), ty::mk_u32()),
            "bswap64"      => (0, vec!( ty::mk_u64() ), ty::mk_u64()),

            "volatile_load" =>
                (1, vec!( ty::mk_imm_ptr(tcx, param(ccx, 0)) ), param(ccx, 0)),
            "volatile_store" =>
                (1, vec!( ty::mk_mut_ptr(tcx, param(ccx, 0)), param(ccx, 0) ), ty::mk_nil()),

            "i8_add_with_overflow" | "i8_sub_with_overflow" | "i8_mul_with_overflow" =>
                (0, vec!(ty::mk_i8(), ty::mk_i8()),
                ty::mk_tup(tcx, vec!(ty::mk_i8(), ty::mk_bool()))),

            "i16_add_with_overflow" | "i16_sub_with_overflow" | "i16_mul_with_overflow" =>
                (0, vec!(ty::mk_i16(), ty::mk_i16()),
                ty::mk_tup(tcx, vec!(ty::mk_i16(), ty::mk_bool()))),

            "i32_add_with_overflow" | "i32_sub_with_overflow" | "i32_mul_with_overflow" =>
                (0, vec!(ty::mk_i32(), ty::mk_i32()),
                ty::mk_tup(tcx, vec!(ty::mk_i32(), ty::mk_bool()))),

            "i64_add_with_overflow" | "i64_sub_with_overflow" | "i64_mul_with_overflow" =>
                (0, vec!(ty::mk_i64(), ty::mk_i64()),
                ty::mk_tup(tcx, vec!(ty::mk_i64(), ty::mk_bool()))),

            "u8_add_with_overflow" | "u8_sub_with_overflow" | "u8_mul_with_overflow" =>
                (0, vec!(ty::mk_u8(), ty::mk_u8()),
                ty::mk_tup(tcx, vec!(ty::mk_u8(), ty::mk_bool()))),

            "u16_add_with_overflow" | "u16_sub_with_overflow" | "u16_mul_with_overflow" =>
                (0, vec!(ty::mk_u16(), ty::mk_u16()),
                ty::mk_tup(tcx, vec!(ty::mk_u16(), ty::mk_bool()))),

            "u32_add_with_overflow" | "u32_sub_with_overflow" | "u32_mul_with_overflow"=>
                (0, vec!(ty::mk_u32(), ty::mk_u32()),
                ty::mk_tup(tcx, vec!(ty::mk_u32(), ty::mk_bool()))),

            "u64_add_with_overflow" | "u64_sub_with_overflow"  | "u64_mul_with_overflow" =>
                (0, vec!(ty::mk_u64(), ty::mk_u64()),
                ty::mk_tup(tcx, vec!(ty::mk_u64(), ty::mk_bool()))),

            "return_address" => (0, vec![], ty::mk_imm_ptr(tcx, ty::mk_u8())),

            ref other => {
                span_err!(tcx.sess, it.span, E0093,
                    "unrecognized intrinsic function: `{}`", *other);
                return;
            }
        }
    };
    let fty = ty::mk_bare_fn(tcx, ty::BareFnTy {
        fn_style: ast::UnsafeFn,
        abi: abi::RustIntrinsic,
        sig: FnSig {
            binder_id: it.id,
            inputs: inputs,
            output: output,
            variadic: false,
        }
    });
    let i_ty = ty::lookup_item_type(ccx.tcx, local_def(it.id));
    let i_n_tps = i_ty.generics.types.len(subst::FnSpace);
    if i_n_tps != n_tps {
        span_err!(tcx.sess, it.span, E0094,
            "intrinsic has wrong number of type \
             parameters: found {}, expected {}",
             i_n_tps, n_tps);
    } else {
        require_same_types(tcx,
                           None,
                           false,
                           it.span,
                           i_ty.ty,
                           fty,
                           || {
                format!("intrinsic has wrong type: expected `{}`",
                        ppaux::ty_to_string(ccx.tcx, fty))
            });
    }
}

impl Repr for RegionObligation {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        format!("RegionObligation(sub_region={}, sup_type={}, origin={})",
                self.sub_region.repr(tcx),
                self.sup_type.repr(tcx),
                self.origin.repr(tcx))
    }
}
