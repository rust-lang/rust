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

typeck.rs, an introduction

The type checker is responsible for:

1. Determining the type of each expression
2. Resolving methods and traits
3. Guaranteeing that most type rules are met ("most?", you say, "why most?"
   Well, dear reader, read on)

The main entry point is `check_crate()`.  Type checking operates in
several major phases:

1. The collect phase first passes over all items and determines their
   type, without examining their "innards".

2. Variance inference then runs to compute the variance of each parameter

3. Coherence checks for overlapping or orphaned impls

4. Finally, the check phase then checks function bodies and so forth.
   Within the check phase, we check each function body one at a time
   (bodies of function expressions are checked as part of the
   containing function).  Inference is used to supply types wherever
   they are unknown. The actual checking of a function itself has
   several phases (check, regionck, writeback), as discussed in the
   documentation for the `check` module.

The type checker is defined into various submodules which are documented
independently:

- astconv: converts the AST representation of types
  into the `ty` representation

- collect: computes the types of each top-level item and enters them into
  the `cx.tcache` table for later use

- coherence: enforces coherence rules, builds some tables

- variance: variance inference

- check: walks over function bodies and type checks them, inferring types for
  local variables, type parameters, etc as necessary.

- infer: finds the types to use for each type variable such that
  all subtyping and assignment constraints are met.  In essence, the check
  module specifies the constraints, and the infer module solves them.

*/

#![allow(non_camel_case_types)]

use driver::config;

use middle::def;
use middle::resolve;
use middle::subst;
use middle::subst::VecPerParamSpace;
use middle::ty;
use util::common::time;
use util::ppaux::Repr;
use util::ppaux;
use util::nodemap::{DefIdMap, FnvHashMap};

use std::cell::RefCell;
use syntax::codemap::Span;
use syntax::print::pprust::*;
use syntax::{ast, ast_map, abi};

pub mod check;
pub mod rscope;
pub mod astconv;
pub mod infer;
pub mod collect;
pub mod coherence;
pub mod variance;

#[deriving(Clone, Encodable, Decodable, PartialEq, PartialOrd)]
pub struct param_index {
    pub space: subst::ParamSpace,
    pub index: uint
}

#[deriving(Clone, Encodable, Decodable)]
pub enum MethodOrigin {
    // fully statically resolved method
    MethodStatic(ast::DefId),

    // fully statically resolved unboxed closure invocation
    MethodStaticUnboxedClosure(ast::DefId),

    // method invoked on a type parameter with a bounded trait
    MethodParam(MethodParam),

    // method invoked on a trait instance
    MethodObject(MethodObject),

}

// details for a method invoked with a receiver whose type is a type parameter
// with a bounded trait.
#[deriving(Clone, Encodable, Decodable)]
pub struct MethodParam {
    // the trait containing the method to be invoked
    pub trait_id: ast::DefId,

    // index of the method to be invoked amongst the trait's methods
    pub method_num: uint,

    // index of the type parameter (from those that are in scope) that is
    // the type of the receiver
    pub param_num: param_index,

    // index of the bound for this type parameter which specifies the trait
    pub bound_num: uint,
}

// details for a method invoked with a receiver whose type is an object
#[deriving(Clone, Encodable, Decodable)]
pub struct MethodObject {
    // the (super)trait containing the method to be invoked
    pub trait_id: ast::DefId,

    // the actual base trait id of the object
    pub object_trait_id: ast::DefId,

    // index of the method to be invoked amongst the trait's methods
    pub method_num: uint,

    // index into the actual runtime vtable.
    // the vtable is formed by concatenating together the method lists of
    // the base object trait and all supertraits;  this is the index into
    // that vtable
    pub real_index: uint,
}

#[deriving(Clone)]
pub struct MethodCallee {
    pub origin: MethodOrigin,
    pub ty: ty::t,
    pub substs: subst::Substs
}

/**
 * With method calls, we store some extra information in
 * side tables (i.e method_map, vtable_map). We use
 * MethodCall as a key to index into these tables instead of
 * just directly using the expression's NodeId. The reason
 * for this being that we may apply adjustments (coercions)
 * with the resulting expression also needing to use the
 * side tables. The problem with this is that we don't
 * assign a separate NodeId to this new expression
 * and so it would clash with the base expression if both
 * needed to add to the side tables. Thus to disambiguate
 * we also keep track of whether there's an adjustment in
 * our key.
 */
#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub struct MethodCall {
    pub expr_id: ast::NodeId,
    pub adjustment: ExprAdjustment
}

#[deriving(Clone, PartialEq, Eq, Hash, Show, Encodable, Decodable)]
pub enum ExprAdjustment {
    NoAdjustment,
    AutoDeref(uint),
    AutoObject
}

pub struct TypeAndSubsts {
    pub substs: subst::Substs,
    pub ty: ty::t,
}

impl MethodCall {
    pub fn expr(id: ast::NodeId) -> MethodCall {
        MethodCall {
            expr_id: id,
            adjustment: NoAdjustment
        }
    }

    pub fn autoobject(id: ast::NodeId) -> MethodCall {
        MethodCall {
            expr_id: id,
            adjustment: AutoObject
        }
    }

    pub fn autoderef(expr_id: ast::NodeId, autoderef: uint) -> MethodCall {
        MethodCall {
            expr_id: expr_id,
            adjustment: AutoDeref(1 + autoderef)
        }
    }
}

// maps from an expression id that corresponds to a method call to the details
// of the method to be invoked
pub type MethodMap = RefCell<FnvHashMap<MethodCall, MethodCallee>>;

pub type vtable_param_res = Vec<vtable_origin>;

// Resolutions for bounds of all parameters, left to right, for a given path.
pub type vtable_res = VecPerParamSpace<vtable_param_res>;

#[deriving(Clone)]
pub enum vtable_origin {
    /*
      Statically known vtable. def_id gives the impl item
      from whence comes the vtable, and tys are the type substs.
      vtable_res is the vtable itself.
     */
    vtable_static(ast::DefId, subst::Substs, vtable_res),

    /*
      Dynamic vtable, comes from a parameter that has a bound on it:
      fn foo<T:quux,baz,bar>(a: T) -- a's vtable would have a
      vtable_param origin

      The first argument is the param index (identifying T in the example),
      and the second is the bound number (identifying baz)
     */
    vtable_param(param_index, uint),

    /*
      Vtable automatically generated for an unboxed closure. The def ID is the
      ID of the closure expression.
     */
    vtable_unboxed_closure(ast::DefId),

    /*
      Asked to determine the vtable for ty_err. This is the value used
      for the vtables of `Self` in a virtual call like `foo.bar()`
      where `foo` is of object type. The same value is also used when
      type errors occur.
     */
    vtable_error,
}

impl Repr for vtable_origin {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            vtable_static(def_id, ref tys, ref vtable_res) => {
                format!("vtable_static({:?}:{}, {}, {})",
                        def_id,
                        ty::item_path_str(tcx, def_id),
                        tys.repr(tcx),
                        vtable_res.repr(tcx))
            }

            vtable_param(x, y) => {
                format!("vtable_param({:?}, {:?})", x, y)
            }

            vtable_unboxed_closure(def_id) => {
                format!("vtable_unboxed_closure({})", def_id)
            }

            vtable_error => {
                format!("vtable_error")
            }
        }
    }
}

pub type vtable_map = RefCell<FnvHashMap<MethodCall, vtable_res>>;


pub type impl_vtable_map = RefCell<DefIdMap<vtable_res>>;

pub struct CrateCtxt<'a> {
    // A mapping from method call sites to traits that have that method.
    trait_map: resolve::TraitMap,
    tcx: &'a ty::ctxt
}

// Functions that write types into the node type table
pub fn write_ty_to_tcx(tcx: &ty::ctxt, node_id: ast::NodeId, ty: ty::t) {
    debug!("write_ty_to_tcx({}, {})", node_id, ppaux::ty_to_string(tcx, ty));
    assert!(!ty::type_needs_infer(ty));
    tcx.node_types.borrow_mut().insert(node_id as uint, ty);
}
pub fn write_substs_to_tcx(tcx: &ty::ctxt,
                           node_id: ast::NodeId,
                           item_substs: ty::ItemSubsts) {
    if !item_substs.is_noop() {
        debug!("write_substs_to_tcx({}, {})",
               node_id,
               item_substs.repr(tcx));

        assert!(item_substs.substs.types.all(|t| !ty::type_needs_infer(*t)));

        tcx.item_substs.borrow_mut().insert(node_id, item_substs);
    }
}
pub fn lookup_def_tcx(tcx:&ty::ctxt, sp: Span, id: ast::NodeId) -> def::Def {
    match tcx.def_map.borrow().find(&id) {
        Some(&x) => x,
        _ => {
            tcx.sess.span_fatal(sp, "internal error looking up a definition")
        }
    }
}

pub fn lookup_def_ccx(ccx: &CrateCtxt, sp: Span, id: ast::NodeId)
                   -> def::Def {
    lookup_def_tcx(ccx.tcx, sp, id)
}

pub fn no_params(t: ty::t) -> ty::Polytype {
    ty::Polytype {
        generics: ty::Generics {types: VecPerParamSpace::empty(),
                                regions: VecPerParamSpace::empty()},
        ty: t
    }
}

pub fn require_same_types(tcx: &ty::ctxt,
                          maybe_infcx: Option<&infer::InferCtxt>,
                          t1_is_expected: bool,
                          span: Span,
                          t1: ty::t,
                          t2: ty::t,
                          msg: || -> String)
                          -> bool {
    let result = match maybe_infcx {
        None => {
            let infcx = infer::new_infer_ctxt(tcx);
            infer::mk_eqty(&infcx, t1_is_expected, infer::Misc(span), t1, t2)
        }
        Some(infcx) => {
            infer::mk_eqty(infcx, t1_is_expected, infer::Misc(span), t1, t2)
        }
    };

    match result {
        Ok(_) => true,
        Err(ref terr) => {
            tcx.sess.span_err(span,
                              format!("{}: {}",
                                      msg(),
                                      ty::type_err_to_str(tcx,
                                                          terr)).as_slice());
            ty::note_and_explain_type_err(tcx, terr);
            false
        }
    }
}

fn check_main_fn_ty(ccx: &CrateCtxt,
                    main_id: ast::NodeId,
                    main_span: Span) {
    let tcx = ccx.tcx;
    let main_t = ty::node_id_to_type(tcx, main_id);
    match ty::get(main_t).sty {
        ty::ty_bare_fn(..) => {
            match tcx.map.find(main_id) {
                Some(ast_map::NodeItem(it)) => {
                    match it.node {
                        ast::ItemFn(_, _, _, ref ps, _)
                        if ps.is_parameterized() => {
                            span_err!(ccx.tcx.sess, main_span, E0131,
                                      "main function is not allowed to have type parameters");
                            return;
                        }
                        _ => ()
                    }
                }
                _ => ()
            }
            let se_ty = ty::mk_bare_fn(tcx, ty::BareFnTy {
                fn_style: ast::NormalFn,
                abi: abi::Rust,
                sig: ty::FnSig {
                    binder_id: main_id,
                    inputs: Vec::new(),
                    output: ty::mk_nil(),
                    variadic: false
                }
            });

            require_same_types(tcx, None, false, main_span, main_t, se_ty,
                || {
                    format!("main function expects type: `{}`",
                            ppaux::ty_to_string(ccx.tcx, se_ty))
                });
        }
        _ => {
            tcx.sess.span_bug(main_span,
                              format!("main has a non-function type: found \
                                       `{}`",
                                      ppaux::ty_to_string(tcx,
                                                       main_t)).as_slice());
        }
    }
}

fn check_start_fn_ty(ccx: &CrateCtxt,
                     start_id: ast::NodeId,
                     start_span: Span) {
    let tcx = ccx.tcx;
    let start_t = ty::node_id_to_type(tcx, start_id);
    match ty::get(start_t).sty {
        ty::ty_bare_fn(_) => {
            match tcx.map.find(start_id) {
                Some(ast_map::NodeItem(it)) => {
                    match it.node {
                        ast::ItemFn(_,_,_,ref ps,_)
                        if ps.is_parameterized() => {
                            span_err!(tcx.sess, start_span, E0132,
                                      "start function is not allowed to have type parameters");
                            return;
                        }
                        _ => ()
                    }
                }
                _ => ()
            }

            let se_ty = ty::mk_bare_fn(tcx, ty::BareFnTy {
                fn_style: ast::NormalFn,
                abi: abi::Rust,
                sig: ty::FnSig {
                    binder_id: start_id,
                    inputs: vec!(
                        ty::mk_int(),
                        ty::mk_imm_ptr(tcx, ty::mk_imm_ptr(tcx, ty::mk_u8()))
                    ),
                    output: ty::mk_int(),
                    variadic: false
                }
            });

            require_same_types(tcx, None, false, start_span, start_t, se_ty,
                || {
                    format!("start function expects type: `{}`",
                            ppaux::ty_to_string(ccx.tcx, se_ty))
                });

        }
        _ => {
            tcx.sess.span_bug(start_span,
                              format!("start has a non-function type: found \
                                       `{}`",
                                      ppaux::ty_to_string(tcx,
                                                       start_t)).as_slice());
        }
    }
}

fn check_for_entry_fn(ccx: &CrateCtxt) {
    let tcx = ccx.tcx;
    match *tcx.sess.entry_fn.borrow() {
        Some((id, sp)) => match tcx.sess.entry_type.get() {
            Some(config::EntryMain) => check_main_fn_ty(ccx, id, sp),
            Some(config::EntryStart) => check_start_fn_ty(ccx, id, sp),
            Some(config::EntryNone) => {}
            None => tcx.sess.bug("entry function without a type")
        },
        None => {}
    }
}

pub fn check_crate(tcx: &ty::ctxt,
                   trait_map: resolve::TraitMap,
                   krate: &ast::Crate) {
    let time_passes = tcx.sess.time_passes();
    let ccx = CrateCtxt {
        trait_map: trait_map,
        tcx: tcx
    };

    time(time_passes, "type collecting", (), |_|
        collect::collect_item_types(&ccx, krate));

    // this ensures that later parts of type checking can assume that items
    // have valid types and not error
    tcx.sess.abort_if_errors();

    time(time_passes, "variance inference", (), |_|
         variance::infer_variance(tcx, krate));

    time(time_passes, "coherence checking", (), |_|
        coherence::check_coherence(&ccx, krate));

    time(time_passes, "type checking", (), |_|
        check::check_item_types(&ccx, krate));

    check_for_entry_fn(&ccx);
    tcx.sess.abort_if_errors();
}
