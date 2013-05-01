// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

The main entry point is `check_crate()`.  Type checking operates in two major
phases: collect and check.  The collect phase passes over all items and
determines their type, without examining their "innards".  The check phase
then checks function bodies and so forth.

Within the check phase, we check each function body one at a time (bodies of
function expressions are checked as part of the containing function).
Inference is used to supply types wherever they are unknown. The actual
checking of a function itself has several phases (check, regionck, writeback),
as discussed in the documentation for the `check` module.

The type checker is defined into various submodules which are documented
independently:

- astconv: converts the AST representation of types
  into the `ty` representation

- collect: computes the types of each top-level item and enters them into
  the `cx.tcache` table for later use

- check: walks over function bodies and type checks them, inferring types for
  local variables, type parameters, etc as necessary.

- infer: finds the types to use for each type variable such that
  all subtyping and assignment constraints are met.  In essence, the check
  module specifies the constraints, and the infer module solves them.

*/

use driver::session;

use middle::resolve;
use middle::ty;
use util::common::time;
use util::ppaux::Repr;
use util::ppaux;

use core::hashmap::HashMap;
use std::list::List;
use std::list;
use syntax::codemap::span;
use syntax::print::pprust::*;
use syntax::{ast, ast_map, abi};
use syntax::opt_vec;

#[path = "check/mod.rs"]
pub mod check;
pub mod rscope;
pub mod astconv;
#[path = "infer/mod.rs"]
pub mod infer;
pub mod collect;
pub mod coherence;

#[auto_encode]
#[auto_decode]
pub enum method_origin {
    // supertrait method invoked on "self" inside a default method
    // first field is supertrait ID;
    // second field is method index (relative to the *supertrait*
    // method list)
    method_super(ast::def_id, uint),

    // fully statically resolved method
    method_static(ast::def_id),

    // method invoked on a type parameter with a bounded trait
    method_param(method_param),

    // method invoked on a trait instance
    method_trait(ast::def_id, uint, ty::TraitStore),

    // method invoked on "self" inside a default method
    method_self(ast::def_id, uint)

}

// details for a method invoked with a receiver whose type is a type parameter
// with a bounded trait.
#[auto_encode]
#[auto_decode]
pub struct method_param {
    // the trait containing the method to be invoked
    trait_id: ast::def_id,

    // index of the method to be invoked amongst the trait's methods
    method_num: uint,

    // index of the type parameter (from those that are in scope) that is
    // the type of the receiver
    param_num: uint,

    // index of the bound for this type parameter which specifies the trait
    bound_num: uint,
}

pub struct method_map_entry {
    // the type of the self parameter, which is not reflected in the fn type
    // (FIXME #3446)
    self_arg: ty::arg,

    // the mode of `self`
    self_mode: ty::SelfMode,

    // the type of explicit self on the method
    explicit_self: ast::self_ty_,

    // method details being invoked
    origin: method_origin,
}

// maps from an expression id that corresponds to a method call to the details
// of the method to be invoked
pub type method_map = @mut HashMap<ast::node_id, method_map_entry>;

// Resolutions for bounds of all parameters, left to right, for a given path.
pub type vtable_res = @~[vtable_origin];

pub enum vtable_origin {
    /*
      Statically known vtable. def_id gives the class or impl item
      from whence comes the vtable, and tys are the type substs.
      vtable_res is the vtable itself
     */
    vtable_static(ast::def_id, ~[ty::t], vtable_res),

    /*
      Dynamic vtable, comes from a parameter that has a bound on it:
      fn foo<T:quux,baz,bar>(a: T) -- a's vtable would have a
      vtable_param origin

      The first uint is the param number (identifying T in the example),
      and the second is the bound number (identifying baz)
     */
    vtable_param(uint, uint)
}

impl Repr for vtable_origin {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        match *self {
            vtable_static(def_id, ref tys, ref vtable_res) => {
                fmt!("vtable_static(%?:%s, %s, %s)",
                     def_id,
                     ty::item_path_str(tcx, def_id),
                     tys.repr(tcx),
                     vtable_res.repr(tcx))
            }

            vtable_param(x, y) => {
                fmt!("vtable_param(%?, %?)", x, y)
            }
        }
    }
}

pub type vtable_map = @mut HashMap<ast::node_id, vtable_res>;

pub struct CrateCtxt {
    // A mapping from method call sites to traits that have that method.
    trait_map: resolve::TraitMap,
    method_map: method_map,
    vtable_map: vtable_map,
    coherence_info: @coherence::CoherenceInfo,
    tcx: ty::ctxt
}

// Functions that write types into the node type table
pub fn write_ty_to_tcx(tcx: ty::ctxt, node_id: ast::node_id, ty: ty::t) {
    debug!("write_ty_to_tcx(%d, %s)", node_id, ppaux::ty_to_str(tcx, ty));
    assert!(!ty::type_needs_infer(ty));
    tcx.node_types.insert(node_id as uint, ty);
}
pub fn write_substs_to_tcx(tcx: ty::ctxt,
                           node_id: ast::node_id,
                           substs: ~[ty::t]) {
    if substs.len() > 0u {
        debug!("write_substs_to_tcx(%d, %?)", node_id,
               substs.map(|t| ppaux::ty_to_str(tcx, *t)));
        assert!(substs.all(|t| !ty::type_needs_infer(*t)));
        tcx.node_type_substs.insert(node_id, substs);
    }
}
pub fn write_tpt_to_tcx(tcx: ty::ctxt,
                        node_id: ast::node_id,
                        tpt: &ty::ty_param_substs_and_ty) {
    write_ty_to_tcx(tcx, node_id, tpt.ty);
    if !tpt.substs.tps.is_empty() {
        write_substs_to_tcx(tcx, node_id, copy tpt.substs.tps);
    }
}

pub fn lookup_def_tcx(tcx: ty::ctxt, sp: span, id: ast::node_id) -> ast::def {
    match tcx.def_map.find(&id) {
      Some(&x) => x,
      _ => {
        tcx.sess.span_fatal(sp, ~"internal error looking up a definition")
      }
    }
}

pub fn lookup_def_ccx(ccx: @mut CrateCtxt, sp: span, id: ast::node_id)
                   -> ast::def {
    lookup_def_tcx(ccx.tcx, sp, id)
}

pub fn no_params(t: ty::t) -> ty::ty_param_bounds_and_ty {
    ty::ty_param_bounds_and_ty {
        generics: ty::Generics {type_param_defs: @~[],
                                region_param: None},
        ty: t
    }
}

pub fn require_same_types(
    tcx: ty::ctxt,
    maybe_infcx: Option<@mut infer::InferCtxt>,
    t1_is_expected: bool,
    span: span,
    t1: ty::t,
    t2: ty::t,
    msg: &fn() -> ~str) -> bool {

    let l_tcx, l_infcx;
    match maybe_infcx {
      None => {
        l_tcx = tcx;
        l_infcx = infer::new_infer_ctxt(tcx);
      }
      Some(i) => {
        l_tcx = i.tcx;
        l_infcx = i;
      }
    }

    match infer::mk_eqty(l_infcx, t1_is_expected, span, t1, t2) {
        result::Ok(()) => true,
        result::Err(ref terr) => {
            l_tcx.sess.span_err(span, msg() + ~": " +
                                ty::type_err_to_str(l_tcx, terr));
            ty::note_and_explain_type_err(l_tcx, terr);
            false
        }
    }
}

// a list of mapping from in-scope-region-names ("isr") to the
// corresponding ty::Region
pub type isr_alist = @List<(ty::bound_region, ty::Region)>;

trait get_and_find_region {
    fn get(&self, br: ty::bound_region) -> ty::Region;
    fn find(&self, br: ty::bound_region) -> Option<ty::Region>;
}

impl get_and_find_region for isr_alist {
    fn get(&self, br: ty::bound_region) -> ty::Region {
        self.find(br).get()
    }

    fn find(&self, br: ty::bound_region) -> Option<ty::Region> {
        for list::each(*self) |isr| {
            let (isr_br, isr_r) = *isr;
            if isr_br == br { return Some(isr_r); }
        }
        return None;
    }
}

fn check_main_fn_ty(ccx: @mut CrateCtxt,
                    main_id: ast::node_id,
                    main_span: span) {
    let tcx = ccx.tcx;
    let main_t = ty::node_id_to_type(tcx, main_id);
    match ty::get(main_t).sty {
        ty::ty_bare_fn(ref fn_ty) => {
            match tcx.items.find(&main_id) {
                Some(&ast_map::node_item(it,_)) => {
                    match it.node {
                        ast::item_fn(_, _, _, ref ps, _)
                        if ps.is_parameterized() => {
                            tcx.sess.span_err(
                                main_span,
                                ~"main function is not allowed \
                                  to have type parameters");
                            return;
                        }
                        _ => ()
                    }
                }
                _ => ()
            }
            let mut ok = ty::type_is_nil(fn_ty.sig.output);
            let num_args = vec::len(fn_ty.sig.inputs);
            ok &= num_args == 0u;
            if !ok {
                tcx.sess.span_err(
                    main_span,
                    fmt!("Wrong type in main function: found `%s`, \
                          expected `fn() -> ()`",
                         ppaux::ty_to_str(tcx, main_t)));
            }
        }
        _ => {
            tcx.sess.span_bug(main_span,
                              ~"main has a non-function type: found `" +
                              ppaux::ty_to_str(tcx, main_t) + ~"`");
        }
    }
}

fn check_start_fn_ty(ccx: @mut CrateCtxt,
                     start_id: ast::node_id,
                     start_span: span) {
    let tcx = ccx.tcx;
    let start_t = ty::node_id_to_type(tcx, start_id);
    match ty::get(start_t).sty {
        ty::ty_bare_fn(_) => {
            match tcx.items.find(&start_id) {
                Some(&ast_map::node_item(it,_)) => {
                    match it.node {
                        ast::item_fn(_,_,_,ref ps,_)
                        if ps.is_parameterized() => {
                            tcx.sess.span_err(
                                start_span,
                                ~"start function is not allowed to have type \
                                parameters");
                            return;
                        }
                        _ => ()
                    }
                }
                _ => ()
            }

            fn arg(ty: ty::t) -> ty::arg {
                ty::arg {
                    ty: ty
                }
            }

            let se_ty = ty::mk_bare_fn(tcx, ty::BareFnTy {
                purity: ast::impure_fn,
                abis: abi::AbiSet::Rust(),
                sig: ty::FnSig {
                    bound_lifetime_names: opt_vec::Empty,
                    inputs: ~[
                        arg(ty::mk_int()),
                        arg(ty::mk_imm_ptr(tcx,
                                           ty::mk_imm_ptr(tcx, ty::mk_u8()))),
                        arg(ty::mk_imm_ptr(tcx, ty::mk_u8()))
                    ],
                    output: ty::mk_int()
                }
            });

            require_same_types(tcx, None, false, start_span, start_t, se_ty,
                || fmt!("start function expects type: `%s`", ppaux::ty_to_str(ccx.tcx, se_ty)));

        }
        _ => {
            tcx.sess.span_bug(start_span,
                              ~"start has a non-function type: found `" +
                              ppaux::ty_to_str(tcx, start_t) + ~"`");
        }
    }
}

fn check_for_entry_fn(ccx: @mut CrateCtxt) {
    let tcx = ccx.tcx;
    if !*tcx.sess.building_library {
        match *tcx.sess.entry_fn {
          Some((id, sp)) => match *tcx.sess.entry_type {
              Some(session::EntryMain) => check_main_fn_ty(ccx, id, sp),
              Some(session::EntryStart) => check_start_fn_ty(ccx, id, sp),
              None => tcx.sess.bug(~"entry function without a type")
          },
          None => tcx.sess.err(~"entry function not found")
        }
    }
}

pub fn check_crate(tcx: ty::ctxt,
                   trait_map: resolve::TraitMap,
                   crate: @ast::crate)
                -> (method_map, vtable_map) {
    let time_passes = tcx.sess.time_passes();
    let ccx = @mut CrateCtxt {
        trait_map: trait_map,
        method_map: @mut HashMap::new(),
        vtable_map: @mut HashMap::new(),
        coherence_info: @coherence::CoherenceInfo(),
        tcx: tcx
    };

    time(time_passes, ~"type collecting", ||
        collect::collect_item_types(ccx, crate));

    time(time_passes, ~"method resolution", ||
        coherence::check_coherence(ccx, crate));

    time(time_passes, ~"type checking", ||
        check::check_item_types(ccx, crate));

    check_for_entry_fn(ccx);
    tcx.sess.abort_if_errors();
    (ccx.method_map, ccx.vtable_map)
}
