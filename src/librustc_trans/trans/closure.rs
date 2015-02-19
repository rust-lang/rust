// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::link::mangle_internal_name_by_path_and_seq;
use llvm::ValueRef;
use middle::mem_categorization::Typer;
use trans::adt;
use trans::base::*;
use trans::build::*;
use trans::cleanup::{CleanupMethods, ScopeId};
use trans::common::*;
use trans::datum::{Datum, rvalue_scratch_datum};
use trans::datum::{Rvalue, ByValue};
use trans::debuginfo;
use trans::expr;
use trans::monomorphize::{self, MonoId};
use trans::type_of::*;
use middle::ty::{self, ClosureTyper};
use middle::subst::{Substs};
use session::config::FullDebugInfo;

use syntax::ast;
use syntax::ast_util;


fn load_closure_environment<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                        arg_scope_id: ScopeId,
                                        freevars: &[ty::Freevar])
                                        -> Block<'blk, 'tcx>
{
    let _icx = push_ctxt("closure::load_closure_environment");

    // Special case for small by-value selfs.
    let closure_id = ast_util::local_def(bcx.fcx.id);
    let self_type = self_type_for_closure(bcx.ccx(), closure_id,
                                                  node_id_type(bcx, closure_id.node));
    let kind = kind_for_closure(bcx.ccx(), closure_id);
    let llenv = if kind == ty::FnOnceClosureKind &&
            !arg_is_indirect(bcx.ccx(), self_type) {
        let datum = rvalue_scratch_datum(bcx,
                                         self_type,
                                         "closure_env");
        store_ty(bcx, bcx.fcx.llenv.unwrap(), datum.val, self_type);
        datum.val
    } else {
        bcx.fcx.llenv.unwrap()
    };

    // Store the pointer to closure data in an alloca for debug info because that's what the
    // llvm.dbg.declare intrinsic expects
    let env_pointer_alloca = if bcx.sess().opts.debuginfo == FullDebugInfo {
        let alloc = alloca(bcx, val_ty(llenv), "__debuginfo_env_ptr");
        Store(bcx, llenv, alloc);
        Some(alloc)
    } else {
        None
    };

    for (i, freevar) in freevars.iter().enumerate() {
        let upvar_id = ty::UpvarId { var_id: freevar.def.local_node_id(),
                                     closure_expr_id: closure_id.node };
        let upvar_capture = bcx.tcx().upvar_capture(upvar_id).unwrap();
        let mut upvar_ptr = GEPi(bcx, llenv, &[0, i]);
        let captured_by_ref = match upvar_capture {
            ty::UpvarCapture::ByValue => false,
            ty::UpvarCapture::ByRef(..) => {
                upvar_ptr = Load(bcx, upvar_ptr);
                true
            }
        };
        let def_id = freevar.def.def_id();
        bcx.fcx.llupvars.borrow_mut().insert(def_id.node, upvar_ptr);

        if kind == ty::FnOnceClosureKind && !captured_by_ref {
            bcx.fcx.schedule_drop_mem(arg_scope_id,
                                      upvar_ptr,
                                      node_id_type(bcx, def_id.node))
        }

        if let Some(env_pointer_alloca) = env_pointer_alloca {
            debuginfo::create_captured_var_metadata(
                bcx,
                def_id.node,
                env_pointer_alloca,
                i,
                captured_by_ref,
                freevar.span);
        }
    }

    bcx
}

pub enum ClosureEnv<'a> {
    NotClosure,
    Closure(&'a [ty::Freevar]),
}

impl<'a> ClosureEnv<'a> {
    pub fn load<'blk,'tcx>(self, bcx: Block<'blk, 'tcx>, arg_scope: ScopeId)
                           -> Block<'blk, 'tcx>
    {
        match self {
            ClosureEnv::NotClosure => bcx,
            ClosureEnv::Closure(freevars) => {
                if freevars.is_empty() {
                    bcx
                } else {
                    load_closure_environment(bcx, arg_scope, freevars)
                }
            }
        }
    }
}

/// Returns the LLVM function declaration for a closure, creating it if
/// necessary. If the ID does not correspond to a closure ID, returns None.
pub fn get_or_create_declaration_if_closure<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                      closure_id: ast::DefId,
                                                      substs: &Substs<'tcx>)
                                                      -> Option<Datum<'tcx, Rvalue>> {
    if !ccx.tcx().closure_kinds.borrow().contains_key(&closure_id) {
        // Not a closure.
        return None
    }

    let function_type = ty::node_id_to_type(ccx.tcx(), closure_id.node);
    let function_type = monomorphize::apply_param_substs(ccx.tcx(), substs, &function_type);

    // Normalize type so differences in regions and typedefs don't cause
    // duplicate declarations
    let function_type = erase_regions(ccx.tcx(), &function_type);
    let params = match function_type.sty {
        ty::ty_closure(_, _, substs) => &substs.types,
        _ => unreachable!()
    };
    let mono_id = MonoId {
        def: closure_id,
        params: params
    };

    match ccx.closure_vals().borrow().get(&mono_id) {
        Some(&llfn) => {
            debug!("get_or_create_declaration_if_closure(): found closure");
            return Some(Datum::new(llfn, function_type, Rvalue::new(ByValue)))
        }
        None => {}
    }

    let symbol = ccx.tcx().map.with_path(closure_id.node, |path| {
        mangle_internal_name_by_path_and_seq(path, "closure")
    });

    let llfn = decl_internal_rust_fn(ccx, function_type, &symbol[..]);

    // set an inline hint for all closures
    set_inline_hint(llfn);

    debug!("get_or_create_declaration_if_closure(): inserting new \
            closure {:?} (type {})",
           mono_id,
           ccx.tn().type_to_string(val_ty(llfn)));
    ccx.closure_vals().borrow_mut().insert(mono_id, llfn);

    Some(Datum::new(llfn, function_type, Rvalue::new(ByValue)))
}

pub enum Dest<'a, 'tcx: 'a> {
    SaveIn(Block<'a, 'tcx>, ValueRef),
    Ignore(&'a CrateContext<'a, 'tcx>)
}

pub fn trans_closure_expr<'a, 'tcx>(dest: Dest<'a, 'tcx>,
                                    decl: &ast::FnDecl,
                                    body: &ast::Block,
                                    id: ast::NodeId,
                                    param_substs: &'tcx Substs<'tcx>)
                                    -> Option<Block<'a, 'tcx>>
{
    let ccx = match dest {
        Dest::SaveIn(bcx, _) => bcx.ccx(),
        Dest::Ignore(ccx) => ccx
    };
    let tcx = ccx.tcx();
    let _icx = push_ctxt("closure::trans_closure");

    debug!("trans_closure()");

    let closure_id = ast_util::local_def(id);
    let llfn = get_or_create_declaration_if_closure(
        ccx,
        closure_id,
        param_substs).unwrap();

    // Get the type of this closure. Use the current `param_substs` as
    // the closure substitutions. This makes sense because the closure
    // takes the same set of type arguments as the enclosing fn, and
    // this function (`trans_closure`) is invoked at the point
    // of the closure expression.
    let typer = NormalizingClosureTyper::new(tcx);
    let function_type = typer.closure_type(closure_id, param_substs);

    let freevars: Vec<ty::Freevar> =
        ty::with_freevars(tcx, id, |fv| fv.iter().cloned().collect());

    let sig = ty::erase_late_bound_regions(tcx, &function_type.sig);

    trans_closure(ccx,
                  decl,
                  body,
                  llfn.val,
                  param_substs,
                  id,
                  &[],
                  sig.output,
                  function_type.abi,
                  ClosureEnv::Closure(&freevars[..]));

    // Don't hoist this to the top of the function. It's perfectly legitimate
    // to have a zero-size closure (in which case dest will be `Ignore`) and
    // we must still generate the closure body.
    let (mut bcx, dest_addr) = match dest {
        Dest::SaveIn(bcx, p) => (bcx, p),
        Dest::Ignore(_) => {
            debug!("trans_closure() ignoring result");
            return None;
        }
    };

    let repr = adt::represent_type(ccx, node_id_type(bcx, id));

    // Create the closure.
    for (i, freevar) in freevars.iter().enumerate() {
        let datum = expr::trans_local_var(bcx, freevar.def);
        let upvar_slot_dest = adt::trans_field_ptr(bcx,
                                                   &*repr,
                                                   dest_addr,
                                                   0,
                                                   i);
        let upvar_id = ty::UpvarId { var_id: freevar.def.local_node_id(),
                                     closure_expr_id: id };
        match tcx.upvar_capture(upvar_id).unwrap() {
            ty::UpvarCapture::ByValue => {
                bcx = datum.store_to(bcx, upvar_slot_dest);
            }
            ty::UpvarCapture::ByRef(..) => {
                Store(bcx, datum.to_llref(), upvar_slot_dest);
            }
        }
    }
    adt::trans_set_discr(bcx, &*repr, dest_addr, 0);

    Some(bcx)
}

