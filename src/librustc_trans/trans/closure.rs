// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::ClosureKind::*;

use back::link::mangle_internal_name_by_path_and_seq;
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
use middle::ty::{self, UnboxedClosureTyper};
use middle::subst::{Substs};
use session::config::FullDebugInfo;

use syntax::ast;
use syntax::ast_util;


fn load_unboxed_closure_environment<'blk, 'tcx>(
                                    bcx: Block<'blk, 'tcx>,
                                    arg_scope_id: ScopeId,
                                    freevar_mode: ast::CaptureClause,
                                    freevars: &[ty::Freevar])
                                    -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("closure::load_unboxed_closure_environment");

    // Special case for small by-value selfs.
    let closure_id = ast_util::local_def(bcx.fcx.id);
    let self_type = self_type_for_unboxed_closure(bcx.ccx(), closure_id,
                                                  node_id_type(bcx, closure_id.node));
    let kind = kind_for_unboxed_closure(bcx.ccx(), closure_id);
    let llenv = if kind == ty::FnOnceUnboxedClosureKind &&
            !arg_is_indirect(bcx.ccx(), self_type) {
        let datum = rvalue_scratch_datum(bcx,
                                         self_type,
                                         "unboxed_closure_env");
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
        let mut upvar_ptr = GEPi(bcx, llenv, &[0, i]);
        let captured_by_ref = match freevar_mode {
            ast::CaptureByRef => {
                upvar_ptr = Load(bcx, upvar_ptr);
                true
            }
            ast::CaptureByValue => false
        };
        let def_id = freevar.def.def_id();
        bcx.fcx.llupvars.borrow_mut().insert(def_id.node, upvar_ptr);

        if kind == ty::FnOnceUnboxedClosureKind && freevar_mode == ast::CaptureByValue {
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

#[derive(PartialEq)]
pub enum ClosureKind<'tcx> {
    NotClosure,
    // See load_unboxed_closure_environment.
    UnboxedClosure(ast::CaptureClause)
}

pub struct ClosureEnv<'a, 'tcx> {
    freevars: &'a [ty::Freevar],
    pub kind: ClosureKind<'tcx>
}

impl<'a, 'tcx> ClosureEnv<'a, 'tcx> {
    pub fn new(freevars: &'a [ty::Freevar], kind: ClosureKind<'tcx>)
               -> ClosureEnv<'a, 'tcx> {
        ClosureEnv {
            freevars: freevars,
            kind: kind
        }
    }

    pub fn load<'blk>(self, bcx: Block<'blk, 'tcx>, arg_scope: ScopeId)
                      -> Block<'blk, 'tcx> {
        // Don't bother to create the block if there's nothing to load
        if self.freevars.is_empty() {
            return bcx;
        }

        match self.kind {
            NotClosure => bcx,
            UnboxedClosure(freevar_mode) => {
                load_unboxed_closure_environment(bcx, arg_scope, freevar_mode, self.freevars)
            }
        }
    }
}

/// Returns the LLVM function declaration for an unboxed closure, creating it
/// if necessary. If the ID does not correspond to a closure ID, returns None.
pub fn get_or_create_declaration_if_unboxed_closure<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                              closure_id: ast::DefId,
                                                              substs: &Substs<'tcx>)
                                                              -> Option<Datum<'tcx, Rvalue>> {
    if !ccx.tcx().unboxed_closures.borrow().contains_key(&closure_id) {
        // Not an unboxed closure.
        return None
    }

    let function_type = ty::node_id_to_type(ccx.tcx(), closure_id.node);
    let function_type = monomorphize::apply_param_substs(ccx.tcx(), substs, &function_type);

    // Normalize type so differences in regions and typedefs don't cause
    // duplicate declarations
    let function_type = erase_regions(ccx.tcx(), &function_type);
    let params = match function_type.sty {
        ty::ty_unboxed_closure(_, _, ref substs) => substs.types.clone(),
        _ => unreachable!()
    };
    let mono_id = MonoId {
        def: closure_id,
        params: params
    };

    match ccx.unboxed_closure_vals().borrow().get(&mono_id) {
        Some(&llfn) => {
            debug!("get_or_create_declaration_if_unboxed_closure(): found \
                    closure");
            return Some(Datum::new(llfn, function_type, Rvalue::new(ByValue)))
        }
        None => {}
    }

    let symbol = ccx.tcx().map.with_path(closure_id.node, |path| {
        mangle_internal_name_by_path_and_seq(path, "unboxed_closure")
    });

    let llfn = decl_internal_rust_fn(ccx, function_type, &symbol[]);

    // set an inline hint for all closures
    set_inline_hint(llfn);

    debug!("get_or_create_declaration_if_unboxed_closure(): inserting new \
            closure {:?} (type {})",
           mono_id,
           ccx.tn().type_to_string(val_ty(llfn)));
    ccx.unboxed_closure_vals().borrow_mut().insert(mono_id, llfn);

    Some(Datum::new(llfn, function_type, Rvalue::new(ByValue)))
}

pub fn trans_unboxed_closure<'blk, 'tcx>(
                             mut bcx: Block<'blk, 'tcx>,
                             decl: &ast::FnDecl,
                             body: &ast::Block,
                             id: ast::NodeId,
                             dest: expr::Dest)
                             -> Block<'blk, 'tcx>
{
    let _icx = push_ctxt("closure::trans_unboxed_closure");

    debug!("trans_unboxed_closure()");

    let closure_id = ast_util::local_def(id);
    let llfn = get_or_create_declaration_if_unboxed_closure(
        bcx.ccx(),
        closure_id,
        bcx.fcx.param_substs).unwrap();

    // Get the type of this closure. Use the current `param_substs` as
    // the closure substitutions. This makes sense because the closure
    // takes the same set of type arguments as the enclosing fn, and
    // this function (`trans_unboxed_closure`) is invoked at the point
    // of the closure expression.
    let typer = NormalizingUnboxedClosureTyper::new(bcx.tcx());
    let function_type = typer.unboxed_closure_type(closure_id, bcx.fcx.param_substs);

    let freevars: Vec<ty::Freevar> =
        ty::with_freevars(bcx.tcx(), id, |fv| fv.iter().map(|&fv| fv).collect());
    let freevar_mode = bcx.tcx().capture_mode(id);

    let sig = ty::erase_late_bound_regions(bcx.tcx(), &function_type.sig);

    trans_closure(bcx.ccx(),
                  decl,
                  body,
                  llfn.val,
                  bcx.fcx.param_substs,
                  id,
                  &[],
                  sig.output,
                  function_type.abi,
                  ClosureEnv::new(&freevars[],
                                  UnboxedClosure(freevar_mode)));

    // Don't hoist this to the top of the function. It's perfectly legitimate
    // to have a zero-size unboxed closure (in which case dest will be
    // `Ignore`) and we must still generate the closure body.
    let dest_addr = match dest {
        expr::SaveIn(p) => p,
        expr::Ignore => {
            debug!("trans_unboxed_closure() ignoring result");
            return bcx
        }
    };

    let repr = adt::represent_type(bcx.ccx(), node_id_type(bcx, id));

    // Create the closure.
    for (i, freevar) in freevars.iter().enumerate() {
        let datum = expr::trans_local_var(bcx, freevar.def);
        let upvar_slot_dest = adt::trans_field_ptr(bcx,
                                                   &*repr,
                                                   dest_addr,
                                                   0,
                                                   i);
        match freevar_mode {
            ast::CaptureByValue => {
                bcx = datum.store_to(bcx, upvar_slot_dest);
            }
            ast::CaptureByRef => {
                Store(bcx, datum.to_llref(), upvar_slot_dest);
            }
        }
    }
    adt::trans_set_discr(bcx, &*repr, dest_addr, 0);

    bcx
}

