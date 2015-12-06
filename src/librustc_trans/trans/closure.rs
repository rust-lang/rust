// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use arena::TypedArena;
use back::link::{self, mangle_internal_name_by_path_and_seq};
use llvm::{ValueRef, get_params};
use middle::def_id::DefId;
use middle::infer;
use trans::adt;
use trans::attributes;
use trans::base::*;
use trans::build::*;
use trans::callee::{self, ArgVals, Callee, TraitItem, MethodData};
use trans::cleanup::{CleanupMethods, CustomScope, ScopeId};
use trans::common::*;
use trans::datum::{self, Datum, rvalue_scratch_datum, Rvalue, ByValue};
use trans::debuginfo::{self, DebugLoc};
use trans::declare;
use trans::expr;
use trans::monomorphize::{MonoId};
use trans::type_of::*;
use middle::ty;
use session::config::FullDebugInfo;

use syntax::abi::RustCall;
use syntax::ast;

use rustc_front::hir;


fn load_closure_environment<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                        closure_def_id: DefId,
                                        arg_scope_id: ScopeId,
                                        freevars: &[ty::Freevar])
                                        -> Block<'blk, 'tcx>
{
    let _icx = push_ctxt("closure::load_closure_environment");

    // Special case for small by-value selfs.
    let closure_ty = node_id_type(bcx, bcx.fcx.id);
    let self_type = self_type_for_closure(bcx.ccx(), closure_def_id, closure_ty);
    let kind = kind_for_closure(bcx.ccx(), closure_def_id);
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
        let upvar_id = ty::UpvarId { var_id: freevar.def.var_id(),
                                     closure_expr_id: bcx.fcx.id };
        let upvar_capture = bcx.tcx().upvar_capture(upvar_id).unwrap();
        let mut upvar_ptr = StructGEP(bcx, llenv, i);
        let captured_by_ref = match upvar_capture {
            ty::UpvarCapture::ByValue => false,
            ty::UpvarCapture::ByRef(..) => {
                upvar_ptr = Load(bcx, upvar_ptr);
                true
            }
        };
        let node_id = freevar.def.var_id();
        bcx.fcx.llupvars.borrow_mut().insert(node_id, upvar_ptr);

        if kind == ty::FnOnceClosureKind && !captured_by_ref {
            let hint = bcx.fcx.lldropflag_hints.borrow().hint_datum(upvar_id.var_id);
            bcx.fcx.schedule_drop_mem(arg_scope_id,
                                      upvar_ptr,
                                      node_id_type(bcx, node_id),
                                      hint)
        }

        if let Some(env_pointer_alloca) = env_pointer_alloca {
            debuginfo::create_captured_var_metadata(
                bcx,
                node_id,
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
    Closure(DefId, &'a [ty::Freevar]),
}

impl<'a> ClosureEnv<'a> {
    pub fn load<'blk,'tcx>(self, bcx: Block<'blk, 'tcx>, arg_scope: ScopeId)
                           -> Block<'blk, 'tcx>
    {
        match self {
            ClosureEnv::NotClosure => bcx,
            ClosureEnv::Closure(def_id, freevars) => {
                if freevars.is_empty() {
                    bcx
                } else {
                    load_closure_environment(bcx, def_id, arg_scope, freevars)
                }
            }
        }
    }
}

/// Returns the LLVM function declaration for a closure, creating it if
/// necessary. If the ID does not correspond to a closure ID, returns None.
pub fn get_or_create_closure_declaration<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                   closure_id: DefId,
                                                   substs: &ty::ClosureSubsts<'tcx>)
                                                   -> ValueRef {
    // Normalize type so differences in regions and typedefs don't cause
    // duplicate declarations
    let substs = ccx.tcx().erase_regions(substs);
    let mono_id = MonoId {
        def: closure_id,
        params: &substs.func_substs.types
    };

    if let Some(&llfn) = ccx.closure_vals().borrow().get(&mono_id) {
        debug!("get_or_create_closure_declaration(): found closure {:?}: {:?}",
               mono_id, ccx.tn().val_to_string(llfn));
        return llfn;
    }

    let path = ccx.tcx().def_path(closure_id);
    let symbol = mangle_internal_name_by_path_and_seq(path, "closure");

    let function_type = ccx.tcx().mk_closure_from_closure_substs(closure_id, Box::new(substs));
    let llfn = declare::define_internal_rust_fn(ccx, &symbol[..], function_type);

    // set an inline hint for all closures
    attributes::inline(llfn, attributes::InlineAttr::Hint);

    debug!("get_or_create_declaration_if_closure(): inserting new \
            closure {:?} (type {}): {:?}",
           mono_id,
           ccx.tn().type_to_string(val_ty(llfn)),
           ccx.tn().val_to_string(llfn));
    ccx.closure_vals().borrow_mut().insert(mono_id, llfn);

    llfn
}

pub enum Dest<'a, 'tcx: 'a> {
    SaveIn(Block<'a, 'tcx>, ValueRef),
    Ignore(&'a CrateContext<'a, 'tcx>)
}

pub fn trans_closure_expr<'a, 'tcx>(dest: Dest<'a, 'tcx>,
                                    decl: &hir::FnDecl,
                                    body: &hir::Block,
                                    id: ast::NodeId,
                                    closure_def_id: DefId, // (*)
                                    closure_substs: &'tcx ty::ClosureSubsts<'tcx>)
                                    -> Option<Block<'a, 'tcx>>
{
    // (*) Note that in the case of inlined functions, the `closure_def_id` will be the
    // defid of the closure in its original crate, whereas `id` will be the id of the local
    // inlined copy.

    let param_substs = closure_substs.func_substs;

    let ccx = match dest {
        Dest::SaveIn(bcx, _) => bcx.ccx(),
        Dest::Ignore(ccx) => ccx
    };
    let tcx = ccx.tcx();
    let _icx = push_ctxt("closure::trans_closure_expr");

    debug!("trans_closure_expr(id={:?}, closure_def_id={:?}, closure_substs={:?})",
           id, closure_def_id, closure_substs);

    let llfn = get_or_create_closure_declaration(ccx, closure_def_id, closure_substs);

    // Get the type of this closure. Use the current `param_substs` as
    // the closure substitutions. This makes sense because the closure
    // takes the same set of type arguments as the enclosing fn, and
    // this function (`trans_closure`) is invoked at the point
    // of the closure expression.

    let infcx = infer::normalizing_infer_ctxt(ccx.tcx(), &ccx.tcx().tables);
    let function_type = infcx.closure_type(closure_def_id, closure_substs);

    let freevars: Vec<ty::Freevar> =
        tcx.with_freevars(id, |fv| fv.iter().cloned().collect());

    let sig = tcx.erase_late_bound_regions(&function_type.sig);
    let sig = infer::normalize_associated_type(ccx.tcx(), &sig);

    trans_closure(ccx,
                  decl,
                  body,
                  llfn,
                  param_substs,
                  id,
                  &[],
                  sig.output,
                  function_type.abi,
                  ClosureEnv::Closure(closure_def_id, &freevars));

    // Don't hoist this to the top of the function. It's perfectly legitimate
    // to have a zero-size closure (in which case dest will be `Ignore`) and
    // we must still generate the closure body.
    let (mut bcx, dest_addr) = match dest {
        Dest::SaveIn(bcx, p) => (bcx, p),
        Dest::Ignore(_) => {
            debug!("trans_closure_expr() ignoring result");
            return None;
        }
    };

    let repr = adt::represent_type(ccx, node_id_type(bcx, id));

    // Create the closure.
    for (i, freevar) in freevars.iter().enumerate() {
        let datum = expr::trans_local_var(bcx, freevar.def);
        let upvar_slot_dest = adt::trans_field_ptr(
            bcx, &*repr, adt::MaybeSizedValue::sized(dest_addr), 0, i);
        let upvar_id = ty::UpvarId { var_id: freevar.def.var_id(),
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

pub fn trans_closure_method<'a, 'tcx>(ccx: &'a CrateContext<'a, 'tcx>,
                                      closure_def_id: DefId,
                                      substs: ty::ClosureSubsts<'tcx>,
                                      trait_closure_kind: ty::ClosureKind)
                                      -> ValueRef
{
    // If this is a closure, redirect to it.
    let llfn = get_or_create_closure_declaration(ccx, closure_def_id, &substs);

    // If the closure is a Fn closure, but a FnOnce is needed (etc),
    // then adapt the self type
    let closure_kind = ccx.tcx().closure_kind(closure_def_id);
    trans_closure_adapter_shim(ccx,
                               closure_def_id,
                               substs,
                               closure_kind,
                               trait_closure_kind,
                               llfn)
}

fn trans_closure_adapter_shim<'a, 'tcx>(
    ccx: &'a CrateContext<'a, 'tcx>,
    closure_def_id: DefId,
    substs: ty::ClosureSubsts<'tcx>,
    llfn_closure_kind: ty::ClosureKind,
    trait_closure_kind: ty::ClosureKind,
    llfn: ValueRef)
    -> ValueRef
{
    let _icx = push_ctxt("trans_closure_adapter_shim");
    let tcx = ccx.tcx();

    debug!("trans_closure_adapter_shim(llfn_closure_kind={:?}, \
           trait_closure_kind={:?}, \
           llfn={})",
           llfn_closure_kind,
           trait_closure_kind,
           ccx.tn().val_to_string(llfn));

    match (llfn_closure_kind, trait_closure_kind) {
        (ty::FnClosureKind, ty::FnClosureKind) |
        (ty::FnMutClosureKind, ty::FnMutClosureKind) |
        (ty::FnOnceClosureKind, ty::FnOnceClosureKind) => {
            // No adapter needed.
            llfn
        }
        (ty::FnClosureKind, ty::FnMutClosureKind) => {
            // The closure fn `llfn` is a `fn(&self, ...)`.  We want a
            // `fn(&mut self, ...)`. In fact, at trans time, these are
            // basically the same thing, so we can just return llfn.
            llfn
        }
        (ty::FnClosureKind, ty::FnOnceClosureKind) |
        (ty::FnMutClosureKind, ty::FnOnceClosureKind) => {
            // The closure fn `llfn` is a `fn(&self, ...)` or `fn(&mut
            // self, ...)`.  We want a `fn(self, ...)`. We can produce
            // this by doing something like:
            //
            //     fn call_once(self, ...) { call_mut(&self, ...) }
            //     fn call_once(mut self, ...) { call_mut(&mut self, ...) }
            //
            // These are both the same at trans time.
            trans_fn_once_adapter_shim(ccx, closure_def_id, substs, llfn)
        }
        _ => {
            tcx.sess.bug(&format!("trans_closure_adapter_shim: cannot convert {:?} to {:?}",
                                  llfn_closure_kind,
                                  trait_closure_kind));
        }
    }
}

fn trans_fn_once_adapter_shim<'a, 'tcx>(
    ccx: &'a CrateContext<'a, 'tcx>,
    closure_def_id: DefId,
    substs: ty::ClosureSubsts<'tcx>,
    llreffn: ValueRef)
    -> ValueRef
{
    debug!("trans_fn_once_adapter_shim(closure_def_id={:?}, substs={:?}, llreffn={})",
           closure_def_id,
           substs,
           ccx.tn().val_to_string(llreffn));

    let tcx = ccx.tcx();
    let infcx = infer::normalizing_infer_ctxt(ccx.tcx(), &ccx.tcx().tables);

    // Find a version of the closure type. Substitute static for the
    // region since it doesn't really matter.
    let closure_ty = tcx.mk_closure_from_closure_substs(closure_def_id, Box::new(substs.clone()));
    let ref_closure_ty = tcx.mk_imm_ref(tcx.mk_region(ty::ReStatic), closure_ty);

    // Make a version with the type of by-ref closure.
    let ty::ClosureTy { unsafety, abi, mut sig } = infcx.closure_type(closure_def_id, &substs);
    sig.0.inputs.insert(0, ref_closure_ty); // sig has no self type as of yet
    let llref_bare_fn_ty = tcx.mk_bare_fn(ty::BareFnTy { unsafety: unsafety,
                                                               abi: abi,
                                                               sig: sig.clone() });
    let llref_fn_ty = tcx.mk_fn(None, llref_bare_fn_ty);
    debug!("trans_fn_once_adapter_shim: llref_fn_ty={:?}",
           llref_fn_ty);

    // Make a version of the closure type with the same arguments, but
    // with argument #0 being by value.
    assert_eq!(abi, RustCall);
    sig.0.inputs[0] = closure_ty;
    let llonce_bare_fn_ty = tcx.mk_bare_fn(ty::BareFnTy { unsafety: unsafety,
                                                                abi: abi,
                                                                sig: sig });
    let llonce_fn_ty = tcx.mk_fn(None, llonce_bare_fn_ty);

    // Create the by-value helper.
    let function_name = link::mangle_internal_name_by_type_and_seq(ccx, llonce_fn_ty, "once_shim");
    let lloncefn = declare::define_internal_rust_fn(ccx, &function_name,
                                                    llonce_fn_ty);
    let sig = tcx.erase_late_bound_regions(&llonce_bare_fn_ty.sig);
    let sig = infer::normalize_associated_type(ccx.tcx(), &sig);

    let (block_arena, fcx): (TypedArena<_>, FunctionContext);
    block_arena = TypedArena::new();
    fcx = new_fn_ctxt(ccx,
                      lloncefn,
                      ast::DUMMY_NODE_ID,
                      false,
                      sig.output,
                      substs.func_substs,
                      None,
                      &block_arena);
    let mut bcx = init_function(&fcx, false, sig.output);

    let llargs = get_params(fcx.llfn);

    // the first argument (`self`) will be the (by value) closure env.
    let self_scope = fcx.push_custom_cleanup_scope();
    let self_scope_id = CustomScope(self_scope);
    let rvalue_mode = datum::appropriate_rvalue_mode(ccx, closure_ty);
    let self_idx = fcx.arg_offset();
    let llself = llargs[self_idx];
    let env_datum = Datum::new(llself, closure_ty, Rvalue::new(rvalue_mode));
    let env_datum = unpack_datum!(bcx,
                                  env_datum.to_lvalue_datum_in_scope(bcx, "self",
                                                                     self_scope_id));

    debug!("trans_fn_once_adapter_shim: env_datum={}",
           bcx.val_to_string(env_datum.val));

    let dest =
        fcx.llretslotptr.get().map(
            |_| expr::SaveIn(fcx.get_ret_slot(bcx, sig.output, "ret_slot")));

    let callee_data = TraitItem(MethodData { llfn: llreffn,
                                             llself: env_datum.val });

    bcx = callee::trans_call_inner(bcx, DebugLoc::None, |bcx, _| {
        Callee {
            bcx: bcx,
            data: callee_data,
            ty: llref_fn_ty
        }
    }, ArgVals(&llargs[(self_idx + 1)..]), dest).bcx;

    fcx.pop_custom_cleanup_scope(self_scope);

    finish_fn(&fcx, bcx, sig.output, DebugLoc::None);

    lloncefn
}
