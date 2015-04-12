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
use llvm::{ValueRef, get_param};
use middle::mem_categorization::Typer;
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
use trans::monomorphize::{self, MonoId};
use trans::type_of::*;
use middle::ty::{self, ClosureTyper};
use middle::subst::Substs;
use session::config::FullDebugInfo;
use util::ppaux::Repr;

use syntax::abi::RustCall;
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
        ty::ty_closure(_, substs) => &substs.types,
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

    // Currently there’s only a single user of get_or_create_declaration_if_closure and it
    // unconditionally defines the function, therefore we use define_* here.
    let llfn = declare::define_internal_rust_fn(ccx, &symbol[..], function_type).unwrap_or_else(||{
        ccx.sess().bug(&format!("symbol `{}` already defined", symbol));
    });

    // set an inline hint for all closures
    attributes::inline(llfn, attributes::InlineAttr::Hint);

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
        let upvar_slot_dest = adt::trans_field_ptr(bcx, &*repr, dest_addr, 0, i);
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

pub fn trans_closure_method<'a, 'tcx>(ccx: &'a CrateContext<'a, 'tcx>,
                                      closure_def_id: ast::DefId,
                                      substs: Substs<'tcx>,
                                      node: ExprOrMethodCall,
                                      param_substs: &'tcx Substs<'tcx>,
                                      trait_closure_kind: ty::ClosureKind)
                                      -> ValueRef
{
    // The substitutions should have no type parameters remaining
    // after passing through fulfill_obligation
    let llfn = callee::trans_fn_ref_with_substs(ccx,
                                                closure_def_id,
                                                node,
                                                param_substs,
                                                substs.clone()).val;

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
    closure_def_id: ast::DefId,
    substs: Substs<'tcx>,
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
    closure_def_id: ast::DefId,
    substs: Substs<'tcx>,
    llreffn: ValueRef)
    -> ValueRef
{
    debug!("trans_fn_once_adapter_shim(closure_def_id={}, substs={}, llreffn={})",
           closure_def_id.repr(ccx.tcx()),
           substs.repr(ccx.tcx()),
           ccx.tn().val_to_string(llreffn));

    let tcx = ccx.tcx();
    let typer = NormalizingClosureTyper::new(tcx);

    // Find a version of the closure type. Substitute static for the
    // region since it doesn't really matter.
    let substs = tcx.mk_substs(substs);
    let closure_ty = ty::mk_closure(tcx, closure_def_id, substs);
    let ref_closure_ty = ty::mk_imm_rptr(tcx, tcx.mk_region(ty::ReStatic), closure_ty);

    // Make a version with the type of by-ref closure.
    let ty::ClosureTy { unsafety, abi, mut sig } = typer.closure_type(closure_def_id, substs);
    sig.0.inputs.insert(0, ref_closure_ty); // sig has no self type as of yet
    let llref_bare_fn_ty = tcx.mk_bare_fn(ty::BareFnTy { unsafety: unsafety,
                                                               abi: abi,
                                                               sig: sig.clone() });
    let llref_fn_ty = ty::mk_bare_fn(tcx, None, llref_bare_fn_ty);
    debug!("trans_fn_once_adapter_shim: llref_fn_ty={}",
           llref_fn_ty.repr(tcx));

    // Make a version of the closure type with the same arguments, but
    // with argument #0 being by value.
    assert_eq!(abi, RustCall);
    sig.0.inputs[0] = closure_ty;
    let llonce_bare_fn_ty = tcx.mk_bare_fn(ty::BareFnTy { unsafety: unsafety,
                                                                abi: abi,
                                                                sig: sig });
    let llonce_fn_ty = ty::mk_bare_fn(tcx, None, llonce_bare_fn_ty);

    // Create the by-value helper.
    let function_name = link::mangle_internal_name_by_type_and_seq(ccx, llonce_fn_ty, "once_shim");
    let lloncefn = declare::define_internal_rust_fn(ccx, &function_name[..], llonce_fn_ty)
        .unwrap_or_else(||{
            ccx.sess().bug(&format!("symbol `{}` already defined", function_name));
        });

    let sig = ty::erase_late_bound_regions(tcx, &llonce_bare_fn_ty.sig);
    let (block_arena, fcx): (TypedArena<_>, FunctionContext);
    block_arena = TypedArena::new();
    fcx = new_fn_ctxt(ccx,
                      lloncefn,
                      ast::DUMMY_NODE_ID,
                      false,
                      sig.output,
                      substs,
                      None,
                      &block_arena);
    let mut bcx = init_function(&fcx, false, sig.output);

    // the first argument (`self`) will be the (by value) closure env.
    let self_scope = fcx.push_custom_cleanup_scope();
    let self_scope_id = CustomScope(self_scope);
    let rvalue_mode = datum::appropriate_rvalue_mode(ccx, closure_ty);
    let llself = get_param(lloncefn, fcx.arg_pos(0) as u32);
    let env_datum = Datum::new(llself, closure_ty, Rvalue::new(rvalue_mode));
    let env_datum = unpack_datum!(bcx,
                                  env_datum.to_lvalue_datum_in_scope(bcx, "self",
                                                                     self_scope_id));

    debug!("trans_fn_once_adapter_shim: env_datum={}",
           bcx.val_to_string(env_datum.val));

    // the remaining arguments will be packed up in a tuple.
    let input_tys = match sig.inputs[1].sty {
        ty::ty_tup(ref tys) => &**tys,
        _ => bcx.sess().bug(&format!("trans_fn_once_adapter_shim: not rust-call! \
                                      closure_def_id={}",
                                     closure_def_id.repr(tcx)))
    };
    let llargs: Vec<_> =
        input_tys.iter()
                 .enumerate()
                 .map(|(i, _)| get_param(lloncefn, fcx.arg_pos(i+1) as u32))
                 .collect();

    let dest =
        fcx.llretslotptr.get().map(
            |_| expr::SaveIn(fcx.get_ret_slot(bcx, sig.output, "ret_slot")));

    let callee_data = TraitItem(MethodData { llfn: llreffn,
                                             llself: env_datum.val });

    bcx = callee::trans_call_inner(bcx,
                                   DebugLoc::None,
                                   llref_fn_ty,
                                   |bcx, _| Callee { bcx: bcx, data: callee_data },
                                   ArgVals(&llargs),
                                   dest).bcx;

    fcx.pop_custom_cleanup_scope(self_scope);

    finish_fn(&fcx, bcx, sig.output, DebugLoc::None);

    lloncefn
}
