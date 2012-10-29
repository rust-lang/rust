// Translation of automatically-derived trait implementations. This handles
// enums and structs only; other types cannot be automatically derived.

use lib::llvm::llvm;
use middle::trans::base::{finish_fn, get_insn_ctxt, get_item_val};
use middle::trans::base::{new_fn_ctxt, sub_block, top_scope_block};
use middle::trans::build::{Br, CondBr, GEPi, Load, PointerCast, Store};
use middle::trans::build::{ValueRef};
use middle::trans::callee;
use middle::trans::callee::{ArgVals, Callee, DontAutorefArg, Method};
use middle::trans::callee::{MethodData};
use middle::trans::common;
use middle::trans::common::{C_bool, T_ptr, block, crate_ctxt};
use middle::trans::expr::SaveIn;
use middle::trans::type_of::type_of;
use middle::typeck::method_static;
use syntax::ast;
use syntax::ast::{def_id, ident, node_id, ty_param};
use syntax::ast_map::path;
use syntax::ast_util;
use syntax::ast_util::local_def;

/// The main "translation" pass for automatically-derived impls. Generates
/// code for monomorphic methods only. Other methods will be generated when
/// they are invoked with specific type parameters; see
/// `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
pub fn trans_deriving_impl(ccx: @crate_ctxt, _path: path, _name: ident,
                           tps: ~[ty_param], id: node_id) {
    let _icx = ccx.insn_ctxt("deriving::trans_deriving_impl");
    if tps.len() > 0 { return; }

    let impl_def_id = local_def(id);
    let self_ty = ty::lookup_item_type(ccx.tcx, impl_def_id);
    let method_dids = ccx.tcx.automatically_derived_methods_for_impl.get(
        impl_def_id);

    for method_dids.each |method_did| {
        let llfn = get_item_val(ccx, method_did.node);
        match ty::get(self_ty.ty).sty {
            ty::ty_class(*) => {
                trans_deriving_struct_method(ccx, llfn, impl_def_id,
                                             self_ty.ty);
            }
            _ => {
                ccx.tcx.sess.unimpl(~"translation of non-struct deriving \
                                      method");
            }
        }
    }
}

fn trans_deriving_struct_method(ccx: @crate_ctxt, llfn: ValueRef,
                                impl_did: def_id, self_ty: ty::t) {
    let _icx = ccx.insn_ctxt("trans_deriving_struct_method");
    let fcx = new_fn_ctxt(ccx, ~[], llfn, None);
    let top_bcx = top_scope_block(fcx, None);
    let lltop = top_bcx.llbb;
    let mut bcx = top_bcx;

    let llselfty = type_of(ccx, self_ty);
    let llselfval = PointerCast(bcx, fcx.llenv, T_ptr(llselfty));
    let llotherval = llvm::LLVMGetParam(llfn, 2);

    let struct_field_tys;
    match ty::get(self_ty).sty {
        ty::ty_class(struct_id, ref struct_substs) => {
            struct_field_tys = ty::class_items_as_fields(
                ccx.tcx, struct_id, struct_substs);
        }
        _ => {
            ccx.tcx.sess.bug(~"passed non-struct to \
                               trans_deriving_struct_method");
        }
    }

    // Iterate over every element of the struct.
    for ccx.tcx.deriving_struct_methods.get(impl_did).eachi
            |i, derived_method_info| {
        let target_method_def_id;
        match *derived_method_info {
            method_static(did) => target_method_def_id = did,
            _ => fail ~"derived method didn't resolve to a static method"
        }

        let fn_expr_ty =
            ty::lookup_item_type(ccx.tcx, target_method_def_id).ty;

        let llselfval = GEPi(bcx, llselfval, [0, 0, i]);
        let llotherval = GEPi(bcx, llotherval, [0, 0, i]);

        // XXX: Cross-crate won't work!
        let llfn = get_item_val(ccx, target_method_def_id.node);
        let cb: &fn(block) -> Callee = |block| {
            Callee {
                bcx: block,
                data: Method(MethodData {
                    llfn: llfn,
                    llself: llselfval,
                    self_ty: struct_field_tys[i].mt.ty,
                    self_mode: ast::by_copy
                })
            }
        };

        bcx = callee::trans_call_inner(bcx,
                                       None,
                                       fn_expr_ty,
                                       ty::mk_bool(ccx.tcx),
                                       cb,
                                       ArgVals(~[llotherval]),
                                       SaveIn(fcx.llretptr),
                                       DontAutorefArg);

        // Return immediately if the call returned false.
        let next_block = sub_block(top_bcx, ~"next");
        let llcond = Load(bcx, fcx.llretptr);
        CondBr(bcx, llcond, next_block.llbb, fcx.llreturn);
        bcx = next_block;
    }

    Store(bcx, C_bool(true), fcx.llretptr);
    Br(bcx, fcx.llreturn);

    finish_fn(fcx, lltop);
}

