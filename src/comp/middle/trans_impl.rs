import trans::*;
import trans_common::*;
import trans_build::*;
import option::{some, none};
import syntax::{ast, ast_util};
import back::link;
import lib::llvm;
import llvm::llvm::{ValueRef, TypeRef, LLVMGetParam};

fn trans_impl(cx: @local_ctxt, name: ast::ident, methods: [@ast::method],
              id: ast::node_id, tps: [ast::ty_param],
              _ifce: option::t<@ast::ty>) {
    let sub_cx = extend_path(cx, name);
    for m in methods {
        alt cx.ccx.item_ids.find(m.id) {
          some(llfn) {
            trans_fn(extend_path(sub_cx, m.ident), m.span, m.decl, m.body,
                     llfn, impl_self(ty::node_id_to_monotype(cx.ccx.tcx, id)),
                     tps + m.tps, m.id);
          }
        }
    }
}

fn trans_self_arg(bcx: @block_ctxt, base: @ast::expr) -> result {
    let tz = [], tr = [];
    let basety = ty::expr_ty(bcx_tcx(bcx), base);
    let {bcx, val} = trans_arg_expr(bcx, {mode: ast::by_ref, ty: basety},
                                    T_ptr(type_of_or_i8(bcx, basety)), tz,
                                    tr, base);
    rslt(bcx, PointerCast(bcx, val, T_opaque_boxed_closure_ptr(bcx_ccx(bcx))))
}

fn trans_static_callee(bcx: @block_ctxt, e: @ast::expr, base: @ast::expr,
                       did: ast::def_id) -> lval_maybe_callee {
    let {bcx, val} = trans_self_arg(bcx, base);
    {env: obj_env(val) with lval_static_fn(bcx, did, e.id)}
}

fn trans_dict_callee(bcx: @block_ctxt, _e: @ast::expr, base: @ast::expr,
                     iface_id: ast::def_id, n_method: uint,
                     n_param: uint, n_bound: uint) -> lval_maybe_callee {
    let {bcx, val} = trans_self_arg(bcx, base);
    let dict = option::get(bcx.fcx.lltyparams[n_param].dicts)[n_bound];
    let method = ty::iface_methods(bcx_tcx(bcx), iface_id)[n_method];
    let bare_fn_ty = type_of_fn(bcx_ccx(bcx), ast_util::dummy_sp(),
                                false, method.fty.inputs, method.fty.output,
                                *method.tps);
    let {inputs: bare_inputs, output} = llfn_arg_tys(bare_fn_ty);
    let fn_ty = T_fn([val_ty(dict)] + bare_inputs, output);
    let vtable = PointerCast(bcx, Load(bcx, GEPi(bcx, dict, [0, 0])),
                             T_ptr(T_array(T_ptr(fn_ty), n_method + 1u)));
    let mptr = Load(bcx, GEPi(bcx, vtable, [0, n_method as int]));
    {bcx: bcx, val: mptr, kind: owned,
     env: dict_env(dict, val),
     generic: none} // FIXME[impl] fetch generic info for method
}

fn llfn_arg_tys(ft: TypeRef) -> {inputs: [TypeRef], output: TypeRef} {
    let out_ty = llvm::llvm::LLVMGetReturnType(ft);
    let n_args = llvm::llvm::LLVMCountParamTypes(ft);
    let args = vec::init_elt(0 as TypeRef, n_args);
    unsafe { llvm::llvm::LLVMGetParamTypes(ft, vec::to_ptr(args)); }
    {inputs: args, output: out_ty}
}

fn trans_wrapper(ccx: @crate_ctxt, pt: [ast::ident],
                 extra_tps: [ty::param_bounds], m: @ast::method) -> ValueRef {
    let real_fn = ccx.item_ids.get(m.id);
    let {inputs: real_args, output: real_ret} =
        llfn_arg_tys(llvm::llvm::LLVMGetElementType(val_ty(real_fn)));
    let extra_ptrs = [];
    for tp in extra_tps {
        extra_ptrs += [T_ptr(ccx.tydesc_type)];
        for bound in *tp {
            alt bound {
              ty::bound_iface(_) { extra_ptrs += [T_ptr(T_dict())]; }
              _ {}
            }
        }
    }
    let env_ty = T_ptr(T_struct([T_ptr(T_i8())] + extra_ptrs));
    let n_extra_ptrs = vec::len(extra_ptrs);

    let wrap_args = [T_ptr(T_dict())] + vec::slice(real_args, 0u, 2u) +
        vec::slice(real_args, 2u + vec::len(extra_ptrs), vec::len(real_args));
    let llfn_ty = T_fn(wrap_args, real_ret);

    let lcx = @{path: pt + ["wrapper", m.ident], module_path: [],
                obj_typarams: [], obj_fields: [], ccx: ccx};
    let name = link::mangle_internal_name_by_path_and_seq(ccx, pt, m.ident);
    let llfn = decl_internal_cdecl_fn(ccx.llmod, name, llfn_ty);
    let fcx = new_fn_ctxt(lcx, ast_util::dummy_sp(), llfn);
    let bcx = new_top_block_ctxt(fcx), lltop = bcx.llbb;

    let dict = PointerCast(bcx, LLVMGetParam(llfn, 0u), env_ty);
    // retptr, self
    let args = [LLVMGetParam(llfn, 1u), LLVMGetParam(llfn, 2u)], i = 0u;
    // saved tydescs/dicts
    while i < n_extra_ptrs {
        i += 1u;
        args += [load_inbounds(bcx, dict, [0, i as int])];
    }
    // the rest of the parameters
    let i = 3u, params_total = llvm::llvm::LLVMCountParamTypes(llfn_ty);
    while i < params_total {
        args += [LLVMGetParam(llfn, i)];
        i += 1u;
    }
    Call(bcx, ccx.item_ids.get(m.id), args);
    build_return(bcx);
    finish_fn(fcx, lltop);
    ret llfn;
}

// FIXME[impl] cache these on the function level somehow
fn get_dict(bcx: @block_ctxt, origin: typeck::dict_origin) -> result {
    let bcx = bcx, ccx = bcx_ccx(bcx);
    alt origin {
      typeck::dict_static(impl_did, tys, sub_origins) {
        assert impl_did.crate == ast::local_crate; // FIXME[impl]
        let vtable = ccx.item_ids.get(impl_did.node);
        let impl_params = ty::lookup_item_type(ccx.tcx, impl_did).bounds;
        let ptrs = [vtable], i = 0u, origin = 0u, ti = none;
        for param in *impl_params {
            let rslt = get_tydesc(bcx, tys[i], false, tps_normal, ti).result;
            ptrs += [rslt.val];
            bcx = rslt.bcx;
            for bound in *param {
                alt bound {
                  ty::bound_iface(_) {
                    let res = get_dict(bcx, sub_origins[origin]);
                    ptrs += [res.val];
                    bcx = res.bcx;
                    origin += 1u;
                  }
                  _ {}
                }
            }
            i += 1u;
        }
        let pty = T_ptr(T_i8()), dict_ty = T_array(pty, vec::len(ptrs));
        let dict = alloca(bcx, dict_ty), i = 0;
        for ptr in ptrs {
            Store(bcx, PointerCast(bcx, ptr, pty), GEPi(bcx, dict, [0, i]));
            i += 1;
        }
        rslt(bcx, PointerCast(bcx, dict, T_ptr(T_dict())))
      }
      typeck::dict_param(_param) { fail "FIXME[impl]"; }
    }
}