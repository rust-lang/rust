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
    let env_ty = T_ptr(T_struct([T_ptr(T_i8())] +
                                vec::map(extra_tps,
                                         {|_p| T_ptr(ccx.tydesc_type)})));
    // FIXME[impl] filter and pass along dicts for bounds
    let wrap_args = [env_ty] + vec::slice(real_args, 0u, 2u) +
        vec::slice(real_args, 2u + vec::len(extra_tps), vec::len(real_args));
    let llfn_ty = T_fn(wrap_args, real_ret);

    let lcx = @{path: pt + ["wrapper", m.ident], module_path: [],
                obj_typarams: [], obj_fields: [], ccx: ccx};
    let name = link::mangle_internal_name_by_path_and_seq(ccx, pt, m.ident);
    let llfn = decl_internal_cdecl_fn(ccx.llmod, name, llfn_ty);
    let fcx = new_fn_ctxt(lcx, ast_util::dummy_sp(), llfn);
    let bcx = new_top_block_ctxt(fcx), lltop = bcx.llbb;

    let dict = LLVMGetParam(llfn, 0u);
    // retptr, self
    let args = [LLVMGetParam(llfn, 1u), LLVMGetParam(llfn, 2u)], i = 1;
    // saved tydescs/dicts
    for extra_tp in extra_tps {
        args += [load_inbounds(bcx, dict, [0, i])];
        i += 1;
    }
    // the rest of the parameters
    let i = 3u, params_total = llvm::llvm::LLVMCountParamTypes(llfn_ty);
    while i < params_total {
        args += [LLVMGetParam(llfn, i)];
        i += 1u;
    }
    Call(bcx, ccx.item_ids.get(m.id), args);
    finish_fn(fcx, lltop);
    ret llfn;
}

