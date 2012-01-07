import trans::*;
import trans_common::*;
import trans_build::*;
import option::{some, none};
import syntax::{ast, ast_util};
import metadata::csearch;
import back::{link, abi};
import lib::llvm::llvm;
import llvm::{ValueRef, TypeRef, LLVMGetParam};

// Translation functionality related to impls and ifaces
//
// Terminology:
//  vtable:  a table of function pointers pointing to method wrappers
//           of an impl that implements an iface
//  dict:    a record containing a vtable pointer along with pointers to
//           all tydescs and other dicts needed to run methods in this vtable
//           (i.e. corresponding to the type parameters of the impl)
//  wrapper: a function that takes a dict as first argument, along
//           with the method-specific tydescs for a method (and all
//           other args the method expects), which fetches the extra
//           tydescs and dicts from the dict, splices them into the
//           arglist, and calls through to the actual method
//
// Generic functions take, along with their normal arguments, a number
// of extra tydesc and dict arguments -- one tydesc for each type
// parameter, one dict (following the tydesc in the arg order) for
// each interface bound on a type parameter.
//
// Most dicts are completely static, and are allocated and filled at
// compile time. Dicts that depend on run-time values (tydescs or
// dicts for type parameter types) are built at run-time, and interned
// through upcall_intern_dict in the runtime. This means that dict
// pointers are self-contained things that do not need to be cleaned
// up.
//
// The trans_constants pass in trans.rs outputs the vtables. Typeck
// annotates notes with information about the methods and dicts that
// are referenced (ccx.method_map and ccx.dict_map).

fn trans_impl(cx: @local_ctxt, name: ast::ident, methods: [@ast::method],
              id: ast::node_id, tps: [ast::ty_param]) {
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
    rslt(bcx, PointerCast(bcx, val, T_opaque_cbox_ptr(bcx_ccx(bcx))))
}

// Method callee where the method is statically known
fn trans_static_callee(bcx: @block_ctxt, e: @ast::expr, base: @ast::expr,
                       did: ast::def_id) -> lval_maybe_callee {
    let {bcx, val} = trans_self_arg(bcx, base);
    {env: obj_env(val) with lval_static_fn(bcx, did, e.id)}
}

fn trans_vtable_callee(bcx: @block_ctxt, self: ValueRef, dict: ValueRef,
                       fld_expr: @ast::expr, iface_id: ast::def_id,
                       n_method: uint) -> lval_maybe_callee {
    let bcx = bcx, ccx = bcx_ccx(bcx), tcx = ccx.tcx;
    let method = ty::iface_methods(tcx, iface_id)[n_method];
    let fty = ty::expr_ty(tcx, fld_expr);
    let bare_fn_ty = type_of_fn_from_ty(ccx, ast_util::dummy_sp(),
                                        fty, *method.tps);
    let {inputs: bare_inputs, output} = llfn_arg_tys(bare_fn_ty);
    let fn_ty = T_fn([val_ty(dict)] + bare_inputs, output);
    let vtable = PointerCast(bcx, Load(bcx, GEPi(bcx, dict, [0, 0])),
                             T_ptr(T_array(T_ptr(fn_ty), n_method + 1u)));
    let mptr = Load(bcx, GEPi(bcx, vtable, [0, n_method as int]));
    let generic = none;
    if vec::len(*method.tps) > 0u {
        let tydescs = [], tis = [];
        let tptys = ty::node_id_to_type_params(tcx, fld_expr.id);
        for t in vec::tail_n(tptys, vec::len(tptys) - vec::len(*method.tps)) {
            let ti = none;
            let td = get_tydesc(bcx, t, true, tps_normal, ti).result;
            tis += [ti];
            tydescs += [td.val];
            bcx = td.bcx;
        }
        generic = some({item_type: fty,
                        static_tis: tis,
                        tydescs: tydescs,
                        param_bounds: method.tps,
                        origins: bcx_ccx(bcx).dict_map.find(fld_expr.id)});
    }
    {bcx: bcx, val: mptr, kind: owned,
     env: dict_env(dict, self),
     generic: generic}
}

// Method callee where the dict comes from a type param
fn trans_param_callee(bcx: @block_ctxt, fld_expr: @ast::expr,
                      base: @ast::expr, iface_id: ast::def_id, n_method: uint,
                      n_param: uint, n_bound: uint) -> lval_maybe_callee {
    let {bcx, val} = trans_self_arg(bcx, base);
    let dict = option::get(bcx.fcx.lltyparams[n_param].dicts)[n_bound];
    trans_vtable_callee(bcx, val, dict, fld_expr, iface_id, n_method)
}

// Method callee where the dict comes from a boxed iface
fn trans_iface_callee(bcx: @block_ctxt, fld_expr: @ast::expr, base: @ast::expr,
                      n_method: uint)
    -> lval_maybe_callee {
    let tcx = bcx_tcx(bcx);
    let {bcx, val} = trans_temp_expr(bcx, base);
    let box_body = GEPi(bcx, val, [0, abi::box_rc_field_body]);
    let dict = Load(bcx, PointerCast(bcx, GEPi(bcx, box_body, [0, 1]),
                                     T_ptr(T_ptr(T_dict()))));
    // FIXME[impl] I doubt this is alignment-safe
    let self = PointerCast(bcx, GEPi(bcx, box_body, [0, 2]),
                           T_opaque_cbox_ptr(bcx_ccx(bcx)));
    let iface_id = alt ty::struct(tcx, ty::expr_ty(tcx, base)) {
        ty::ty_iface(did, _) { did }
    };
    trans_vtable_callee(bcx, self, dict, fld_expr, iface_id, n_method)
}

fn llfn_arg_tys(ft: TypeRef) -> {inputs: [TypeRef], output: TypeRef} {
    let out_ty = llvm::LLVMGetReturnType(ft);
    let n_args = llvm::LLVMCountParamTypes(ft);
    let args = vec::init_elt(0 as TypeRef, n_args);
    unsafe { llvm::LLVMGetParamTypes(ft, vec::to_ptr(args)); }
    {inputs: args, output: out_ty}
}

fn trans_wrapper(ccx: @crate_ctxt, pt: [ast::ident],
                 extra_tps: [ty::param_bounds], m: @ast::method) -> ValueRef {
    let real_fn = ccx.item_ids.get(m.id);
    let {inputs: real_args, output: real_ret} =
        llfn_arg_tys(llvm::LLVMGetElementType(val_ty(real_fn)));
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
    let i = 3u, params_total = llvm::LLVMCountParamTypes(llfn_ty);
    while i < params_total {
        args += [LLVMGetParam(llfn, i)];
        i += 1u;
    }
    Call(bcx, ccx.item_ids.get(m.id), args);
    build_return(bcx);
    finish_fn(fcx, lltop);
    ret llfn;
}

fn dict_is_static(tcx: ty::ctxt, origin: typeck::dict_origin) -> bool {
    alt origin {
      typeck::dict_static(_, ts, origs) {
        vec::all(ts, {|t| !ty::type_contains_params(tcx, t)}) &&
        vec::all(*origs, {|o| dict_is_static(tcx, o)})
      }
      typeck::dict_param(_, _) { false }
    }
}

fn get_dict(bcx: @block_ctxt, origin: typeck::dict_origin) -> result {
    let ccx = bcx_ccx(bcx);
    alt origin {
      typeck::dict_static(impl_did, tys, sub_origins) {
        if dict_is_static(ccx.tcx, origin) {
            ret rslt(bcx, get_static_dict(bcx, origin));
        }
        let {bcx, ptrs} = get_dict_ptrs(bcx, origin);
        let pty = T_ptr(T_i8()), dict_ty = T_array(pty, vec::len(ptrs));
        let dict = alloca(bcx, dict_ty), i = 0;
        for ptr in ptrs {
            Store(bcx, PointerCast(bcx, ptr, pty), GEPi(bcx, dict, [0, i]));
            i += 1;
        }
        dict = Call(bcx, ccx.upcalls.intern_dict,
                    [C_uint(ccx, vec::len(ptrs)),
                     PointerCast(bcx, dict, T_ptr(T_dict()))]);
        rslt(bcx, dict)
      }
      typeck::dict_param(n_param, n_bound) {
        rslt(bcx, option::get(bcx.fcx.lltyparams[n_param].dicts)[n_bound])
      }
    }
}

fn dict_id(tcx: ty::ctxt, origin: typeck::dict_origin) -> dict_id {
    alt origin {
      typeck::dict_static(did, ts, origs) {
        let d_params = [], orig = 0u;
        if vec::len(ts) == 0u { ret @{impl_def: did, params: d_params}; }
        let impl_params = ty::lookup_item_type(tcx, did).bounds;
        vec::iter2(ts, *impl_params) {|t, bounds|
            d_params += [dict_param_ty(t)];
            for bound in *bounds {
                alt bound {
                  ty::bound_iface(_) {
                    d_params += [dict_param_dict(dict_id(tcx, origs[orig]))];
                    orig += 1u;
                  }
                }
            }
        }
        @{impl_def: did, params: d_params}
      }
    }
}

fn get_static_dict(bcx: @block_ctxt, origin: typeck::dict_origin)
    -> ValueRef {
    let ccx = bcx_ccx(bcx);
    let id = dict_id(ccx.tcx, origin);
    alt ccx.dicts.find(id) {
      some(d) { ret d; }
      none. {}
    }
    let ptrs = C_struct(get_dict_ptrs(bcx, origin).ptrs);
    let name = ccx.names.next("dict");
    let gvar = str::as_buf(name, {|buf|
        llvm::LLVMAddGlobal(ccx.llmod, val_ty(ptrs), buf)
    });
    llvm::LLVMSetGlobalConstant(gvar, lib::llvm::True);
    llvm::LLVMSetInitializer(gvar, ptrs);
    llvm::LLVMSetLinkage(gvar,
                         lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    let cast = llvm::LLVMConstPointerCast(gvar, T_ptr(T_dict()));
    ccx.dicts.insert(id, cast);
    cast
}

fn get_dict_ptrs(bcx: @block_ctxt, origin: typeck::dict_origin)
    -> {bcx: @block_ctxt, ptrs: [ValueRef]} {
    let ccx = bcx_ccx(bcx);
    alt origin {
      typeck::dict_static(impl_did, tys, sub_origins) {
        let vtable = if impl_did.crate == ast::local_crate {
            ccx.item_ids.get(impl_did.node)
        } else {
            let name = csearch::get_symbol(ccx.sess.get_cstore(), impl_did);
            get_extern_const(ccx.externs, ccx.llmod, name, T_ptr(T_i8()))
        };
        let impl_params = ty::lookup_item_type(ccx.tcx, impl_did).bounds;
        let ptrs = [vtable], origin = 0u, ti = none, bcx = bcx;
        vec::iter2(*impl_params, tys) {|param, ty|
            let rslt = get_tydesc(bcx, ty, true, tps_normal, ti).result;
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
        }
        {bcx: bcx, ptrs: ptrs}
      }
    }
}

fn trans_cast(bcx: @block_ctxt, val: @ast::expr, id: ast::node_id, dest: dest)
    -> @block_ctxt {
    let ccx = bcx_ccx(bcx), tcx = ccx.tcx;
    let val_ty = ty::expr_ty(tcx, val);
    let {bcx, val: dict} = get_dict(bcx, ccx.dict_map.get(id)[0]);
    let body_ty = ty::mk_tup(tcx, [ty::mk_type(tcx), ty::mk_type(tcx),
                                   val_ty]);
    let ti = none;
    let {bcx, val: tydesc} = get_tydesc(bcx, body_ty, true,
                                        tps_normal, ti).result;
    lazily_emit_all_tydesc_glue(bcx, ti);
    let {bcx, box, body: box_body} = trans_malloc_boxed(bcx, body_ty);
    Store(bcx, tydesc, GEPi(bcx, box_body, [0, 0]));
    Store(bcx, PointerCast(bcx, dict, T_ptr(ccx.tydesc_type)),
          GEPi(bcx, box_body, [0, 1]));
    bcx = trans_expr_save_in(bcx, val, GEPi(bcx, box_body, [0, 2]));
    store_in_dest(bcx, PointerCast(bcx, box, T_opaque_iface_ptr(ccx)), dest)
}
