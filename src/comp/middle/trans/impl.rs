import ctypes::c_uint;
import base::*;
import common::*;
import type_of::*;
import build::*;
import driver::session::session;
import syntax::{ast, ast_util};
import metadata::csearch;
import back::{link, abi};
import lib::llvm::llvm;
import lib::llvm::{ValueRef, TypeRef};
import lib::llvm::llvm::LLVMGetParam;
import ast_map::{path, path_mod, path_name};

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
// annotates nodes with information about the methods and dicts that
// are referenced (ccx.method_map and ccx.dict_map).

fn trans_impl(ccx: crate_ctxt, path: path, name: ast::ident,
              methods: [@ast::method], id: ast::node_id,
              tps: [ast::ty_param]) {
    let sub_path = path + [path_name(name)];
    for m in methods {
        alt ccx.item_ids.find(m.id) {
          some(llfn) {
            let m_bounds = param_bounds(ccx, tps + m.tps);
            trans_fn(ccx, sub_path + [path_name(m.ident)], m.decl, m.body,
                     llfn, impl_self(ty::node_id_to_type(ccx.tcx, id)),
                     m_bounds, none, m.id);
          }
          _ {
            ccx.sess.bug("Unbound id in trans_impl");
          }
        }
    }
}

fn trans_self_arg(bcx: block, base: @ast::expr) -> result {
    let basety = expr_ty(bcx, base);
    let m_by_ref = ast::expl(ast::by_ref);
    let temp_cleanups = [];
    let result = trans_arg_expr(bcx, {mode: m_by_ref, ty: basety},
                                T_ptr(type_of_or_i8(bcx.ccx(), basety)), base,
                                temp_cleanups);

    // by-ref self argument should not require cleanup in the case of
    // other arguments failing:
    assert temp_cleanups == [];

    ret result;
}

fn trans_method_callee(bcx: block, callee_id: ast::node_id,
                       self: @ast::expr, origin: typeck::method_origin)
    -> lval_maybe_callee {
    alt origin {
      typeck::method_static(did) {
        trans_static_callee(bcx, callee_id, self, did, none)
      }
      typeck::method_param(iid, off, p, b) {
        alt bcx.fcx.param_substs {
          some(substs) {
            trans_monomorphized_callee(bcx, callee_id, self,
                                       iid, off, p, b, substs)
          }
          none {
            trans_param_callee(bcx, callee_id, self, iid, off, p, b)
          }
        }
      }
      typeck::method_iface(iid, off) {
        trans_iface_callee(bcx, callee_id, self, iid, off)
      }
    }
}

// Method callee where the method is statically known
fn trans_static_callee(bcx: block, callee_id: ast::node_id,
                       base: @ast::expr, did: ast::def_id,
                       substs: option<([ty::t], typeck::dict_res)>)
    -> lval_maybe_callee {
    let {bcx, val} = trans_self_arg(bcx, base);
    {env: self_env(val, node_id_type(bcx, base.id))
     with lval_static_fn(bcx, did, callee_id, substs)}
}

fn wrapper_fn_ty(ccx: crate_ctxt, dict_ty: TypeRef, fty: ty::t,
                 tps: @[ty::param_bounds]) -> {ty: ty::t, llty: TypeRef} {
    let bare_fn_ty = type_of_fn_from_ty(ccx, fty, *tps);
    let {inputs, output} = llfn_arg_tys(bare_fn_ty);
    {ty: fty, llty: T_fn([dict_ty] + inputs, output)}
}

fn trans_vtable_callee(bcx: block, env: callee_env, dict: ValueRef,
                       callee_id: ast::node_id, iface_id: ast::def_id,
                       n_method: uint) -> lval_maybe_callee {
    let bcx = bcx, ccx = bcx.ccx(), tcx = ccx.tcx;
    let method = ty::iface_methods(tcx, iface_id)[n_method];
    let method_ty = ty::mk_fn(tcx, method.fty);
    let {ty: fty, llty: llfty} =
        wrapper_fn_ty(ccx, val_ty(dict), method_ty, method.tps);
    let vtable = PointerCast(bcx, Load(bcx, GEPi(bcx, dict, [0, 0])),
                             T_ptr(T_array(T_ptr(llfty), n_method + 1u)));
    let mptr = Load(bcx, GEPi(bcx, vtable, [0, n_method as int]));
    let generic = generic_none;
    if (*method.tps).len() > 0u || ty::type_has_params(fty) {
        let tydescs = [], tis = [];
        let tptys = node_id_type_params(bcx, callee_id);
        for t in vec::tail_n(tptys, tptys.len() - (*method.tps).len()) {
            let ti = none;
            let td = get_tydesc(bcx, t, true, ti);
            tis += [ti];
            tydescs += [td.val];
            bcx = td.bcx;
        }
        generic = generic_full({item_type: fty,
                                static_tis: tis,
                                tydescs: tydescs,
                                param_bounds: method.tps,
                                origins: ccx.maps.dict_map.find(callee_id)});
    }
    {bcx: bcx, val: mptr, kind: owned,
     env: env,
     generic: generic}
}

fn trans_monomorphized_callee(bcx: block, callee_id: ast::node_id,
                              base: @ast::expr, iface_id: ast::def_id,
                              n_method: uint, n_param: uint, n_bound: uint,
                              substs: param_substs) -> lval_maybe_callee {
    alt find_dict_in_fn_ctxt(substs, n_param, n_bound) {
      typeck::dict_static(impl_did, tys, sub_origins) {
        let tcx = bcx.tcx();
        if impl_did.crate != ast::local_crate {
            ret trans_param_callee(bcx, callee_id, base, iface_id,
                                   n_method, n_param, n_bound);
        }
        let mname = ty::iface_methods(tcx, iface_id)[n_method].ident;
        let mth = alt check tcx.items.get(impl_did.node) {
          ast_map::node_item(@{node: ast::item_impl(_, _, _, ms), _}, _) {
            option::get(vec::find(ms, {|m| m.ident == mname}))
          }
        };
        ret trans_static_callee(bcx, callee_id, base,
                                ast_util::local_def(mth.id),
                                some((tys, sub_origins)));
      }
      typeck::dict_iface(iid) {
        ret trans_iface_callee(bcx, callee_id, base, iid, n_method);
      }
      typeck::dict_param(n_param, n_bound) {
        fail "dict_param left in monomorphized function's dict substs";
      }
    }
}


// Method callee where the dict comes from a type param
fn trans_param_callee(bcx: block, callee_id: ast::node_id,
                      base: @ast::expr, iface_id: ast::def_id, n_method: uint,
                      n_param: uint, n_bound: uint) -> lval_maybe_callee {
    let {bcx, val} = trans_self_arg(bcx, base);
    let dict = option::get(bcx.fcx.lltyparams[n_param].dicts)[n_bound];
    trans_vtable_callee(bcx, dict_env(dict, val), dict,
                        callee_id, iface_id, n_method)
}

// Method callee where the dict comes from a boxed iface
fn trans_iface_callee(bcx: block, callee_id: ast::node_id,
                      base: @ast::expr, iface_id: ast::def_id, n_method: uint)
    -> lval_maybe_callee {
    let {bcx, val} = trans_temp_expr(bcx, base);
    let dict = Load(bcx, PointerCast(bcx, GEPi(bcx, val, [0, 0]),
                                     T_ptr(T_ptr(T_dict()))));
    let box = Load(bcx, GEPi(bcx, val, [0, 1]));
    // FIXME[impl] I doubt this is alignment-safe
    let self = GEPi(bcx, box, [0, abi::box_field_body]);
    trans_vtable_callee(bcx, dict_env(dict, self), dict,
                        callee_id, iface_id, n_method)
}

fn llfn_arg_tys(ft: TypeRef) -> {inputs: [TypeRef], output: TypeRef} {
    let out_ty = llvm::LLVMGetReturnType(ft);
    let n_args = llvm::LLVMCountParamTypes(ft);
    let args = vec::init_elt(n_args as uint, 0 as TypeRef);
    unsafe { llvm::LLVMGetParamTypes(ft, vec::unsafe::to_ptr(args)); }
    {inputs: args, output: out_ty}
}

fn trans_vtable(ccx: crate_ctxt, id: ast::node_id, name: str,
                ptrs: [ValueRef]) {
    let tbl = C_struct(ptrs);
    let vt_gvar = str::as_buf(name, {|buf|
        llvm::LLVMAddGlobal(ccx.llmod, val_ty(tbl), buf)
    });
    llvm::LLVMSetInitializer(vt_gvar, tbl);
    llvm::LLVMSetGlobalConstant(vt_gvar, lib::llvm::True);
    ccx.item_ids.insert(id, vt_gvar);
    ccx.item_symbols.insert(id, name);
}

fn find_dict_in_fn_ctxt(ps: param_substs, n_param: uint, n_bound: uint)
    -> typeck::dict_origin {
    let dict_off = n_bound, i = 0u;
    // Dicts are stored in a flat array, finding the right one is
    // somewhat awkward
    for bounds in *ps.bounds {
        i += 1u;
        if i >= n_param { break; }
        for bound in *bounds {
            alt bound { ty::bound_iface(_) { dict_off += 1u; } _ {} }
        }
    }
    option::get(ps.dicts)[dict_off]
}

fn resolve_dicts_in_fn_ctxt(fcx: fn_ctxt, dicts: typeck::dict_res)
    -> option<typeck::dict_res> {
    let result = [];
    for dict in *dicts {
        result += [alt dict {
          typeck::dict_static(iid, tys, sub) {
            alt resolve_dicts_in_fn_ctxt(fcx, sub) {
              some(sub) {
                let tys = alt fcx.param_substs {
                  some(substs) {
                    vec::map(tys, {|t|
                        ty::substitute_type_params(fcx.ccx.tcx, substs.tys, t)
                    })
                  }
                  _ { tys }
                };
                typeck::dict_static(iid, tys, sub)
              }
              none { ret none; }
            }
          }
          typeck::dict_param(n_param, n_bound) {
            alt fcx.param_substs {
              some(substs) {
                find_dict_in_fn_ctxt(substs, n_param, n_bound)
              }
              none { ret none; }
            }
          }
          _ { dict }
        }];
    }
    some(@result)
}

fn trans_wrapper(ccx: crate_ctxt, pt: path, llfty: TypeRef,
                 fill: fn(ValueRef, block) -> block)
    -> ValueRef {
    let name = link::mangle_internal_name_by_path(ccx, pt);
    let llfn = decl_internal_cdecl_fn(ccx.llmod, name, llfty);
    let fcx = new_fn_ctxt(ccx, [], llfn, none);
    let bcx = top_scope_block(fcx, none), lltop = bcx.llbb;
    let bcx = fill(llfn, bcx);
    build_return(bcx);
    finish_fn(fcx, lltop);
    ret llfn;
}

fn trans_impl_wrapper(ccx: crate_ctxt, pt: path,
                      extra_tps: [ty::param_bounds], real_fn: ValueRef)
    -> ValueRef {
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
    let n_extra_ptrs = extra_ptrs.len();

    let wrap_args = [T_ptr(T_dict())] +
        vec::slice(real_args, 0u, first_tp_arg) +
        vec::slice(real_args, first_tp_arg + n_extra_ptrs, real_args.len());
    let llfn_ty = T_fn(wrap_args, real_ret);
    trans_wrapper(ccx, pt, llfn_ty, {|llfn, bcx|
        let dict = PointerCast(bcx, LLVMGetParam(llfn, 0 as c_uint), env_ty);
        // retptr, self
        let args = [LLVMGetParam(llfn, 1 as c_uint),
                    LLVMGetParam(llfn, 2 as c_uint)];
        let i = 0u;
        // saved tydescs/dicts
        while i < n_extra_ptrs {
            i += 1u;
            args += [load_inbounds(bcx, dict, [0, i as int])];
        }
        // the rest of the parameters
        let j = 3u as c_uint;
        let params_total = llvm::LLVMCountParamTypes(llfn_ty);
        while j < params_total {
            args += [LLVMGetParam(llfn, j)];
            j += 1u as c_uint;
        }
        Call(bcx, real_fn, args);
        bcx
    })
}

fn trans_impl_vtable(ccx: crate_ctxt, pt: path,
                     iface_id: ast::def_id, ms: [@ast::method],
                     tps: [ast::ty_param], it: @ast::item) {
    let new_pt = pt + [path_name(it.ident), path_name(int::str(it.id)),
                       path_name("wrap")];
    let extra_tps = param_bounds(ccx, tps);
    let ptrs = vec::map(*ty::iface_methods(ccx.tcx, iface_id), {|im|
        alt vec::find(ms, {|m| m.ident == im.ident}) {
          some(m) {
            let target = ccx.item_ids.get(m.id);
            trans_impl_wrapper(ccx, new_pt + [path_name(m.ident)],
                               extra_tps, target)
          }
          _ {
            ccx.sess.span_bug(it.span, "No matching method \
               in trans_impl_vtable");
          }
        }
    });
    let s = link::mangle_internal_name_by_path(
        ccx, new_pt + [path_name("!vtable")]);
    trans_vtable(ccx, it.id, s, ptrs);
}

fn trans_iface_wrapper(ccx: crate_ctxt, pt: path, m: ty::method,
                       n: uint) -> ValueRef {
    let {llty: llfty, _} = wrapper_fn_ty(ccx, T_ptr(T_i8()),
                                         ty::mk_fn(ccx.tcx, m.fty), m.tps);
    trans_wrapper(ccx, pt, llfty, {|llfn, bcx|
        let param = PointerCast(bcx, LLVMGetParam(llfn, 2u as c_uint),
                                T_ptr(T_opaque_iface(ccx)));
        let dict = Load(bcx, GEPi(bcx, param, [0, 0]));
        let box = Load(bcx, GEPi(bcx, param, [0, 1]));
        let self = GEPi(bcx, box, [0, abi::box_field_body]);
        let vtable = PointerCast(bcx, Load(bcx, GEPi(bcx, dict, [0, 0])),
                                 T_ptr(T_array(T_ptr(llfty), n + 1u)));
        let mptr = Load(bcx, GEPi(bcx, vtable, [0, n as int]));
        let args = [PointerCast(bcx, dict, T_ptr(T_i8())),
                    LLVMGetParam(llfn, 1u as c_uint),
                    PointerCast(bcx, self, T_opaque_cbox_ptr(ccx))];
        let i = 3u as c_uint, total = llvm::LLVMCountParamTypes(llfty);
        while i < total {
            args += [LLVMGetParam(llfn, i)];
            i += 1u as c_uint;
        }
        Call(bcx, mptr, args);
        bcx
    })
}

fn trans_iface_vtable(ccx: crate_ctxt, pt: path, it: @ast::item) {
    let new_pt = pt + [path_name(it.ident), path_name(int::str(it.id))];
    let i_did = ast_util::local_def(it.id), i = 0u;
    let ptrs = vec::map(*ty::iface_methods(ccx.tcx, i_did), {|m|
        let w = trans_iface_wrapper(ccx, new_pt + [path_name(m.ident)], m, i);
        i += 1u;
        w
    });
    let s = link::mangle_internal_name_by_path(
        ccx, new_pt + [path_name("!vtable")]);
    trans_vtable(ccx, it.id, s, ptrs);
}

fn dict_is_static(tcx: ty::ctxt, origin: typeck::dict_origin) -> bool {
    alt origin {
      typeck::dict_static(_, ts, origs) {
        vec::all(ts, {|t| !ty::type_has_params(t)}) &&
        vec::all(*origs, {|o| dict_is_static(tcx, o)})
      }
      typeck::dict_iface(_) { true }
      _ { false }
    }
}

fn get_dict(bcx: block, origin: typeck::dict_origin) -> result {
    let ccx = bcx.ccx();
    alt origin {
      typeck::dict_static(impl_did, tys, sub_origins) {
        if dict_is_static(ccx.tcx, origin) {
            ret rslt(bcx, get_static_dict(bcx, origin));
        }
        let {bcx, ptrs} = get_dict_ptrs(bcx, origin);
        let pty = T_ptr(T_i8()), dict_ty = T_array(pty, ptrs.len());
        let dict = alloca(bcx, dict_ty), i = 0;
        for ptr in ptrs {
            Store(bcx, PointerCast(bcx, ptr, pty), GEPi(bcx, dict, [0, i]));
            i += 1;
        }
        dict = Call(bcx, ccx.upcalls.intern_dict,
                    [C_uint(ccx, ptrs.len()),
                     PointerCast(bcx, dict, T_ptr(T_dict()))]);
        rslt(bcx, dict)
      }
      typeck::dict_param(n_param, n_bound) {
        rslt(bcx, option::get(bcx.fcx.lltyparams[n_param].dicts)[n_bound])
      }
      typeck::dict_iface(did) {
        ret rslt(bcx, get_static_dict(bcx, origin));
      }
    }
}

fn dict_id(tcx: ty::ctxt, origin: typeck::dict_origin) -> dict_id {
    alt origin {
      typeck::dict_static(did, ts, origs) {
        let d_params = [], orig = 0u;
        if ts.len() == 0u { ret @{def: did, params: d_params}; }
        let impl_params = ty::lookup_item_type(tcx, did).bounds;
        vec::iter2(ts, *impl_params) {|t, bounds|
            d_params += [dict_param_ty(t)];
            for bound in *bounds {
                alt bound {
                  ty::bound_iface(_) {
                    d_params += [dict_param_dict(dict_id(tcx, origs[orig]))];
                    orig += 1u;
                  }
                  _ {}
                }
            }
        }
        @{def: did, params: d_params}
      }
      typeck::dict_iface(did) {
        @{def: did, params: []}
      }
      _ {
        tcx.sess.bug("Unexpected dict_param in dict_id");
      }
    }
}

fn get_static_dict(bcx: block, origin: typeck::dict_origin)
    -> ValueRef {
    let ccx = bcx.ccx();
    let id = dict_id(ccx.tcx, origin);
    alt ccx.dicts.find(id) {
      some(d) { ret d; }
      none {}
    }
    let ptrs = C_struct(get_dict_ptrs(bcx, origin).ptrs);
    let name = ccx.names("dict");
    let gvar = str::as_buf(name, {|buf|
        llvm::LLVMAddGlobal(ccx.llmod, val_ty(ptrs), buf)
    });
    llvm::LLVMSetGlobalConstant(gvar, lib::llvm::True);
    llvm::LLVMSetInitializer(gvar, ptrs);
    lib::llvm::SetLinkage(gvar, lib::llvm::InternalLinkage);
    let cast = llvm::LLVMConstPointerCast(gvar, T_ptr(T_dict()));
    ccx.dicts.insert(id, cast);
    cast
}

fn get_dict_ptrs(bcx: block, origin: typeck::dict_origin)
    -> {bcx: block, ptrs: [ValueRef]} {
    let ccx = bcx.ccx();
    fn get_vtable(ccx: crate_ctxt, did: ast::def_id) -> ValueRef {
        if did.crate == ast::local_crate {
            ccx.item_ids.get(did.node)
        } else {
            let name = csearch::get_symbol(ccx.sess.cstore, did);
            get_extern_const(ccx.externs, ccx.llmod, name, T_ptr(T_i8()))
        }
    }
    alt origin {
      typeck::dict_static(impl_did, tys, sub_origins) {
        let impl_params = ty::lookup_item_type(ccx.tcx, impl_did).bounds;
        let ptrs = [get_vtable(ccx, impl_did)];
        let origin = 0u, bcx = bcx;
        vec::iter2(*impl_params, tys) {|param, ty|
            let rslt = get_tydesc_simple(bcx, ty, true);
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
      typeck::dict_iface(did) {
        {bcx: bcx, ptrs: [get_vtable(ccx, did)]}
      }
      _ {
        bcx.tcx().sess.bug("Unexpected dict_param in get_dict_ptrs");
      }
    }
}

fn trans_cast(bcx: block, val: @ast::expr, id: ast::node_id, dest: dest)
    -> block {
    if dest == ignore { ret trans_expr(bcx, val, ignore); }
    let ccx = bcx.ccx();
    let v_ty = expr_ty(bcx, val);
    let {bcx, box, body} = trans_malloc_boxed(bcx, v_ty);
    add_clean_free(bcx, box, false);
    bcx = trans_expr_save_in(bcx, val, body);
    revoke_clean(bcx, box);
    let result = get_dest_addr(dest);
    Store(bcx, box, PointerCast(bcx, GEPi(bcx, result, [0, 1]),
                                T_ptr(val_ty(box))));
    let {bcx, val: dict} = get_dict(bcx, ccx.maps.dict_map.get(id)[0]);
    Store(bcx, dict, PointerCast(bcx, GEPi(bcx, result, [0, 0]),
                                 T_ptr(val_ty(dict))));
    bcx
}
