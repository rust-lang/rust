import libc::c_uint;
import base::*;
import common::*;
import type_of::*;
import build::*;
import driver::session::session;
import syntax::ast;
import syntax::ast_util::local_def;
import metadata::csearch;
import back::{link, abi};
import lib::llvm::llvm;
import lib::llvm::{ValueRef, TypeRef};
import lib::llvm::llvm::LLVMGetParam;
import ast_map::{path, path_mod, path_name};
import std::map::hashmap;

fn trans_impl(ccx: @crate_ctxt, path: path, name: ast::ident,
              methods: [@ast::method], tps: [ast::ty_param]) {
    if tps.len() > 0u { ret; }
    let sub_path = path + [path_name(name)];
    for m in methods {
        if m.tps.len() == 0u {
            let llfn = get_item_val(ccx, m.id);
            trans_fn(ccx, sub_path + [path_name(m.ident)], m.decl, m.body,
                     llfn, impl_self(ty::node_id_to_type(ccx.tcx, m.self_id)),
                     none, m.id, none);
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
        alt check bcx.fcx.param_substs {
          some(substs) {
            trans_monomorphized_callee(bcx, callee_id, self,
                                       iid, off, p, b, substs)
          }
        }
      }
      typeck::method_iface(_, off) {
        trans_iface_callee(bcx, self, callee_id, off)
      }
    }
}

// Method callee where the method is statically known
fn trans_static_callee(bcx: block, callee_id: ast::node_id,
                       base: @ast::expr, did: ast::def_id,
                       substs: option<([ty::t], typeck::vtable_res)>)
    -> lval_maybe_callee {
    let {bcx, val} = trans_self_arg(bcx, base);
    {env: self_env(val, node_id_type(bcx, base.id))
     with lval_static_fn(bcx, did, callee_id, substs)}
}

fn wrapper_fn_ty(ccx: @crate_ctxt, vtable_ty: TypeRef, fty: ty::t,
                 tps: @[ty::param_bounds]) -> {ty: ty::t, llty: TypeRef} {
    let bare_fn_ty = type_of_fn_from_ty(ccx, fty, *tps);
    let {inputs, output} = llfn_arg_tys(bare_fn_ty);
    {ty: fty, llty: T_fn([vtable_ty] + inputs, output)}
}

fn trans_vtable_callee(bcx: block, env: callee_env, vtable: ValueRef,
                       callee_id: ast::node_id, n_method: uint)
    -> lval_maybe_callee {
    let bcx = bcx, ccx = bcx.ccx();
    let fty = node_id_type(bcx, callee_id);
    let llfty = type_of::type_of_fn_from_ty(ccx, fty, []);
    let vtable = PointerCast(bcx, vtable,
                             T_ptr(T_array(T_ptr(llfty), n_method + 1u)));
    let mptr = Load(bcx, GEPi(bcx, vtable, [0, n_method as int]));
    {bcx: bcx, val: mptr, kind: owned,
     env: env,
     generic: generic_none}
}

fn method_with_name(ccx: @crate_ctxt, impl_id: ast::def_id,
                    name: ast::ident) -> ast::def_id {
    if impl_id.crate == ast::local_crate {
        alt check ccx.tcx.items.get(impl_id.node) {
          ast_map::node_item(@{node: ast::item_impl(_, _, _, ms), _}, _) {
            local_def(option::get(vec::find(ms, {|m| m.ident == name})).id)
          }
        }
    } else {
        csearch::get_impl_method(ccx.sess.cstore, impl_id, name)
    }
}

fn trans_monomorphized_callee(bcx: block, callee_id: ast::node_id,
                              base: @ast::expr, iface_id: ast::def_id,
                              n_method: uint, n_param: uint, n_bound: uint,
                              substs: param_substs) -> lval_maybe_callee {
    alt find_vtable_in_fn_ctxt(substs, n_param, n_bound) {
      typeck::vtable_static(impl_did, tys, sub_origins) {
        let tcx = bcx.tcx();
        let mname = ty::iface_methods(tcx, iface_id)[n_method].ident;
        let mth_id = method_with_name(bcx.ccx(), impl_did, mname);
        ret trans_static_callee(bcx, callee_id, base, mth_id,
                                some((tys, sub_origins)));
      }
      typeck::vtable_iface(iid, tps) {
        ret trans_iface_callee(bcx, base, callee_id, n_method);
      }
      typeck::vtable_param(n_param, n_bound) {
        fail "vtable_param left in monomorphized function's vtable substs";
      }
    }
}

// Method callee where the vtable comes from a boxed iface
fn trans_iface_callee(bcx: block, base: @ast::expr,
                      callee_id: ast::node_id, n_method: uint)
    -> lval_maybe_callee {
    let {bcx, val} = trans_temp_expr(bcx, base);
    let vtable = Load(bcx, PointerCast(bcx, GEPi(bcx, val, [0, 0]),
                                     T_ptr(T_ptr(T_vtable()))));
    let box = Load(bcx, GEPi(bcx, val, [0, 1]));
    // FIXME[impl] I doubt this is alignment-safe
    let self = GEPi(bcx, box, [0, abi::box_field_body]);
    trans_vtable_callee(bcx, self_env(self, expr_ty(bcx, base)), vtable,
                        callee_id, n_method)
}

fn llfn_arg_tys(ft: TypeRef) -> {inputs: [TypeRef], output: TypeRef} {
    let out_ty = llvm::LLVMGetReturnType(ft);
    let n_args = llvm::LLVMCountParamTypes(ft);
    let args = vec::from_elem(n_args as uint, 0 as TypeRef);
    unsafe { llvm::LLVMGetParamTypes(ft, vec::unsafe::to_ptr(args)); }
    {inputs: args, output: out_ty}
}

fn find_vtable_in_fn_ctxt(ps: param_substs, n_param: uint, n_bound: uint)
    -> typeck::vtable_origin {
    let vtable_off = n_bound, i = 0u;
    // Vtables are stored in a flat array, finding the right one is
    // somewhat awkward
    for bounds in *ps.bounds {
        i += 1u;
        if i >= n_param { break; }
        for bound in *bounds {
            alt bound { ty::bound_iface(_) { vtable_off += 1u; } _ {} }
        }
    }
    option::get(ps.vtables)[vtable_off]
}

fn resolve_vtables_in_fn_ctxt(fcx: fn_ctxt, vts: typeck::vtable_res)
    -> typeck::vtable_res {
    @vec::map(*vts, {|d| resolve_vtable_in_fn_ctxt(fcx, d)})
}

fn resolve_vtable_in_fn_ctxt(fcx: fn_ctxt, vt: typeck::vtable_origin)
    -> typeck::vtable_origin {
    alt vt {
      typeck::vtable_static(iid, tys, sub) {
        let tys = alt fcx.param_substs {
          some(substs) {
            vec::map(tys, {|t|
                ty::substitute_type_params(fcx.ccx.tcx, substs.tys, t)
            })
          }
          _ { tys }
        };
        typeck::vtable_static(iid, tys, resolve_vtables_in_fn_ctxt(fcx, sub))
      }
      typeck::vtable_param(n_param, n_bound) {
        alt check fcx.param_substs {
          some(substs) {
            find_vtable_in_fn_ctxt(substs, n_param, n_bound)
          }
        }
      }
      _ { vt }
    }
}

fn vtable_id(origin: typeck::vtable_origin) -> mono_id {
    alt check origin {
      typeck::vtable_static(impl_id, substs, sub_vtables) {
        @{def: impl_id, substs: substs,
          vtables: if (*sub_vtables).len() == 0u { no_vts }
                 else { some_vts(vec::map(*sub_vtables, vtable_id)) } }
      }
      typeck::vtable_iface(iface_id, substs) {
        @{def: iface_id, substs: substs, vtables: no_vts}
      }
    }
}

fn get_vtable(ccx: @crate_ctxt, origin: typeck::vtable_origin)
    -> ValueRef {
    let hash_id = vtable_id(origin);
    alt ccx.vtables.find(hash_id) {
      some(val) { val }
      none {
        alt check origin {
          typeck::vtable_static(id, substs, sub_vtables) {
            make_impl_vtable(ccx, id, substs, sub_vtables)
          }
        }
      }
    }
}

fn make_vtable(ccx: @crate_ctxt, ptrs: [ValueRef]) -> ValueRef {
    let tbl = C_struct(ptrs);
    let vt_gvar = str::as_c_str(ccx.names("vtable"), {|buf|
        llvm::LLVMAddGlobal(ccx.llmod, val_ty(tbl), buf)
    });
    llvm::LLVMSetInitializer(vt_gvar, tbl);
    llvm::LLVMSetGlobalConstant(vt_gvar, lib::llvm::True);
    lib::llvm::SetLinkage(vt_gvar, lib::llvm::InternalLinkage);
    vt_gvar
}

fn make_impl_vtable(ccx: @crate_ctxt, impl_id: ast::def_id, substs: [ty::t],
                    vtables: typeck::vtable_res) -> ValueRef {
    let tcx = ccx.tcx;
    let ifce_id = ty::ty_to_def_id(option::get(ty::impl_iface(tcx, impl_id)));
    make_vtable(ccx, vec::map(*ty::iface_methods(tcx, ifce_id), {|im|
        let fty = ty::substitute_type_params(tcx, substs,
                                             ty::mk_fn(tcx, im.fty));
        if (*im.tps).len() > 0u || ty::type_has_vars(fty) {
            C_null(type_of_fn_from_ty(ccx, fty, []))
        } else {
            let m_id = method_with_name(ccx, impl_id, im.ident);
            option::get(monomorphic_fn(ccx, m_id, substs, some(vtables))).llfn
        }
    }))
}

/*
fn make_iface_vtable(ccx: @crate_ctxt, pt: path, it: @ast::item) {
    let new_pt = pt + [path_name(it.ident), path_name(int::str(it.id))];
    let i_did = local_def(it.id), i = 0u;
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
        tcx.sess.bug("unexpected dict_param in dict_id");
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
    let gvar = str::as_c_str(name, {|buf|
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
    fn get_vtable(ccx: @crate_ctxt, did: ast::def_id) -> ValueRef {
        if did.crate == ast::local_crate {
            get_item_val(ccx, did.node)
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
        bcx.tcx().sess.bug("unexpected dict_param in get_dict_ptrs");
      }
    }
}
}*/

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
    let orig = ccx.maps.vtable_map.get(id)[0];
    let orig = resolve_vtable_in_fn_ctxt(bcx.fcx, orig);
    let vtable = get_vtable(bcx.ccx(), orig);
    Store(bcx, vtable, PointerCast(bcx, GEPi(bcx, result, [0, 0]),
                                   T_ptr(val_ty(vtable))));
    bcx
}
