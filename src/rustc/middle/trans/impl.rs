import libc::c_uint;
import base::*;
import common::*;
import type_of::*;
import build::*;
import driver::session::session;
import syntax::{ast, ast_map};
import ast_map::{path, path_mod, path_name, node_id_to_str};
import driver::session::expect;
import syntax::ast_util::{local_def, split_class_items};
import metadata::csearch;
import back::{link, abi};
import lib::llvm::llvm;
import lib::llvm::{ValueRef, TypeRef};
import lib::llvm::llvm::LLVMGetParam;
import std::map::hashmap;

fn trans_impl(ccx: @crate_ctxt, path: path, name: ast::ident,
              methods: ~[@ast::method], tps: ~[ast::ty_param]) {
    let _icx = ccx.insn_ctxt("impl::trans_impl");
    if tps.len() > 0u { ret; }
    let sub_path = vec::append_one(path, path_name(name));
    for vec::each(methods) {|m|
        if m.tps.len() == 0u {
            let llfn = get_item_val(ccx, m.id);
            trans_fn(ccx,
                     vec::append_one(sub_path, path_name(m.ident)),
                     m.decl, m.body,
                     llfn, impl_self(ty::node_id_to_type(ccx.tcx, m.self_id)),
                     none, m.id);
        }
    }
}

fn trans_self_arg(bcx: block, base: @ast::expr, derefs: uint) -> result {
    let _icx = bcx.insn_ctxt("impl::trans_self_arg");
    let basety = expr_ty(bcx, base);
    let m_by_ref = ast::expl(ast::by_ref);
    let mut temp_cleanups = ~[];
    let result = trans_arg_expr(bcx, {mode: m_by_ref, ty: basety},
                                T_ptr(type_of::type_of(bcx.ccx(), basety)),
                                base, temp_cleanups, none, derefs);

    // by-ref self argument should not require cleanup in the case of
    // other arguments failing:
    assert temp_cleanups == ~[];

    ret result;
}

fn trans_method_callee(bcx: block, callee_id: ast::node_id,
                       self: @ast::expr, mentry: typeck::method_map_entry)
    -> lval_maybe_callee {
    let _icx = bcx.insn_ctxt("impl::trans_method_callee");
    alt mentry.origin {
      typeck::method_static(did) {
        let {bcx, val} = trans_self_arg(bcx, self, mentry.derefs);
        {env: self_env(val, node_id_type(bcx, self.id), none)
         with lval_static_fn(bcx, did, callee_id)}
      }
      typeck::method_param({iface_id:iid, method_num:off,
                            param_num:p, bound_num:b}) {
        alt check bcx.fcx.param_substs {
          some(substs) {
            trans_monomorphized_callee(bcx, callee_id, self, mentry.derefs,
                                       iid, off, p, b, substs)
          }
        }
      }
      typeck::method_iface(_, off) {
        let {bcx, val} = trans_temp_expr(bcx, self);
        let fty = node_id_type(bcx, callee_id);
        trans_iface_callee(bcx, val, fty, off)
      }
    }
}

fn method_from_methods(ms: ~[@ast::method], name: ast::ident)
    -> ast::def_id {
  local_def(option::get(vec::find(ms, {|m| m.ident == name})).id)
}

fn method_with_name(ccx: @crate_ctxt, impl_id: ast::def_id,
                    name: ast::ident) -> ast::def_id {
    if impl_id.crate == ast::local_crate {
        alt check ccx.tcx.items.get(impl_id.node) {
          ast_map::node_item(@{node: ast::item_impl(_, _, _, _, ms), _}, _) {
            method_from_methods(ms, name)
          }
          ast_map::node_item(@{node:
              ast::item_class(_, _, items, _, _, _), _}, _) {
            let (_,ms) = split_class_items(items);
            method_from_methods(ms, name)
          }
        }
    } else {
        csearch::get_impl_method(ccx.sess.cstore, impl_id, name)
    }
}

fn method_ty_param_count(ccx: @crate_ctxt, m_id: ast::def_id,
                         i_id: ast::def_id) -> uint {
    if m_id.crate == ast::local_crate {
        alt check ccx.tcx.items.get(m_id.node) {
          ast_map::node_method(m, _, _) { vec::len(m.tps) }
        }
    } else {
        csearch::get_type_param_count(ccx.sess.cstore, m_id) -
            csearch::get_type_param_count(ccx.sess.cstore, i_id)
    }
}

fn trans_monomorphized_callee(bcx: block, callee_id: ast::node_id,
                              base: @ast::expr, derefs: uint,
                              iface_id: ast::def_id, n_method: uint,
                              n_param: uint, n_bound: uint,
                              substs: param_substs) -> lval_maybe_callee {
    let _icx = bcx.insn_ctxt("impl::trans_monomorphized_callee");
    alt find_vtable_in_fn_ctxt(substs, n_param, n_bound) {
      typeck::vtable_static(impl_did, impl_substs, sub_origins) {
        let ccx = bcx.ccx();
        let mname = ty::iface_methods(ccx.tcx, iface_id)[n_method].ident;
        let mth_id = method_with_name(bcx.ccx(), impl_did, mname);
        let n_m_tps = method_ty_param_count(ccx, mth_id, impl_did);
        let node_substs = node_id_type_params(bcx, callee_id);
        let ty_substs
            = vec::append(impl_substs,
                          vec::tailn(node_substs,
                                     node_substs.len() - n_m_tps));
        let {bcx, val} = trans_self_arg(bcx, base, derefs);
        let lval = lval_static_fn_inner(bcx, mth_id, callee_id, ty_substs,
                                        some(sub_origins));
        {env: self_env(val, node_id_type(bcx, base.id), none),
         val: PointerCast(bcx, lval.val, T_ptr(type_of_fn_from_ty(
             ccx, node_id_type(bcx, callee_id))))
         with lval}
      }
      typeck::vtable_iface(iid, tps) {
        let {bcx, val} = trans_temp_expr(bcx, base);
        let fty = node_id_type(bcx, callee_id);
        trans_iface_callee(bcx, val, fty, n_method)
      }
      typeck::vtable_param(n_param, n_bound) {
        fail "vtable_param left in monomorphized function's vtable substs";
      }
    }
}

// Method callee where the vtable comes from a boxed iface
fn trans_iface_callee(bcx: block, val: ValueRef,
                      callee_ty: ty::t, n_method: uint)
    -> lval_maybe_callee {
    let _icx = bcx.insn_ctxt("impl::trans_iface_callee");
    let ccx = bcx.ccx();
    let vtable = Load(bcx, PointerCast(bcx, GEPi(bcx, val, ~[0u, 0u]),
                                       T_ptr(T_ptr(T_vtable()))));
    let llbox = Load(bcx, GEPi(bcx, val, ~[0u, 1u]));
    // FIXME[impl] I doubt this is alignment-safe (#2534)
    let self = GEPi(bcx, llbox, ~[0u, abi::box_field_body]);
    let env = self_env(self, ty::mk_opaque_box(bcx.tcx()), some(llbox));
    let llfty = type_of::type_of_fn_from_ty(ccx, callee_ty);
    let vtable = PointerCast(bcx, vtable,
                             T_ptr(T_array(T_ptr(llfty), n_method + 1u)));
    let mptr = Load(bcx, GEPi(bcx, vtable, ~[0u, n_method]));
    {bcx: bcx, val: mptr, kind: owned, env: env}
}

fn find_vtable_in_fn_ctxt(ps: param_substs, n_param: uint, n_bound: uint)
    -> typeck::vtable_origin {
    let mut vtable_off = n_bound, i = 0u;
    // Vtables are stored in a flat array, finding the right one is
    // somewhat awkward
    for vec::each(*ps.bounds) {|bounds|
        if i >= n_param { break; }
        for vec::each(*bounds) {|bound|
            alt bound { ty::bound_iface(_) { vtable_off += 1u; } _ {} }
        }
        i += 1u;
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
                ty::subst_tps(fcx.ccx.tcx, substs.tys, t)
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

fn vtable_id(ccx: @crate_ctxt, origin: typeck::vtable_origin) -> mono_id {
    alt check origin {
      typeck::vtable_static(impl_id, substs, sub_vtables) {
        make_mono_id(ccx, impl_id, substs,
                     if (*sub_vtables).len() == 0u { none }
                     else { some(sub_vtables) }, none)
      }
      typeck::vtable_iface(iface_id, substs) {
        @{def: iface_id,
          params: vec::map(substs, {|t| mono_precise(t, none)})}
      }
    }
}

fn get_vtable(ccx: @crate_ctxt, origin: typeck::vtable_origin)
    -> ValueRef {
    let hash_id = vtable_id(ccx, origin);
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

fn make_vtable(ccx: @crate_ctxt, ptrs: ~[ValueRef]) -> ValueRef {
    let _icx = ccx.insn_ctxt("impl::make_vtable");
    let tbl = C_struct(ptrs);
    let vt_gvar = str::as_c_str(ccx.names("vtable"), {|buf|
        llvm::LLVMAddGlobal(ccx.llmod, val_ty(tbl), buf)
    });
    llvm::LLVMSetInitializer(vt_gvar, tbl);
    llvm::LLVMSetGlobalConstant(vt_gvar, lib::llvm::True);
    lib::llvm::SetLinkage(vt_gvar, lib::llvm::InternalLinkage);
    vt_gvar
}

fn make_impl_vtable(ccx: @crate_ctxt, impl_id: ast::def_id, substs: ~[ty::t],
                    vtables: typeck::vtable_res) -> ValueRef {
    let _icx = ccx.insn_ctxt("impl::make_impl_vtable");
    let tcx = ccx.tcx;
    let ifce_id = expect(ccx.sess,
                         ty::ty_to_def_id(option::get(ty::impl_iface(tcx,
                                                             impl_id))),
                         {|| "make_impl_vtable: non-iface-type implemented"});
    let has_tps = (*ty::lookup_item_type(ccx.tcx, impl_id).bounds).len() > 0u;
    make_vtable(ccx, vec::map(*ty::iface_methods(tcx, ifce_id)) {|im|
        let fty = ty::subst_tps(tcx, substs, ty::mk_fn(tcx, im.fty));
        if (*im.tps).len() > 0u || ty::type_has_self(fty) {
            C_null(T_ptr(T_nil()))
        } else {
            let mut m_id = method_with_name(ccx, impl_id, im.ident);
            if has_tps {
                // If the method is in another crate, need to make an inlined
                // copy first
                if m_id.crate != ast::local_crate {
                    m_id = maybe_instantiate_inline(ccx, m_id);
                }
                monomorphic_fn(ccx, m_id, substs, some(vtables), none).val
            } else if m_id.crate == ast::local_crate {
                get_item_val(ccx, m_id.node)
            } else {
                trans_external_path(ccx, m_id, fty)
            }
        }
    })
}

fn trans_cast(bcx: block, val: @ast::expr, id: ast::node_id, dest: dest)
    -> block {
    let _icx = bcx.insn_ctxt("impl::trans_cast");
    if dest == ignore { ret trans_expr(bcx, val, ignore); }
    let ccx = bcx.ccx();
    let v_ty = expr_ty(bcx, val);
    let {box: llbox, body: body} = malloc_boxed(bcx, v_ty);
    add_clean_free(bcx, llbox, heap_shared);
    let bcx = trans_expr_save_in(bcx, val, body);
    revoke_clean(bcx, llbox);
    let result = get_dest_addr(dest);
    Store(bcx, llbox, PointerCast(bcx, GEPi(bcx, result, ~[0u, 1u]),
                                  T_ptr(val_ty(llbox))));
    let orig = ccx.maps.vtable_map.get(id)[0];
    let orig = resolve_vtable_in_fn_ctxt(bcx.fcx, orig);
    let vtable = get_vtable(bcx.ccx(), orig);
    Store(bcx, vtable, PointerCast(bcx, GEPi(bcx, result, ~[0u, 0u]),
                                   T_ptr(val_ty(vtable))));
    bcx
}
