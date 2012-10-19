use common::*;
use syntax::ast;
use syntax::ast_util::local_def;
use syntax::ast_map::{path, path_mod, path_name};
use base::{trans_item, get_item_val, no_self, self_arg, trans_fn,
              impl_self, decl_internal_cdecl_fn,
              set_inline_hint_if_appr, set_inline_hint,
              trans_enum_variant, trans_class_dtor,
              get_insn_ctxt};
use syntax::parse::token::special_idents;
use type_of::type_of_fn_from_ty;
use back::link::mangle_exported_name;
use middle::ty::{FnTyBase, FnMeta, FnSig};

fn monomorphic_fn(ccx: @crate_ctxt,
                  fn_id: ast::def_id,
                  real_substs: ~[ty::t],
                  vtables: Option<typeck::vtable_res>,
                  impl_did_opt: Option<ast::def_id>,
                  ref_id: Option<ast::node_id>) ->
                  {val: ValueRef, must_cast: bool} {
    let _icx = ccx.insn_ctxt("monomorphic_fn");
    let mut must_cast = false;
    let substs = vec::map(real_substs, |t| {
        match normalize_for_monomorphization(ccx.tcx, *t) {
          Some(t) => { must_cast = true; t }
          None => *t
        }
    });

    for real_substs.each() |s| { assert !ty::type_has_params(*s); }
    for substs.each() |s| { assert !ty::type_has_params(*s); }
    let param_uses = type_use::type_uses_for(ccx, fn_id, substs.len());
    let hash_id = make_mono_id(ccx, fn_id, substs, vtables, impl_did_opt,
                               Some(param_uses));
    if vec::any(hash_id.params,
                |p| match *p { mono_precise(_, _) => false, _ => true }) {
        must_cast = true;
    }

    debug!("monomorphic_fn(fn_id=%? (%s), real_substs=%?, substs=%?, \
           hash_id = %?",
           fn_id, ty::item_path_str(ccx.tcx, fn_id),
           real_substs.map(|s| ty_to_str(ccx.tcx, *s)),
           substs.map(|s| ty_to_str(ccx.tcx, *s)), hash_id);

    match ccx.monomorphized.find(hash_id) {
      Some(val) => {
        debug!("leaving monomorphic fn %s",
               ty::item_path_str(ccx.tcx, fn_id));
        return {val: val, must_cast: must_cast};
      }
      None => ()
    }

    let tpt = ty::lookup_item_type(ccx.tcx, fn_id);
    let mut llitem_ty = tpt.ty;

    let map_node = session::expect(ccx.sess, ccx.tcx.items.find(fn_id.node),
     || fmt!("While monomorphizing %?, couldn't find it in the item map \
        (may have attempted to monomorphize an item defined in a different \
        crate?)", fn_id));
    // Get the path so that we can create a symbol
    let (pt, name, span) = match map_node {
      ast_map::node_item(i, pt) => (pt, i.ident, i.span),
      ast_map::node_variant(v, enm, pt) => (pt, v.node.name, enm.span),
      ast_map::node_method(m, _, pt) => (pt, m.ident, m.span),
      ast_map::node_foreign_item(i, ast::foreign_abi_rust_intrinsic, pt)
      => (pt, i.ident, i.span),
      ast_map::node_foreign_item(*) => {
        // Foreign externs don't have to be monomorphized.
        return {val: get_item_val(ccx, fn_id.node),
                must_cast: true};
      }
      ast_map::node_dtor(_, dtor, _, pt) =>
          (pt, special_idents::dtor, dtor.span),
      ast_map::node_trait_method(@ast::provided(m), _, pt) => {
        (pt, m.ident, m.span)
      }
      ast_map::node_trait_method(@ast::required(_), _, _) => {
        ccx.tcx.sess.bug(~"Can't monomorphize a required trait method")
      }
      ast_map::node_expr(*) => {
        ccx.tcx.sess.bug(~"Can't monomorphize an expr")
      }
      ast_map::node_stmt(*) => {
        ccx.tcx.sess.bug(~"Can't monomorphize a stmt")
      }
      ast_map::node_export(*) => {
          ccx.tcx.sess.bug(~"Can't monomorphize an export")
      }
      ast_map::node_arg(*) => ccx.tcx.sess.bug(~"Can't monomorphize an arg"),
      ast_map::node_block(*) => {
          ccx.tcx.sess.bug(~"Can't monomorphize a block")
      }
      ast_map::node_local(*) => {
          ccx.tcx.sess.bug(~"Can't monomorphize a local")
      }
    };

    // Look up the impl type if we're translating a default method.
    // XXX: Generics.
    let impl_ty_opt;
    match impl_did_opt {
        None => impl_ty_opt = None,
        Some(impl_did) => {
            impl_ty_opt = Some(ty::lookup_item_type(ccx.tcx, impl_did).ty);
        }
    }

    let mono_ty = ty::subst_tps(ccx.tcx, substs, impl_ty_opt, llitem_ty);
    let llfty = type_of_fn_from_ty(ccx, mono_ty);

    ccx.stats.n_monos += 1;

    let depth = option::get_default(ccx.monomorphizing.find(fn_id), 0u);
    // Random cut-off -- code that needs to instantiate the same function
    // recursively more than ten times can probably safely be assumed to be
    // causing an infinite expansion.
    if depth > 10 {
        ccx.sess.span_fatal(
            span, ~"overly deep expansion of inlined function");
    }
    ccx.monomorphizing.insert(fn_id, depth + 1);

    let pt = vec::append(*pt,
                         ~[path_name(ccx.names(ccx.sess.str_of(name)))]);
    let s = mangle_exported_name(ccx, pt, mono_ty);

    let mk_lldecl = || {
        let lldecl = decl_internal_cdecl_fn(ccx.llmod, s, llfty);
        ccx.monomorphized.insert(hash_id, lldecl);
        lldecl
    };

    let psubsts = Some({
        tys: substs,
        vtables: vtables,
        bounds: tpt.bounds,
        self_ty: impl_ty_opt
    });

    let lldecl = match map_node {
      ast_map::node_item(i@@{node: ast::item_fn(decl, _, _, body), _}, _) => {
        let d = mk_lldecl();
        set_inline_hint_if_appr(i.attrs, d);
        trans_fn(ccx, pt, decl, body, d, no_self, psubsts, fn_id.node, None);
        d
      }
      ast_map::node_item(*) => {
          ccx.tcx.sess.bug(~"Can't monomorphize this kind of item")
      }
      ast_map::node_foreign_item(i, _, _) => {
          let d = mk_lldecl();
          foreign::trans_intrinsic(ccx, d, i, pt, psubsts.get(),
                                ref_id);
          d
      }
      ast_map::node_variant(v, enum_item, _) => {
        let tvs = ty::enum_variants(ccx.tcx, local_def(enum_item.id));
        let this_tv = option::get(vec::find(*tvs, |tv| {
            tv.id.node == fn_id.node}));
        let d = mk_lldecl();
        set_inline_hint(d);
        match v.node.kind {
            ast::tuple_variant_kind(args) => {
                trans_enum_variant(ccx, enum_item.id, v, args,
                                   this_tv.disr_val, (*tvs).len() == 1u,
                                   psubsts, d);
            }
            ast::struct_variant_kind(_) =>
                ccx.tcx.sess.bug(~"can't monomorphize struct variants"),
            ast::enum_variant_kind(_) =>
                ccx.tcx.sess.bug(~"can't monomorphize enum variants")
        }
        d
      }
      ast_map::node_method(mth, supplied_impl_did, _) => {
        // XXX: What should the self type be here?
        let d = mk_lldecl();
        set_inline_hint_if_appr(mth.attrs, d);

        // Override the impl def ID if necessary.
        let impl_did;
        match impl_did_opt {
            None => impl_did = supplied_impl_did,
            Some(override_impl_did) => impl_did = override_impl_did
        }

        meth::trans_method(ccx, pt, mth, psubsts, None, d, impl_did);
        d
      }
      ast_map::node_dtor(_, dtor, _, pt) => {
        let parent_id = match ty::ty_to_def_id(ty::node_id_to_type(ccx.tcx,
                                              dtor.node.self_id)) {
                Some(did) => did,
                None      => ccx.sess.span_bug(dtor.span, ~"Bad self ty in \
                                                            dtor")
        };
        trans_class_dtor(ccx, *pt, dtor.node.body,
          dtor.node.id, psubsts, Some(hash_id), parent_id)
      }
      ast_map::node_trait_method(@ast::provided(mth), _, pt) => {
        let d = mk_lldecl();
        set_inline_hint_if_appr(mth.attrs, d);
        io::println(fmt!("monomorphic_fn impl_did_opt is %?", impl_did_opt));
        meth::trans_method(ccx, *pt, mth, psubsts, None, d,
                           impl_did_opt.get());
        d
      }

      // Ugh -- but this ensures any new variants won't be forgotten
      ast_map::node_expr(*) |
      ast_map::node_stmt(*) |
      ast_map::node_trait_method(*) |
      ast_map::node_export(*) |
      ast_map::node_arg(*) |
      ast_map::node_block(*) |
      ast_map::node_local(*) => {
        ccx.tcx.sess.bug(fmt!("Can't monomorphize a %?", map_node))
      }
    };
    ccx.monomorphizing.insert(fn_id, depth);

    debug!("leaving monomorphic fn %s", ty::item_path_str(ccx.tcx, fn_id));
    {val: lldecl, must_cast: must_cast}
}

fn normalize_for_monomorphization(tcx: ty::ctxt, ty: ty::t) -> Option<ty::t> {
    // FIXME[mono] could do this recursively. is that worthwhile? (#2529)
    match ty::get(ty).sty {
        ty::ty_box(*) => {
            Some(ty::mk_opaque_box(tcx))
        }
        ty::ty_fn(ref fty) => {
            Some(ty::mk_fn(
                tcx,
                FnTyBase {meta: FnMeta {purity: ast::impure_fn,
                                        proto: fty.meta.proto,
                                        bounds: @~[],
                                        ret_style: ast::return_val},
                          sig: FnSig {inputs: ~[],
                                      output: ty::mk_nil(tcx)}}))
        }
        ty::ty_trait(_, _, _) => {
            let box_proto = ty::proto_vstore(ty::vstore_box);
            Some(ty::mk_fn(
                tcx,
                FnTyBase {meta: FnMeta {purity: ast::impure_fn,
                                        proto: box_proto,
                                        bounds: @~[],
                                        ret_style: ast::return_val},
                          sig: FnSig {inputs: ~[],
                                      output: ty::mk_nil(tcx)}}))
        }
        ty::ty_ptr(_) => {
            Some(ty::mk_uint(tcx))
        }
        _ => {
            None
        }
    }
}

fn make_mono_id(ccx: @crate_ctxt, item: ast::def_id, substs: ~[ty::t],
                vtables: Option<typeck::vtable_res>,
                impl_did_opt: Option<ast::def_id>,
                param_uses: Option<~[type_use::type_uses]>) -> mono_id {
    let precise_param_ids = match vtables {
      Some(vts) => {
        let bounds = ty::lookup_item_type(ccx.tcx, item).bounds;
        let mut i = 0u;
        vec::map2(*bounds, substs, |bounds, subst| {
            let mut v = ~[];
            for bounds.each |bound| {
                match *bound {
                  ty::bound_trait(_) => {
                    v.push(meth::vtable_id(ccx, vts[i]));
                    i += 1u;
                  }
                  _ => ()
                }
            }
            (*subst, if v.len() > 0u { Some(v) } else { None })
        })
      }
      None => {
        vec::map(substs, |subst| (*subst, None))
      }
    };
    let param_ids = match param_uses {
      Some(uses) => {
        vec::map2(precise_param_ids, uses, |id, uses| {
            match *id {
                (a, b@Some(_)) => mono_precise(a, b),
                (subst, None) => {
                    if *uses == 0u {
                        mono_any
                    } else if *uses == type_use::use_repr &&
                        !ty::type_needs_drop(ccx.tcx, subst)
                    {
                        let llty = type_of::type_of(ccx, subst);
                        let size = shape::llsize_of_real(ccx, llty);
                        let align = shape::llalign_of_pref(ccx, llty);
                        let mode = datum::appropriate_mode(subst);

                        // FIXME(#3547)---scalars and floats are
                        // treated differently in most ABIs.  But we
                        // should be doing something more detailed
                        // here.
                        let is_float = match ty::get(subst).sty {
                            ty::ty_float(_) => true,
                            _ => false
                        };

                        // Special value for nil to prevent problems
                        // with undef return pointers.
                        if size == 1u && ty::type_is_nil(subst) {
                            mono_repr(0u, 0u, is_float, mode)
                        } else {
                            mono_repr(size, align, is_float, mode)
                        }
                    } else {
                        mono_precise(subst, None)
                    }
                }
            }
        })
      }
      None => {
          precise_param_ids.map(|x| {
              let (a, b) = *x;
              mono_precise(a, b)
          })
      }
    };
    @{def: item, params: param_ids, impl_did_opt: impl_did_opt}
}
