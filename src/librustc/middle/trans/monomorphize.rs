// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use back::link::mangle_exported_name;
use driver::session;
use lib::llvm::ValueRef;
use middle::trans::base::{get_insn_ctxt};
use middle::trans::base::{set_inline_hint_if_appr, set_inline_hint};
use middle::trans::base::{trans_enum_variant, trans_struct_dtor};
use middle::trans::base::{trans_fn, decl_internal_cdecl_fn};
use middle::trans::base::{get_item_val, no_self};
use middle::trans::base;
use middle::trans::common::*;
use middle::trans::datum;
use middle::trans::foreign;
use middle::trans::machine;
use middle::trans::meth;
use middle::trans::type_of::type_of_fn_from_ty;
use middle::trans::type_of;
use middle::trans::type_use;
use middle::ty;
use middle::ty::{FnSig};
use middle::typeck;
use util::ppaux::ty_to_str;

use core::vec;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_map::path_name;
use syntax::ast_util::local_def;
use syntax::opt_vec;
use syntax::parse::token::special_idents;
use syntax::abi::AbiSet;

pub fn monomorphic_fn(ccx: @CrateContext,
                      fn_id: ast::def_id,
                      real_substs: &[ty::t],
                      vtables: Option<typeck::vtable_res>,
                      impl_did_opt: Option<ast::def_id>,
                      ref_id: Option<ast::node_id>) ->
                      (ValueRef, bool) {
    assert!(real_substs.all(|t| !ty::type_needs_infer(*t)));
    let _icx = ccx.insn_ctxt("monomorphic_fn");
    let mut must_cast = false;
    let substs = vec::map(real_substs, |t| {
        match normalize_for_monomorphization(ccx.tcx, *t) {
          Some(t) => { must_cast = true; t }
          None => *t
        }
    });

    for real_substs.each() |s| { assert!(!ty::type_has_params(*s)); }
    for substs.each() |s| { assert!(!ty::type_has_params(*s)); }
    let param_uses = type_use::type_uses_for(ccx, fn_id, substs.len());
    // XXX: Bad copy.
    let hash_id = make_mono_id(ccx, fn_id, copy substs, vtables, impl_did_opt,
                               Some(param_uses));
    if vec::any(hash_id.params,
                |p| match *p { mono_precise(_, _) => false, _ => true }) {
        must_cast = true;
    }

    debug!("monomorphic_fn(fn_id=%? (%s), vtables=%?, \
            real_substs=%?, substs=%?, \
            hash_id = %?",
           fn_id, ty::item_path_str(ccx.tcx, fn_id),
           vtables,
           real_substs.map(|s| ty_to_str(ccx.tcx, *s)),
           substs.map(|s| ty_to_str(ccx.tcx, *s)), hash_id);

    match ccx.monomorphized.find(&hash_id) {
      Some(&val) => {
        debug!("leaving monomorphic fn %s",
               ty::item_path_str(ccx.tcx, fn_id));
        return (val, must_cast);
      }
      None => ()
    }

    let tpt = ty::lookup_item_type(ccx.tcx, fn_id);
    let mut llitem_ty = tpt.ty;

    let map_node = session::expect(ccx.sess, ccx.tcx.items.find(&fn_id.node),
     || fmt!("While monomorphizing %?, couldn't find it in the item map \
        (may have attempted to monomorphize an item defined in a different \
        crate?)", fn_id));
    // Get the path so that we can create a symbol
    let (pt, name, span) = match *map_node {
      ast_map::node_item(i, pt) => (pt, i.ident, i.span),
      ast_map::node_variant(ref v, enm, pt) => (pt, (*v).node.name, enm.span),
      ast_map::node_method(m, _, pt) => (pt, m.ident, m.span),
      ast_map::node_foreign_item(i, abis, _, pt) if abis.is_intrinsic()
      => (pt, i.ident, i.span),
      ast_map::node_foreign_item(*) => {
        // Foreign externs don't have to be monomorphized.
        return (get_item_val(ccx, fn_id.node), true);
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
      ast_map::node_arg(*) => ccx.tcx.sess.bug(~"Can't monomorphize an arg"),
      ast_map::node_block(*) => {
          ccx.tcx.sess.bug(~"Can't monomorphize a block")
      }
      ast_map::node_local(*) => {
          ccx.tcx.sess.bug(~"Can't monomorphize a local")
      }
      ast_map::node_struct_ctor(_, i, pt) => (pt, i.ident, i.span)
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

    let depth = match ccx.monomorphizing.find(&fn_id) {
        Some(&d) => d, None => 0
    };
    // Random cut-off -- code that needs to instantiate the same function
    // recursively more than thirty times can probably safely be assumed to be
    // causing an infinite expansion.
    if depth > 30 {
        ccx.sess.span_fatal(
            span, ~"overly deep expansion of inlined function");
    }
    ccx.monomorphizing.insert(fn_id, depth + 1);

    let pt = vec::append(/*bad*/copy *pt,
                         ~[path_name((ccx.names)(
                             copy *ccx.sess.str_of(name)))]);
    let s = mangle_exported_name(ccx, /*bad*/copy pt, mono_ty);

    let mk_lldecl = || {
        let lldecl = decl_internal_cdecl_fn(ccx.llmod, /*bad*/copy s, llfty);
        ccx.monomorphized.insert(hash_id, lldecl);
        lldecl
    };

    let psubsts = Some(@param_substs {
        tys: substs,
        vtables: vtables,
        bounds: tpt.bounds,
        self_ty: impl_ty_opt
    });

    let lldecl = match *map_node {
      ast_map::node_item(i@@ast::item {
                node: ast::item_fn(ref decl, _, _, _, ref body),
                _
            }, _) => {
        let d = mk_lldecl();
        set_inline_hint_if_appr(/*bad*/copy i.attrs, d);
        trans_fn(ccx, pt, decl, body, d, no_self, psubsts, fn_id.node, None);
        d
      }
      ast_map::node_item(*) => {
          ccx.tcx.sess.bug(~"Can't monomorphize this kind of item")
      }
      ast_map::node_foreign_item(i, _, _, _) => {
          let d = mk_lldecl();
          foreign::trans_intrinsic(ccx, d, i, pt, psubsts.get(),
                                ref_id);
          d
      }
      ast_map::node_variant(ref v, enum_item, _) => {
        let tvs = ty::enum_variants(ccx.tcx, local_def(enum_item.id));
        let this_tv = vec::find(*tvs, |tv| { tv.id.node == fn_id.node}).get();
        let d = mk_lldecl();
        set_inline_hint(d);
        match (*v).node.kind {
            ast::tuple_variant_kind(ref args) => {
                trans_enum_variant(ccx, enum_item.id, *v, /*bad*/copy *args,
                                   this_tv.disr_val, psubsts, d);
            }
            ast::struct_variant_kind(_) =>
                ccx.tcx.sess.bug(~"can't monomorphize struct variants"),
        }
        d
      }
      ast_map::node_method(mth, supplied_impl_did, _) => {
        // XXX: What should the self type be here?
        let d = mk_lldecl();
        set_inline_hint_if_appr(/*bad*/copy mth.attrs, d);

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
        trans_struct_dtor(ccx, /*bad*/copy *pt, &dtor.node.body,
          dtor.node.id, psubsts, Some(hash_id), parent_id)
      }
      ast_map::node_trait_method(@ast::provided(mth), _, pt) => {
        let d = mk_lldecl();
        set_inline_hint_if_appr(/*bad*/copy mth.attrs, d);
        debug!("monomorphic_fn impl_did_opt is %?", impl_did_opt);
        meth::trans_method(ccx, /*bad*/copy *pt, mth, psubsts, None, d,
                           impl_did_opt.get());
        d
      }
      ast_map::node_struct_ctor(struct_def, _, _) => {
        let d = mk_lldecl();
        set_inline_hint(d);
        base::trans_tuple_struct(ccx,
                                 /*bad*/copy struct_def.fields,
                                 struct_def.ctor_id.expect(~"ast-mapped tuple struct \
                                                             didn't have a ctor id"),
                                 psubsts,
                                 d);
        d
      }

      // Ugh -- but this ensures any new variants won't be forgotten
      ast_map::node_expr(*) |
      ast_map::node_stmt(*) |
      ast_map::node_trait_method(*) |
      ast_map::node_arg(*) |
      ast_map::node_block(*) |
      ast_map::node_local(*) => {
        ccx.tcx.sess.bug(fmt!("Can't monomorphize a %?", map_node))
      }
    };
    ccx.monomorphizing.insert(fn_id, depth);

    debug!("leaving monomorphic fn %s", ty::item_path_str(ccx.tcx, fn_id));
    (lldecl, must_cast)
}

pub fn normalize_for_monomorphization(tcx: ty::ctxt,
                                      ty: ty::t) -> Option<ty::t> {
    // FIXME[mono] could do this recursively. is that worthwhile? (#2529)
    return match ty::get(ty).sty {
        ty::ty_box(*) => {
            Some(ty::mk_opaque_box(tcx))
        }
        ty::ty_bare_fn(_) => {
            Some(ty::mk_bare_fn(
                tcx,
                ty::BareFnTy {
                    purity: ast::impure_fn,
                    abis: AbiSet::Rust(),
                    sig: FnSig {bound_lifetime_names: opt_vec::Empty,
                                inputs: ~[],
                                output: ty::mk_nil(tcx)}}))
        }
        ty::ty_closure(ref fty) => {
            Some(normalized_closure_ty(tcx, fty.sigil))
        }
        ty::ty_trait(_, _, ref store) => {
            let sigil = match *store {
                ty::UniqTraitStore => ast::OwnedSigil,
                ty::BoxTraitStore | ty::BareTraitStore => ast::ManagedSigil,
                ty::RegionTraitStore(_) => ast::BorrowedSigil,
            };

            // Traits have the same runtime representation as closures.
            Some(normalized_closure_ty(tcx, sigil))
        }
        ty::ty_ptr(_) => {
            Some(ty::mk_uint(tcx))
        }
        _ => {
            None
        }
    };

    fn normalized_closure_ty(tcx: ty::ctxt,
                             sigil: ast::Sigil) -> ty::t
    {
        ty::mk_closure(
            tcx,
            ty::ClosureTy {
                purity: ast::impure_fn,
                sigil: sigil,
                onceness: ast::Many,
                region: ty::re_static,
                sig: ty::FnSig {bound_lifetime_names: opt_vec::Empty,
                                inputs: ~[],
                                output: ty::mk_nil(tcx)}})
    }
}

pub fn make_mono_id(ccx: @CrateContext, item: ast::def_id, substs: &[ty::t],
                    vtables: Option<typeck::vtable_res>,
                    impl_did_opt: Option<ast::def_id>,
                    +param_uses: Option<~[type_use::type_uses]>) -> mono_id {
    let precise_param_ids = match vtables {
      Some(vts) => {
        let bounds = ty::lookup_item_type(ccx.tcx, item).bounds;
        let mut i = 0;
        vec::map2(*bounds, substs, |bounds, subst| {
            let mut v = ~[];
            for bounds.each |bound| {
                match *bound {
                  ty::bound_trait(_) => {
                    v.push(meth::vtable_id(ccx, /*bad*/copy vts[i]));
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
      Some(ref uses) => {
        vec::map2(precise_param_ids, *uses, |id, uses| {
            if ccx.sess.no_monomorphic_collapse() {
                match copy *id {
                    (a, b) => mono_precise(a, b)
                }
            } else {
                match *id {
                    // XXX: Bad copy.
                    (a, copy b@Some(_)) => mono_precise(a, b),
                    (subst, None) => {
                        if *uses == 0u {
                            mono_any
                        } else if *uses == type_use::use_repr &&
                            !ty::type_needs_drop(ccx.tcx, subst)
                        {
                            let llty = type_of::type_of(ccx, subst);
                            let size = machine::llbitsize_of_real(ccx, llty);
                            let align = machine::llalign_of_pref(ccx, llty);
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
                            if size <= 8u && ty::type_is_nil(subst) {
                                mono_repr(0u, 0u, is_float, mode)
                            } else {
                                mono_repr(size, align, is_float, mode)
                            }
                        } else {
                            mono_precise(subst, None)
                        }
                    }
                }
            }
        })
      }
      None => {
          precise_param_ids.map(|x| {
              let (a, b) = copy *x;
              mono_precise(a, b)
          })
      }
    };
    @mono_id_ {def: item, params: param_ids, impl_did_opt: impl_did_opt}
}
