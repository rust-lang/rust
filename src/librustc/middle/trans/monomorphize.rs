// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::link::mangle_exported_name;
use driver::session;
use lib::llvm::ValueRef;
use middle::trans::base::{set_llvm_fn_attrs, set_inline_hint};
use middle::trans::base::{trans_enum_variant, push_ctxt, get_item_val};
use middle::trans::base::{trans_fn, decl_internal_rust_fn};
use middle::trans::base;
use middle::trans::common::*;
use middle::trans::meth;
use middle::trans::intrinsic;
use middle::ty;
use middle::typeck;
use util::ppaux::Repr;

use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::local_def;

pub fn monomorphic_fn(ccx: @CrateContext,
                      fn_id: ast::DefId,
                      real_substs: &ty::substs,
                      vtables: Option<typeck::vtable_res>,
                      self_vtables: Option<typeck::vtable_param_res>,
                      ref_id: Option<ast::NodeId>)
    -> (ValueRef, bool)
{
    debug!("monomorphic_fn(\
            fn_id={}, \
            real_substs={}, \
            vtables={}, \
            self_vtable={}, \
            ref_id={:?})",
           fn_id.repr(ccx.tcx),
           real_substs.repr(ccx.tcx),
           vtables.repr(ccx.tcx),
           self_vtables.repr(ccx.tcx),
           ref_id);

    assert!(real_substs.tps.iter().all(|t| !ty::type_needs_infer(*t)));
    let _icx = push_ctxt("monomorphic_fn");
    let mut must_cast = false;

    let psubsts = @param_substs {
        tys: real_substs.tps.to_owned(),
        vtables: vtables,
        self_ty: real_substs.self_ty.clone(),
        self_vtables: self_vtables
    };

    for s in real_substs.tps.iter() { assert!(!ty::type_has_params(*s)); }
    for s in psubsts.tys.iter() { assert!(!ty::type_has_params(*s)); }

    let hash_id = make_mono_id(ccx, fn_id, &*psubsts);
    if hash_id.params.iter().any(
                |p| match *p { mono_precise(_, _) => false, _ => true }) {
        must_cast = true;
    }

    debug!("monomorphic_fn(\
            fn_id={}, \
            psubsts={}, \
            hash_id={:?})",
           fn_id.repr(ccx.tcx),
           psubsts.repr(ccx.tcx),
           hash_id);

    {
        let monomorphized = ccx.monomorphized.borrow();
        match monomorphized.get().find(&hash_id) {
          Some(&val) => {
            debug!("leaving monomorphic fn {}",
                   ty::item_path_str(ccx.tcx, fn_id));
            return (val, must_cast);
          }
          None => ()
        }
    }

    let tpt = ty::lookup_item_type(ccx.tcx, fn_id);
    let llitem_ty = tpt.ty;

    // We need to do special handling of the substitutions if we are
    // calling a static provided method. This is sort of unfortunate.
    let mut is_static_provided = None;

    let map_node = {
        session::expect(
            ccx.sess,
            ccx.tcx.items.find(fn_id.node),
            || format!("While monomorphizing {:?}, couldn't find it in the \
                        item map (may have attempted to monomorphize an item \
                        defined in a different crate?)", fn_id))
    };

    // Get the path so that we can create a symbol
    let (pt, name, span) = match map_node {
      ast_map::NodeItem(i, pt) => (pt, i.ident, i.span),
      ast_map::NodeVariant(ref v, enm, pt) => (pt, (*v).node.name, enm.span),
      ast_map::NodeMethod(m, _, pt) => (pt, m.ident, m.span),
      ast_map::NodeForeignItem(i, abis, _, pt) if abis.is_intrinsic()
      => (pt, i.ident, i.span),
      ast_map::NodeForeignItem(..) => {
        // Foreign externs don't have to be monomorphized.
        return (get_item_val(ccx, fn_id.node), true);
      }
      ast_map::NodeTraitMethod(method, _, pt) => {
          match *method {
              ast::Provided(m) => {
                // If this is a static provided method, indicate that
                // and stash the number of params on the method.
                if m.explicit_self.node == ast::SelfStatic {
                    is_static_provided = Some(m.generics.ty_params.len());
                }

                (pt, m.ident, m.span)
              }
              ast::Required(_) => {
                ccx.tcx.sess.bug("Can't monomorphize a required trait method")
              }
          }
      }
      ast_map::NodeExpr(..) => {
        ccx.tcx.sess.bug("Can't monomorphize an expr")
      }
      ast_map::NodeStmt(..) => {
        ccx.tcx.sess.bug("Can't monomorphize a stmt")
      }
      ast_map::NodeArg(..) => ccx.tcx.sess.bug("Can't monomorphize an arg"),
      ast_map::NodeBlock(..) => {
          ccx.tcx.sess.bug("Can't monomorphize a block")
      }
      ast_map::NodeLocal(..) => {
          ccx.tcx.sess.bug("Can't monomorphize a local")
      }
      ast_map::NodeCalleeScope(..) => {
          ccx.tcx.sess.bug("Can't monomorphize a callee-scope")
      }
      ast_map::NodeStructCtor(_, i, pt) => (pt, i.ident, i.span)
    };

    debug!("monomorphic_fn about to subst into {}", llitem_ty.repr(ccx.tcx));
    let mono_ty = match is_static_provided {
        None => ty::subst_tps(ccx.tcx, psubsts.tys,
                              psubsts.self_ty, llitem_ty),
        Some(num_method_ty_params) => {
            // Static default methods are a little unfortunate, in
            // that the "internal" and "external" type of them differ.
            // Internally, the method body can refer to Self, but the
            // externally visable type of the method has a type param
            // inserted in between the trait type params and the
            // method type params. The substs that we are given are
            // the proper substs *internally* to the method body, so
            // we have to use those when compiling it.
            //
            // In order to get the proper substitution to use on the
            // type of the method, we pull apart the substitution and
            // stick a substitution for the self type in.
            // This is a bit unfortunate.

            let idx = psubsts.tys.len() - num_method_ty_params;
            let substs =
                (psubsts.tys.slice(0, idx) +
                 &[psubsts.self_ty.unwrap()] +
                 psubsts.tys.tailn(idx));
            debug!("static default: changed substitution to {}",
                   substs.repr(ccx.tcx));

            ty::subst_tps(ccx.tcx, substs, None, llitem_ty)
        }
    };

    let f = match ty::get(mono_ty).sty {
        ty::ty_bare_fn(ref f) => {
            assert!(f.abis.is_rust() || f.abis.is_intrinsic());
            f
        }
        _ => fail!("expected bare rust fn or an intrinsic")
    };

    ccx.stats.n_monos.set(ccx.stats.n_monos.get() + 1);

    let depth;
    {
        let mut monomorphizing = ccx.monomorphizing.borrow_mut();
        depth = match monomorphizing.get().find(&fn_id) {
            Some(&d) => d, None => 0
        };

        // Random cut-off -- code that needs to instantiate the same function
        // recursively more than thirty times can probably safely be assumed
        // to be causing an infinite expansion.
        if depth > 30 {
            ccx.sess.span_fatal(
                span, "overly deep expansion of inlined function");
        }
        monomorphizing.get().insert(fn_id, depth + 1);
    }

    let (_, elt) = gensym_name(ccx.sess.str_of(name));
    let mut pt = (*pt).clone();
    pt.push(elt);
    let s = mangle_exported_name(ccx, pt.clone(), mono_ty);
    debug!("monomorphize_fn mangled to {}", s);

    let mk_lldecl = || {
        let lldecl = decl_internal_rust_fn(ccx, false,
                                           f.sig.inputs,
                                           f.sig.output, s);
        let mut monomorphized = ccx.monomorphized.borrow_mut();
        monomorphized.get().insert(hash_id, lldecl);
        lldecl
    };

    let lldecl = match map_node {
      ast_map::NodeItem(i, _) => {
          match *i {
            ast::Item {
                node: ast::ItemFn(decl, _, _, _, body),
                ..
            } => {
                let d = mk_lldecl();
                set_llvm_fn_attrs(i.attrs, d);
                trans_fn(ccx, pt, decl, body, d, Some(psubsts), fn_id.node, []);
                d
            }
            _ => {
              ccx.tcx.sess.bug("Can't monomorphize this kind of item")
            }
          }
      }
      ast_map::NodeForeignItem(i, _, _, _) => {
          let d = mk_lldecl();
          intrinsic::trans_intrinsic(ccx, d, i, pt, psubsts, i.attrs,
                                     ref_id);
          d
      }
      ast_map::NodeVariant(v, enum_item, _) => {
        let tvs = ty::enum_variants(ccx.tcx, local_def(enum_item.id));
        let this_tv = *tvs.iter().find(|tv| { tv.id.node == fn_id.node}).unwrap();
        let d = mk_lldecl();
        set_inline_hint(d);
        match v.node.kind {
            ast::TupleVariantKind(ref args) => {
                trans_enum_variant(ccx,
                                   enum_item.id,
                                   v,
                                   (*args).clone(),
                                   this_tv.disr_val,
                                   Some(psubsts),
                                   d);
            }
            ast::StructVariantKind(_) =>
                ccx.tcx.sess.bug("can't monomorphize struct variants"),
        }
        d
      }
      ast_map::NodeMethod(mth, _, _) => {
        let d = mk_lldecl();
        set_llvm_fn_attrs(mth.attrs, d);
        trans_fn(ccx, pt, mth.decl, mth.body, d, Some(psubsts), mth.id, []);
        d
      }
      ast_map::NodeTraitMethod(method, _, pt) => {
          match *method {
              ast::Provided(mth) => {
                  let d = mk_lldecl();
                  set_llvm_fn_attrs(mth.attrs, d);
                  trans_fn(ccx, (*pt).clone(), mth.decl, mth.body,
                           d, Some(psubsts), mth.id, []);
                  d
              }
              _ => {
                ccx.tcx.sess.bug(format!("Can't monomorphize a {:?}",
                                         map_node))
              }
          }
      }
      ast_map::NodeStructCtor(struct_def, _, _) => {
        let d = mk_lldecl();
        set_inline_hint(d);
        base::trans_tuple_struct(ccx,
                                 struct_def.fields,
                                 struct_def.ctor_id.expect("ast-mapped tuple struct \
                                                            didn't have a ctor id"),
                                 Some(psubsts),
                                 d);
        d
      }

      // Ugh -- but this ensures any new variants won't be forgotten
      ast_map::NodeExpr(..) |
      ast_map::NodeStmt(..) |
      ast_map::NodeArg(..) |
      ast_map::NodeBlock(..) |
      ast_map::NodeCalleeScope(..) |
      ast_map::NodeLocal(..) => {
        ccx.tcx.sess.bug(format!("Can't monomorphize a {:?}", map_node))
      }
    };

    {
        let mut monomorphizing = ccx.monomorphizing.borrow_mut();
        monomorphizing.get().insert(fn_id, depth);
    }

    debug!("leaving monomorphic fn {}", ty::item_path_str(ccx.tcx, fn_id));
    (lldecl, must_cast)
}

pub fn make_mono_id(ccx: @CrateContext,
                    item: ast::DefId,
                    substs: &param_substs) -> mono_id {
    // FIXME (possibly #5801): Need a lot of type hints to get
    // .collect() to work.
    let substs_iter = substs.self_ty.iter().chain(substs.tys.iter());
    let precise_param_ids: ~[(ty::t, Option<@~[mono_id]>)] = match substs.vtables {
      Some(vts) => {
        debug!("make_mono_id vtables={} substs={}",
               vts.repr(ccx.tcx), substs.tys.repr(ccx.tcx));
        let vts_iter = substs.self_vtables.iter().chain(vts.iter());
        vts_iter.zip(substs_iter).map(|(vtable, subst)| {
            let v = vtable.map(|vt| meth::vtable_id(ccx, vt));
            (*subst, if !v.is_empty() { Some(@v) } else { None })
        }).collect()
      }
      None => substs_iter.map(|subst| (*subst, None::<@~[mono_id]>)).collect()
    };


    let param_ids = precise_param_ids.iter().map(|x| {
        let (a, b) = *x;
        mono_precise(a, b)
    }).collect();
    @mono_id_ {def: item, params: param_ids}
}
