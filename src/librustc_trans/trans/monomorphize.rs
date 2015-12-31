// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::link::exported_name;
use llvm::ValueRef;
use llvm;
use middle::def_id::DefId;
use middle::infer::normalize_associated_type;
use middle::subst;
use middle::subst::{Subst, Substs};
use middle::ty::fold::{TypeFolder, TypeFoldable};
use trans::attributes;
use trans::base::{trans_enum_variant, push_ctxt, get_item_val};
use trans::base::trans_fn;
use trans::base;
use trans::common::*;
use trans::declare;
use trans::foreign;
use middle::ty::{self, HasTypeFlags, Ty};
use rustc::front::map as hir_map;

use rustc_front::hir;

use syntax::abi;
use syntax::ast;
use syntax::attr;
use syntax::errors;
use std::hash::{Hasher, Hash, SipHasher};

pub fn monomorphic_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                fn_id: DefId,
                                psubsts: &'tcx subst::Substs<'tcx>,
                                ref_id: Option<ast::NodeId>)
                                -> (ValueRef, Ty<'tcx>, bool) {
    debug!("monomorphic_fn(\
            fn_id={:?}, \
            real_substs={:?}, \
            ref_id={:?})",
           fn_id,
           psubsts,
           ref_id);

    assert!(!psubsts.types.needs_infer() && !psubsts.types.has_param_types());

    // we can only monomorphize things in this crate (or inlined into it)
    let fn_node_id = ccx.tcx().map.as_local_node_id(fn_id).unwrap();

    let _icx = push_ctxt("monomorphic_fn");

    let hash_id = MonoId {
        def: fn_id,
        params: &psubsts.types
    };

    let item_ty = ccx.tcx().lookup_item_type(fn_id).ty;

    debug!("monomorphic_fn about to subst into {:?}", item_ty);
    let mono_ty = apply_param_substs(ccx.tcx(), psubsts, &item_ty);
    debug!("mono_ty = {:?} (post-substitution)", mono_ty);

    match ccx.monomorphized().borrow().get(&hash_id) {
        Some(&val) => {
            debug!("leaving monomorphic fn {}",
            ccx.tcx().item_path_str(fn_id));
            return (val, mono_ty, false);
        }
        None => ()
    }

    debug!("monomorphic_fn(\
            fn_id={:?}, \
            psubsts={:?}, \
            hash_id={:?})",
           fn_id,
           psubsts,
           hash_id);


    let map_node = errors::expect(
        ccx.sess().diagnostic(),
        ccx.tcx().map.find(fn_node_id),
        || {
            format!("while monomorphizing {:?}, couldn't find it in \
                     the item map (may have attempted to monomorphize \
                     an item defined in a different crate?)",
                    fn_id)
        });

    if let hir_map::NodeForeignItem(_) = map_node {
        let abi = ccx.tcx().map.get_foreign_abi(fn_node_id);
        if abi != abi::RustIntrinsic && abi != abi::PlatformIntrinsic {
            // Foreign externs don't have to be monomorphized.
            return (get_item_val(ccx, fn_node_id), mono_ty, true);
        }
    }

    ccx.stats().n_monos.set(ccx.stats().n_monos.get() + 1);

    let depth;
    {
        let mut monomorphizing = ccx.monomorphizing().borrow_mut();
        depth = match monomorphizing.get(&fn_id) {
            Some(&d) => d, None => 0
        };

        debug!("monomorphic_fn: depth for fn_id={:?} is {:?}", fn_id, depth+1);

        // Random cut-off -- code that needs to instantiate the same function
        // recursively more than thirty times can probably safely be assumed
        // to be causing an infinite expansion.
        if depth > ccx.sess().recursion_limit.get() {
            ccx.sess().span_fatal(ccx.tcx().map.span(fn_node_id),
                "reached the recursion limit during monomorphization");
        }

        monomorphizing.insert(fn_id, depth + 1);
    }

    let hash;
    let s = {
        let mut state = SipHasher::new();
        hash_id.hash(&mut state);
        mono_ty.hash(&mut state);

        hash = format!("h{}", state.finish());
        let path = ccx.tcx().map.def_path_from_id(fn_node_id);
        exported_name(path, &hash[..])
    };

    debug!("monomorphize_fn mangled to {}", s);

    // This shouldn't need to option dance.
    let mut hash_id = Some(hash_id);
    let mut mk_lldecl = |abi: abi::Abi| {
        let lldecl = if abi != abi::Rust {
            foreign::decl_rust_fn_with_foreign_abi(ccx, mono_ty, &s)
        } else {
            // FIXME(nagisa): perhaps needs a more fine grained selection? See
            // setup_lldecl below.
            declare::define_internal_rust_fn(ccx, &s, mono_ty)
        };

        ccx.monomorphized().borrow_mut().insert(hash_id.take().unwrap(), lldecl);
        lldecl
    };
    let setup_lldecl = |lldecl, attrs: &[ast::Attribute]| {
        base::update_linkage(ccx, lldecl, None, base::OriginalTranslation);
        attributes::from_fn_attrs(ccx, attrs, lldecl);

        let is_first = !ccx.available_monomorphizations().borrow().contains(&s);
        if is_first {
            ccx.available_monomorphizations().borrow_mut().insert(s.clone());
        }

        let trans_everywhere = attr::requests_inline(attrs);
        if trans_everywhere && !is_first {
            llvm::SetLinkage(lldecl, llvm::AvailableExternallyLinkage);
        }

        // If `true`, then `lldecl` should be given a function body.
        // Otherwise, it should be left as a declaration of an external
        // function, with no definition in the current compilation unit.
        trans_everywhere || is_first
    };

    let lldecl = match map_node {
        hir_map::NodeItem(i) => {
            match *i {
              hir::Item {
                  node: hir::ItemFn(ref decl, _, _, abi, _, ref body),
                  ..
              } => {
                  let d = mk_lldecl(abi);
                  let needs_body = setup_lldecl(d, &i.attrs);
                  if needs_body {
                      if abi != abi::Rust {
                          foreign::trans_rust_fn_with_foreign_abi(
                              ccx, &**decl, &**body, &[], d, psubsts, fn_node_id,
                              Some(&hash[..]));
                      } else {
                          trans_fn(ccx,
                                   &**decl,
                                   &**body,
                                   d,
                                   psubsts,
                                   fn_node_id,
                                   &i.attrs);
                      }
                  }

                  d
              }
              _ => {
                ccx.sess().bug("Can't monomorphize this kind of item")
              }
            }
        }
        hir_map::NodeVariant(v) => {
            let variant = inlined_variant_def(ccx, fn_node_id);
            assert_eq!(v.node.name, variant.name);
            let d = mk_lldecl(abi::Rust);
            attributes::inline(d, attributes::InlineAttr::Hint);
            trans_enum_variant(ccx, fn_node_id, variant.disr_val, psubsts, d);
            d
        }
        hir_map::NodeImplItem(impl_item) => {
            match impl_item.node {
                hir::ImplItemKind::Method(ref sig, ref body) => {
                    let d = mk_lldecl(abi::Rust);
                    let needs_body = setup_lldecl(d, &impl_item.attrs);
                    if needs_body {
                        trans_fn(ccx,
                                 &sig.decl,
                                 body,
                                 d,
                                 psubsts,
                                 impl_item.id,
                                 &impl_item.attrs);
                    }
                    d
                }
                _ => {
                    ccx.sess().bug(&format!("can't monomorphize a {:?}",
                                           map_node))
                }
            }
        }
        hir_map::NodeTraitItem(trait_item) => {
            match trait_item.node {
                hir::MethodTraitItem(ref sig, Some(ref body)) => {
                    let d = mk_lldecl(abi::Rust);
                    let needs_body = setup_lldecl(d, &trait_item.attrs);
                    if needs_body {
                        trans_fn(ccx,
                                 &sig.decl,
                                 body,
                                 d,
                                 psubsts,
                                 trait_item.id,
                                 &trait_item.attrs);
                    }
                    d
                }
                _ => {
                    ccx.sess().bug(&format!("can't monomorphize a {:?}",
                                           map_node))
                }
            }
        }
        hir_map::NodeStructCtor(struct_def) => {
            let d = mk_lldecl(abi::Rust);
            attributes::inline(d, attributes::InlineAttr::Hint);
            if struct_def.is_struct() {
                panic!("ast-mapped struct didn't have a ctor id")
            }
            base::trans_tuple_struct(ccx,
                                     struct_def.id(),
                                     psubsts,
                                     d);
            d
        }

        // Ugh -- but this ensures any new variants won't be forgotten
        hir_map::NodeForeignItem(..) |
        hir_map::NodeLifetime(..) |
        hir_map::NodeTyParam(..) |
        hir_map::NodeExpr(..) |
        hir_map::NodeStmt(..) |
        hir_map::NodeBlock(..) |
        hir_map::NodePat(..) |
        hir_map::NodeLocal(..) => {
            ccx.sess().bug(&format!("can't monomorphize a {:?}",
                                   map_node))
        }
    };

    ccx.monomorphizing().borrow_mut().insert(fn_id, depth);

    debug!("leaving monomorphic fn {}", ccx.tcx().item_path_str(fn_id));
    (lldecl, mono_ty, true)
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct MonoId<'tcx> {
    pub def: DefId,
    pub params: &'tcx subst::VecPerParamSpace<Ty<'tcx>>
}

/// Monomorphizes a type from the AST by first applying the in-scope
/// substitutions and then normalizing any associated types.
pub fn apply_param_substs<'tcx,T>(tcx: &ty::ctxt<'tcx>,
                                  param_substs: &Substs<'tcx>,
                                  value: &T)
                                  -> T
    where T : TypeFoldable<'tcx> + HasTypeFlags
{
    let substituted = value.subst(tcx, param_substs);
    normalize_associated_type(tcx, &substituted)
}


/// Returns the normalized type of a struct field
pub fn field_ty<'tcx>(tcx: &ty::ctxt<'tcx>,
                      param_substs: &Substs<'tcx>,
                      f: ty::FieldDef<'tcx>)
                      -> Ty<'tcx>
{
    normalize_associated_type(tcx, &f.ty(tcx, param_substs))
}
