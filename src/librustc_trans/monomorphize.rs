// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::ValueRef;
use llvm;
use rustc::hir::def_id::DefId;
use rustc::infer::TransNormalize;
use rustc::ty::subst;
use rustc::ty::subst::{Subst, Substs};
use rustc::ty::{self, Ty, TypeFoldable, TyCtxt};
use attributes;
use base::{push_ctxt};
use base::trans_fn;
use base;
use common::*;
use declare;
use Disr;
use rustc::hir::map as hir_map;
use rustc::util::ppaux;

use rustc::hir;

use syntax::attr;
use errors;

use std::fmt;

pub fn monomorphic_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                fn_id: DefId,
                                psubsts: &'tcx subst::Substs<'tcx>)
                                -> (ValueRef, Ty<'tcx>) {
    debug!("monomorphic_fn(fn_id={:?}, real_substs={:?})", fn_id, psubsts);

    assert!(!psubsts.types.needs_infer() && !psubsts.types.has_param_types());

    let _icx = push_ctxt("monomorphic_fn");

    let instance = Instance::new(fn_id, psubsts);

    let item_ty = ccx.tcx().lookup_item_type(fn_id).ty;

    debug!("monomorphic_fn about to subst into {:?}", item_ty);
    let mono_ty = apply_param_substs(ccx.tcx(), psubsts, &item_ty);
    debug!("mono_ty = {:?} (post-substitution)", mono_ty);

    if let Some(&val) = ccx.instances().borrow().get(&instance) {
        debug!("leaving monomorphic fn {:?}", instance);
        return (val, mono_ty);
    }

    debug!("monomorphic_fn({:?})", instance);

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
            let error = format!("reached the recursion limit while instantiating `{}`",
                                instance);
            if let Some(id) = ccx.tcx().map.as_local_node_id(fn_id) {
                ccx.sess().span_fatal(ccx.tcx().map.span(id), &error);
            } else {
                ccx.sess().fatal(&error);
            }
        }

        monomorphizing.insert(fn_id, depth + 1);
    }

    let symbol = instance.symbol_name(ccx.shared());

    debug!("monomorphize_fn mangled to {}", symbol);
    assert!(declare::get_defined_value(ccx, &symbol).is_none());

    // FIXME(nagisa): perhaps needs a more fine grained selection?
    let lldecl = declare::define_internal_fn(ccx, &symbol, mono_ty);
    // FIXME(eddyb) Doubt all extern fn should allow unwinding.
    attributes::unwind(lldecl, true);

    ccx.instances().borrow_mut().insert(instance, lldecl);

    // we can only monomorphize things in this crate (or inlined into it)
    let fn_node_id = ccx.tcx().map.as_local_node_id(fn_id).unwrap();
    let map_node = errors::expect(
        ccx.sess().diagnostic(),
        ccx.tcx().map.find(fn_node_id),
        || {
            format!("while instantiating `{}`, couldn't find it in \
                     the item map (may have attempted to monomorphize \
                     an item defined in a different crate?)",
                    instance)
        });
    match map_node {
        hir_map::NodeItem(&hir::Item {
            ref attrs, node: hir::ItemFn(ref decl, _, _, _, _, ref body), ..
        }) |
        hir_map::NodeTraitItem(&hir::TraitItem {
            ref attrs, node: hir::MethodTraitItem(
                hir::MethodSig { ref decl, .. }, Some(ref body)), ..
        }) |
        hir_map::NodeImplItem(&hir::ImplItem {
            ref attrs, node: hir::ImplItemKind::Method(
                hir::MethodSig { ref decl, .. }, ref body), ..
        }) => {
            attributes::from_fn_attrs(ccx, attrs, lldecl);

            let is_first = !ccx.available_monomorphizations().borrow()
                                                             .contains(&symbol);
            if is_first {
                ccx.available_monomorphizations().borrow_mut().insert(symbol.clone());
            }

            let trans_everywhere = attr::requests_inline(attrs);
            if trans_everywhere || is_first {
                let origin = if is_first { base::OriginalTranslation } else { base::InlinedCopy };
                base::update_linkage(ccx, lldecl, None, origin);
                trans_fn(ccx, decl, body, lldecl, psubsts, fn_node_id);
            } else {
                // We marked the value as using internal linkage earlier, but that is illegal for
                // declarations, so switch back to external linkage.
                llvm::SetLinkage(lldecl, llvm::ExternalLinkage);
            }
        }

        hir_map::NodeVariant(_) | hir_map::NodeStructCtor(_) => {
            let disr = match map_node {
                hir_map::NodeVariant(_) => {
                    Disr::from(inlined_variant_def(ccx, fn_node_id).disr_val)
                }
                hir_map::NodeStructCtor(_) => Disr(0),
                _ => bug!()
            };
            attributes::inline(lldecl, attributes::InlineAttr::Hint);
            attributes::set_frame_pointer_elimination(ccx, lldecl);
            base::trans_ctor_shim(ccx, fn_node_id, disr, psubsts, lldecl);
        }

        _ => bug!("can't monomorphize a {:?}", map_node)
    };

    ccx.monomorphizing().borrow_mut().insert(fn_id, depth);

    debug!("leaving monomorphic fn {}", ccx.tcx().item_path_str(fn_id));
    (lldecl, mono_ty)
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Instance<'tcx> {
    pub def: DefId,
    pub substs: &'tcx Substs<'tcx>,
}

impl<'tcx> fmt::Display for Instance<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ppaux::parameterized(f, &self.substs, self.def, ppaux::Ns::Value, &[],
                             |tcx| Some(tcx.lookup_item_type(self.def).generics))
    }
}

impl<'tcx> Instance<'tcx> {
    pub fn new(def_id: DefId, substs: &'tcx Substs<'tcx>)
               -> Instance<'tcx> {
        assert!(substs.regions.iter().all(|&r| r == ty::ReErased));
        Instance { def: def_id, substs: substs }
    }
    pub fn mono<'a>(scx: &SharedCrateContext<'a, 'tcx>, def_id: DefId) -> Instance<'tcx> {
        Instance::new(def_id, scx.empty_substs_for_def_id(def_id))
    }
}

/// Monomorphizes a type from the AST by first applying the in-scope
/// substitutions and then normalizing any associated types.
pub fn apply_param_substs<'a, 'tcx, T>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       param_substs: &Substs<'tcx>,
                                       value: &T)
                                       -> T
    where T: TransNormalize<'tcx>
{
    let substituted = value.subst(tcx, param_substs);
    tcx.normalize_associated_type(&substituted)
}


/// Returns the normalized type of a struct field
pub fn field_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          param_substs: &Substs<'tcx>,
                          f: ty::FieldDef<'tcx>)
                          -> Ty<'tcx>
{
    tcx.normalize_associated_type(&f.ty(tcx, param_substs))
}
