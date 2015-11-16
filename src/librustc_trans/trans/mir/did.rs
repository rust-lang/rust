// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code for translating references to other items (DefIds).

use syntax::codemap::DUMMY_SP;
use rustc::front::map;
use rustc::middle::ty::{self, Ty, HasTypeFlags};
use rustc::middle::subst::Substs;
use rustc::middle::const_eval;
use rustc::middle::def_id::DefId;
use rustc::middle::subst;
use rustc::middle::traits;
use rustc::mir::repr::ItemKind;
use trans::common::{Block, fulfill_obligation};
use trans::base;
use trans::expr;
use trans::monomorphize;
use trans::meth;
use trans::inline;

use super::MirContext;
use super::operand::{OperandRef, OperandValue};

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    /// Translate reference to item.
    pub fn trans_item_ref(&mut self,
                          bcx: Block<'bcx, 'tcx>,
                          ty: Ty<'tcx>,
                          kind: ItemKind,
                          substs: &'tcx Substs<'tcx>,
                          did: DefId)
                          -> OperandRef<'tcx> {
        match kind {
            ItemKind::Function |
            ItemKind::Struct |
            ItemKind::Variant => self.trans_fn_ref(bcx, ty, substs, did),
            ItemKind::Method => match bcx.tcx().impl_or_trait_item(did).container() {
                ty::ImplContainer(_) => self.trans_fn_ref(bcx, ty, substs, did),
                ty::TraitContainer(tdid) => self.trans_static_method(bcx, ty, did, tdid, substs)
            },
            ItemKind::Constant => {
                let did = inline::maybe_instantiate_inline(bcx.ccx(), did);
                let expr = const_eval::lookup_const_by_id(bcx.tcx(), did, None)
                            .expect("def was const, but lookup_const_by_id failed");
                let d = expr::trans(bcx, expr);
                OperandRef::from_rvalue_datum(d.datum.to_rvalue_datum(d.bcx, "").datum)
            }
        }
    }

    /// Translates references to a function-like items.
    ///
    /// That includes regular functions, non-static methods, struct and enum variant constructors,
    /// closures and possibly more.
    ///
    /// This is an adaptation of callee::trans_fn_ref_with_substs.
    pub fn trans_fn_ref(&mut self,
                        bcx: Block<'bcx, 'tcx>,
                        ty: Ty<'tcx>,
                        substs: &'tcx Substs<'tcx>,
                        did: DefId)
                        -> OperandRef<'tcx> {
        let did = inline::maybe_instantiate_inline(bcx.ccx(), did);

        if !substs.types.is_empty() || is_named_tuple_constructor(bcx.tcx(), did) {
            let (val, fn_ty, _) = monomorphize::monomorphic_fn(bcx.ccx(), did, substs, None);
            // FIXME: cast fnptr to proper type if necessary
            OperandRef {
                ty: fn_ty,
                val: OperandValue::Immediate(val)
            }
        } else {
            let val = if let Some(node_id) = bcx.tcx().map.as_local_node_id(did) {
                base::get_item_val(bcx.ccx(), node_id)
            } else {
                base::trans_external_path(bcx.ccx(), did, ty)
            };
            // FIXME: cast fnptr to proper type if necessary
            OperandRef {
                ty: ty,
                val: OperandValue::Immediate(val)
            }
        }
    }

    /// Translates references to static methods.
    ///
    /// This is an adaptation of meth::trans_static_method_callee
    pub fn trans_static_method(&mut self,
                               bcx: Block<'bcx, 'tcx>,
                               ty: Ty<'tcx>,
                               method_id: DefId,
                               trait_id: DefId,
                               substs: &'tcx Substs<'tcx>)
                               -> OperandRef<'tcx> {
        let ccx = bcx.ccx();
        let tcx = bcx.tcx();
        let mname = tcx.item_name(method_id);
        let subst::SeparateVecsPerParamSpace {
            types: rcvr_type,
            selfs: rcvr_self,
            fns: rcvr_method
        } = substs.clone().types.split();
        let trait_substs = Substs::erased(
            subst::VecPerParamSpace::new(rcvr_type, rcvr_self, Vec::new())
        );
        let trait_substs = tcx.mk_substs(trait_substs);
        let trait_ref = ty::Binder(ty::TraitRef::new(trait_id, trait_substs));
        let vtbl = fulfill_obligation(ccx, DUMMY_SP, trait_ref);
        match vtbl {
            traits::VtableImpl(traits::VtableImplData { impl_def_id, substs: imp_substs, .. }) => {
                assert!(!imp_substs.types.needs_infer());
                let subst::SeparateVecsPerParamSpace {
                    types: impl_type,
                    selfs: impl_self,
                    fns: _
                } = imp_substs.types.split();
                let callee_substs = Substs::erased(
                    subst::VecPerParamSpace::new(impl_type, impl_self, rcvr_method)
                );
                let mth = tcx.get_impl_method(impl_def_id, callee_substs, mname);
                let mthsubsts = tcx.mk_substs(mth.substs);
                self.trans_fn_ref(bcx, ty, mthsubsts, mth.method.def_id)
            },
            traits::VtableObject(ref data) => {
                let idx = traits::get_vtable_index_of_object_method(tcx, data, method_id);
                OperandRef::from_rvalue_datum(
                    meth::trans_object_shim(ccx, data.upcast_trait_ref.clone(), method_id, idx)
                )
            }
            _ => {
                tcx.sess.bug(&format!("static call to invalid vtable: {:?}", vtbl));
            }
        }
   }
}

fn is_named_tuple_constructor(tcx: &ty::ctxt, def_id: DefId) -> bool {
    let node_id = match tcx.map.as_local_node_id(def_id) {
        Some(n) => n,
        None => { return false; }
    };
    match tcx.map.find(node_id).expect("local item should be in ast map") {
        map::NodeVariant(v) => {
            v.node.data.is_tuple()
        }
        map::NodeStructCtor(_) => true,
        _ => false
    }
}
