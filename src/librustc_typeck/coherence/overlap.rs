// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Overlap: No two impls for the same trait are implemented for the
//! same type.

use metadata::cstore::LOCAL_CRATE;
use middle::def_id::DefId;
use middle::traits;
use middle::ty;
use middle::infer::{self, new_infer_ctxt};
use syntax::ast;
use syntax::codemap::Span;
use rustc_front::hir;
use rustc_front::intravisit;
use util::nodemap::DefIdMap;

pub fn check(tcx: &ty::ctxt) {
    let mut overlap = OverlapChecker { tcx: tcx, default_impls: DefIdMap() };
    overlap.check_for_overlapping_impls();

    // this secondary walk specifically checks for some other cases,
    // like defaulted traits, for which additional overlap rules exist
    tcx.map.krate().visit_all_items(&mut overlap);
}

struct OverlapChecker<'cx, 'tcx:'cx> {
    tcx: &'cx ty::ctxt<'tcx>,

    // maps from a trait def-id to an impl id
    default_impls: DefIdMap<ast::NodeId>,
}

impl<'cx, 'tcx> OverlapChecker<'cx, 'tcx> {
    fn check_for_overlapping_impls(&self) {
        debug!("check_for_overlapping_impls");

        // Collect this into a vector to avoid holding the
        // refcell-lock during the
        // check_for_overlapping_impls_of_trait() check, since that
        // check can populate this table further with impls from other
        // crates.
        let trait_defs: Vec<_> = self.tcx.trait_defs.borrow().values().cloned().collect();

        for trait_def in trait_defs {
            self.tcx.populate_implementations_for_trait_if_necessary(trait_def.trait_ref.def_id);
            self.check_for_overlapping_impls_of_trait(trait_def);
        }
    }

    fn check_for_overlapping_impls_of_trait(&self,
                                            trait_def: &'tcx ty::TraitDef<'tcx>)
    {
        debug!("check_for_overlapping_impls_of_trait(trait_def={:?})",
               trait_def);

        // We should already know all impls of this trait, so these
        // borrows are safe.
        let blanket_impls = trait_def.blanket_impls.borrow();
        let nonblanket_impls = trait_def.nonblanket_impls.borrow();
        let trait_def_id = trait_def.trait_ref.def_id;

        // Conflicts can only occur between a blanket impl and another impl,
        // or between 2 non-blanket impls of the same kind.

        for (i, &impl1_def_id) in blanket_impls.iter().enumerate() {
            for &impl2_def_id in &blanket_impls[(i+1)..] {
                self.check_if_impls_overlap(trait_def_id,
                                            impl1_def_id,
                                            impl2_def_id);
            }

            for v in nonblanket_impls.values() {
                for &impl2_def_id in v {
                    self.check_if_impls_overlap(trait_def_id,
                                                impl1_def_id,
                                                impl2_def_id);
                }
            }
        }

        for impl_group in nonblanket_impls.values() {
            for (i, &impl1_def_id) in impl_group.iter().enumerate() {
                for &impl2_def_id in &impl_group[(i+1)..] {
                    self.check_if_impls_overlap(trait_def_id,
                                                impl1_def_id,
                                                impl2_def_id);
                }
            }
        }
    }

    // We need to coherently pick which impl will be displayed
    // as causing the error message, and it must be the in the current
    // crate. Just pick the smaller impl in the file.
    fn order_impls(&self, impl1_def_id: DefId, impl2_def_id: DefId)
            -> Option<(DefId, DefId)> {
        if impl1_def_id.krate != LOCAL_CRATE {
            if impl2_def_id.krate != LOCAL_CRATE {
                // we don't need to check impls if both are external;
                // that's the other crate's job.
                None
            } else {
                Some((impl2_def_id, impl1_def_id))
            }
        } else if impl2_def_id.krate != LOCAL_CRATE {
            Some((impl1_def_id, impl2_def_id))
        } else if impl1_def_id < impl2_def_id {
            Some((impl1_def_id, impl2_def_id))
        } else {
            Some((impl2_def_id, impl1_def_id))
        }
    }


    fn check_if_impls_overlap(&self,
                              trait_def_id: DefId,
                              impl1_def_id: DefId,
                              impl2_def_id: DefId)
    {
        if let Some((impl1_def_id, impl2_def_id)) = self.order_impls(
            impl1_def_id, impl2_def_id)
        {
            debug!("check_if_impls_overlap({:?}, {:?}, {:?})",
                   trait_def_id,
                   impl1_def_id,
                   impl2_def_id);

            let infcx = infer::new_infer_ctxt(self.tcx, &self.tcx.tables, None, false);
            if traits::overlapping_impls(&infcx, impl1_def_id, impl2_def_id) {
                self.report_overlap_error(trait_def_id, impl1_def_id, impl2_def_id);
            }
        }
    }

    fn report_overlap_error(&self, trait_def_id: DefId,
                            impl1: DefId, impl2: DefId) {

        span_err!(self.tcx.sess, self.span_of_impl(impl1), E0119,
                  "conflicting implementations for trait `{}`",
                  self.tcx.item_path_str(trait_def_id));

        self.report_overlap_note(impl2);
    }

    fn report_overlap_note(&self, impl2: DefId) {

        if impl2.is_local() {
            span_note!(self.tcx.sess, self.span_of_impl(impl2),
                       "note conflicting implementation here");
        } else {
            let crate_store = &self.tcx.sess.cstore;
            let cdata = crate_store.get_crate_data(impl2.krate);
            self.tcx.sess.note(&format!("conflicting implementation in crate `{}`", cdata.name));
        }
    }

    fn span_of_impl(&self, impl_did: DefId) -> Span {
        let node_id = self.tcx.map.as_local_node_id(impl_did).unwrap();
        self.tcx.map.span(node_id)
    }
}


impl<'cx, 'tcx,'v> intravisit::Visitor<'v> for OverlapChecker<'cx, 'tcx> {
    fn visit_item(&mut self, item: &'v hir::Item) {
        match item.node {
            hir::ItemDefaultImpl(_, _) => {
                // look for another default impl; note that due to the
                // general orphan/coherence rules, it must always be
                // in this crate.
                let impl_def_id = self.tcx.map.local_def_id(item.id);
                let trait_ref = self.tcx.impl_trait_ref(impl_def_id).unwrap();
                let prev_default_impl = self.default_impls.insert(trait_ref.def_id, item.id);
                match prev_default_impl {
                    Some(prev_id) => {
                        self.report_overlap_error(trait_ref.def_id,
                                                  impl_def_id,
                                                  self.tcx.map.local_def_id(prev_id));
                    }
                    None => { }
                }
            }
            hir::ItemImpl(_, _, _, Some(_), ref self_ty, _) => {
                let impl_def_id = self.tcx.map.local_def_id(item.id);
                let trait_ref = self.tcx.impl_trait_ref(impl_def_id).unwrap();
                let trait_def_id = trait_ref.def_id;
                match trait_ref.self_ty().sty {
                    ty::TyTrait(ref data) => {
                        // This is something like impl Trait1 for Trait2. Illegal
                        // if Trait1 is a supertrait of Trait2 or Trait2 is not object safe.

                        if !traits::is_object_safe(self.tcx, data.principal_def_id()) {
                            // FIXME(#27579). This just means the
                            // self-ty is illegal; WF will report this
                            // error. But it will do so as a warning
                            // for a release or two.  For backwards
                            // compat reasons, then, we continue to
                            // report it here so that things which
                            // were errors remain errors.
                            span_err!(self.tcx.sess, self_ty.span, E0372,
                                      "the trait `{}` cannot be made into an object",
                                      self.tcx.item_path_str(data.principal_def_id()));
                        } else {
                            let mut supertrait_def_ids =
                                traits::supertrait_def_ids(self.tcx, data.principal_def_id());
                            if supertrait_def_ids.any(|d| d == trait_def_id) {
                                span_err!(self.tcx.sess, item.span, E0371,
                                          "the object type `{}` automatically \
                                           implements the trait `{}`",
                                          trait_ref.self_ty(),
                                          self.tcx.item_path_str(trait_def_id));
                            }
                        }
                    }
                    _ => { }
                }
            }
            _ => {
            }
        }
    }
}
