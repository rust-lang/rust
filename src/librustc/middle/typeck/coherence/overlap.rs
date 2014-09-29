// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Overlap: No two impls for the same trait are implemented for the
 * same type.
 */

use middle::traits;
use middle::ty;
use middle::typeck::infer::{new_infer_ctxt};
use middle::typeck::infer;
use syntax::ast::{DefId};
use syntax::ast::{LOCAL_CRATE};
use syntax::ast;
use syntax::codemap::{Span};
use util::ppaux::Repr;

pub fn check(tcx: &ty::ctxt) {
    let overlap = OverlapChecker { tcx: tcx };
    overlap.check_for_overlapping_impls();
}

struct OverlapChecker<'cx, 'tcx:'cx> {
    tcx: &'cx ty::ctxt<'tcx>
}

impl<'cx, 'tcx> OverlapChecker<'cx, 'tcx> {
    fn check_for_overlapping_impls(&self) {
        debug!("check_for_overlapping_impls");
        let trait_impls = self.tcx.trait_impls.borrow();
        for trait_def_id in trait_impls.keys() {
            self.check_for_overlapping_impls_of_trait(*trait_def_id);
        }
    }

    fn check_for_overlapping_impls_of_trait(&self,
                                            trait_def_id: ast::DefId)
    {
        debug!("check_for_overlapping_impls_of_trait(trait_def_id={})",
               trait_def_id.repr(self.tcx));

        // FIXME -- it seems like this method actually pushes
        // duplicate impls onto the list
        ty::populate_implementations_for_trait_if_necessary(self.tcx,
                                                            trait_def_id);

        let mut impls = Vec::new();
        self.push_impls_of_trait(trait_def_id, &mut impls);

        for (i, &impl1_def_id) in impls.iter().enumerate() {
            if impl1_def_id.krate != ast::LOCAL_CRATE {
                // we don't need to check impls if both are external;
                // that's the other crate's job.
                continue;
            }

            for &impl2_def_id in impls.slice_from(i+1).iter() {
                self.check_if_impls_overlap(trait_def_id,
                                            impl1_def_id,
                                            impl2_def_id);
            }
        }
    }

    fn check_if_impls_overlap(&self,
                              trait_def_id: ast::DefId,
                              impl1_def_id: ast::DefId,
                              impl2_def_id: ast::DefId)
    {
        assert_eq!(impl1_def_id.krate, ast::LOCAL_CRATE);

        debug!("check_if_impls_overlap({}, {}, {})",
               trait_def_id.repr(self.tcx),
               impl1_def_id.repr(self.tcx),
               impl2_def_id.repr(self.tcx));

        let infcx = infer::new_infer_ctxt(self.tcx);
        if !traits::overlapping_impls(&infcx, impl1_def_id, impl2_def_id) {
            return;
        }

        span_err!(self.tcx.sess, self.span_of_impl(impl1_def_id), E0119,
                  "conflicting implementations for trait `{}`",
                  ty::item_path_str(self.tcx, trait_def_id));

        if impl2_def_id.krate == ast::LOCAL_CRATE {
            span_note!(self.tcx.sess, self.span_of_impl(impl2_def_id),
                       "note conflicting implementation here");
        } else {
            let crate_store = &self.tcx.sess.cstore;
            let cdata = crate_store.get_crate_data(impl2_def_id.krate);
            span_note!(self.tcx.sess, self.span_of_impl(impl1_def_id),
                       "conflicting implementation in crate `{}`",
                       cdata.name);
        }
    }

    fn push_impls_of_trait(&self,
                           trait_def_id: ast::DefId,
                           out: &mut Vec<ast::DefId>) {
        match self.tcx.trait_impls.borrow().find(&trait_def_id) {
            Some(impls) => { out.push_all(impls.borrow().as_slice()); }
            None => { /* no impls */ }
        }
    }

    fn span_of_impl(&self, impl_did: ast::DefId) -> Span {
        assert_eq!(impl_did.krate, ast::LOCAL_CRATE);
        self.tcx.map.span(impl_did.node)
    }
}
