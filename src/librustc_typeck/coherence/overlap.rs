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
//! same type. Likewise, no two inherent impls for a given type
//! constructor provide a method with the same name.

use rustc::traits;
use rustc::ty::{self, TyCtxt, TypeFoldable};
use syntax::ast;

pub fn check_impl<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, node_id: ast::NodeId) {
    let impl_def_id = tcx.hir.local_def_id(node_id);
    let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
    let trait_def_id = trait_ref.def_id;

    if trait_ref.references_error() {
        debug!("coherence: skipping impl {:?} with error {:?}",
               impl_def_id, trait_ref);
        return
    }

    // Trigger building the specialization graph for the trait of this impl.
    // This will detect any overlap errors.
    tcx.specialization_graph_of(trait_def_id);


    // check for overlap with the automatic `impl Trait for Trait`
    if let ty::TyDynamic(ref data, ..) = trait_ref.self_ty().sty {
        // This is something like impl Trait1 for Trait2. Illegal
        // if Trait1 is a supertrait of Trait2 or Trait2 is not object safe.

        if data.principal().map_or(true, |p| !tcx.is_object_safe(p.def_id())) {
            // This is an error, but it will be reported by wfcheck.  Ignore it here.
            // This is tested by `coherence-impl-trait-for-trait-object-safe.rs`.
        } else {
            let mut supertrait_def_ids =
                traits::supertrait_def_ids(tcx,
                                           data.principal().unwrap().def_id());
            if supertrait_def_ids.any(|d| d == trait_def_id) {
                span_err!(tcx.sess,
                          tcx.span_of_impl(impl_def_id).unwrap(),
                          E0371,
                          "the object type `{}` automatically \
                           implements the trait `{}`",
                          trait_ref.self_ty(),
                          tcx.item_path_str(trait_def_id));
            }
        }
    }
}
