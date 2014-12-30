// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::infer::InferCtxt;
use middle::traits::{ObligationCause, ObligationCauseCode, FulfillmentContext};
use middle::ty::{mod, RegionEscape, HasProjectionTypes, Ty};
use middle::ty_fold::{mod, TypeFoldable, TypeFolder};
use syntax::ast;
use syntax::codemap::Span;

pub fn normalize_associated_types_in<'a,'tcx,T>(infcx: &InferCtxt<'a,'tcx>,
                                                fulfillment_cx: &mut FulfillmentContext<'tcx>,
                                                span: Span,
                                                body_id: ast::NodeId,
                                                value: &T)
                                                -> T
    where T : TypeFoldable<'tcx> + HasProjectionTypes + Clone
{
    let value = infcx.resolve_type_vars_if_possible(value);

    if !value.has_projection_types() {
        return value.clone();
    }

    let mut normalizer = AssociatedTypeNormalizer { span: span,
                                                    body_id: body_id,
                                                    infcx: infcx,
                                                    fulfillment_cx: fulfillment_cx };
    value.fold_with(&mut normalizer)
}

struct AssociatedTypeNormalizer<'a,'tcx:'a> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    fulfillment_cx: &'a mut FulfillmentContext<'tcx>,
    span: Span,
    body_id: ast::NodeId,
}

impl<'a,'tcx> TypeFolder<'tcx> for AssociatedTypeNormalizer<'a,'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        // We don't want to normalize associated types that occur inside of region
        // binders, because they may contain bound regions, and we can't cope with that.
        //
        // Example:
        //
        //     for<'a> fn(<T as Foo<&'a>>::A)
        //
        // Instead of normalizing `<T as Foo<&'a>>::A` here, we'll
        // normalize it when we instantiate those bound regions (which
        // should occur eventually).

        match ty.sty {
            ty::ty_projection(ref data) if !data.has_escaping_regions() => { // (*)

                // (*) This is kind of hacky -- we need to be able to
                // handle normalization within binders because
                // otherwise we wind up a need to normalize when doing
                // trait matching (since you can have a trait
                // obligation like `for<'a> T::B : Fn(&'a int)`), but
                // we can't normalize with bound regions in scope. So
                // far now we just ignore binders but only normalize
                // if all bound regions are gone (and then we still
                // have to renormalize whenever we instantiate a
                // binder). It would be better to normalize in a
                // binding-aware fashion.

                let cause =
                    ObligationCause::new(
                        self.span,
                        self.body_id,
                        ObligationCauseCode::MiscObligation);
                self.fulfillment_cx
                    .normalize_projection_type(self.infcx,
                                               data.clone(),
                                               cause)
            }
            _ => {
                ty_fold::super_fold_ty(self, ty)
            }
        }
    }
}
