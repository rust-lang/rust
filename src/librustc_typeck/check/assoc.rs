// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::infer::InferCtxt;
use middle::traits::{ObligationCause, ObligationCauseCode, FulfillmentContext};
use middle::ty::{mod, HasProjectionTypes, Ty};
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
        match ty.sty {
            ty::ty_projection(ref data) => {
                let cause =
                    ObligationCause::new(
                        self.span,
                        self.body_id,
                        ObligationCauseCode::MiscObligation);
                let trait_ref = data.trait_ref.clone();
                self.fulfillment_cx
                    .normalize_associated_type(self.infcx,
                                               trait_ref,
                                               data.item_name,
                                               cause)
            }
            _ => {
                ty_fold::super_fold_ty(self, ty)
            }
        }
    }
}
