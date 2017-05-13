// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::infer::InferCtxt;
use rustc::traits::{self, FulfillmentContext, Normalized, MiscObligation, SelectionContext,
                    ObligationCause};
use rustc::ty::fold::TypeFoldable;
use syntax::ast;
use syntax_pos::Span;

// FIXME(@jroesch): Ideally we should be able to drop the fulfillment_cx argument.
pub fn normalize_associated_types_in<'a, 'gcx, 'tcx, T>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    fulfillment_cx: &mut FulfillmentContext<'tcx>,
    span: Span,
    body_id: ast::NodeId,
    value: &T) -> T

    where T : TypeFoldable<'tcx>
{
    debug!("normalize_associated_types_in(value={:?})", value);
    let mut selcx = SelectionContext::new(infcx);
    let cause = ObligationCause::new(span, body_id, MiscObligation);
    let Normalized { value: result, obligations } = traits::normalize(&mut selcx, cause, value);
    debug!("normalize_associated_types_in: result={:?} predicates={:?}",
           result,
           obligations);
    for obligation in obligations {
        fulfillment_cx.register_predicate_obligation(infcx, obligation);
    }
    result
}
