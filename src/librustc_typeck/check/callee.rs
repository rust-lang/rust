// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use syntax::codemap::Span;
use CrateCtxt;

/// Check that it is legal to call methods of the trait corresponding
/// to `trait_id` (this only cares about the trait, not the specific
/// method that is called)
pub fn check_legal_trait_for_method_call(ccx: &CrateCtxt, span: Span, trait_id: ast::DefId) {
    let tcx = ccx.tcx;
    let did = Some(trait_id);
    let li = &tcx.lang_items;

    if did == li.drop_trait() {
        span_err!(tcx.sess, span, E0040, "explicit use of destructor method");
    } else if !tcx.sess.features.borrow().unboxed_closures {
        // the #[feature(unboxed_closures)] feature isn't
        // activated so we need to enforce the closure
        // restrictions.

        let method = if did == li.fn_trait() {
            "call"
        } else if did == li.fn_mut_trait() {
            "call_mut"
        } else if did == li.fn_once_trait() {
            "call_once"
        } else {
            return // not a closure method, everything is OK.
        };

        span_err!(tcx.sess, span, E0174,
                  "explicit use of unboxed closure method `{}` is experimental",
                  method);
        span_help!(tcx.sess, span,
                   "add `#![feature(unboxed_closures)]` to the crate attributes to enable");
    }
}
