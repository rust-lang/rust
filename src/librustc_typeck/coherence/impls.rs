// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementations checker: builtin traits and default impls are allowed just
//! for structs and enums.

use middle::ty;
use syntax::ast::{Item, ItemImpl};
use syntax::ast;
use syntax::visit;

pub fn check(tcx: &ty::ctxt) {
    let mut impls = ImplsChecker { tcx: tcx };
    visit::walk_crate(&mut impls, tcx.map.krate());
}

struct ImplsChecker<'cx, 'tcx:'cx> {
    tcx: &'cx ty::ctxt<'tcx>
}

impl<'cx, 'tcx,'v> visit::Visitor<'v> for ImplsChecker<'cx, 'tcx> {
    fn visit_item(&mut self, item: &'v ast::Item) {
        match item.node {
            ast::ItemImpl(_, _, _, Some(_), _, _) => {
                let trait_ref = ty::impl_id_to_trait_ref(self.tcx, item.id);
                if let Some(_) = self.tcx.lang_items.to_builtin_kind(trait_ref.def_id) {
                    match trait_ref.self_ty().sty {
                        ty::ty_struct(..) | ty::ty_enum(..) => {}
                        _ => {
                            span_err!(self.tcx.sess, item.span, E0209,
                                "builtin traits can only be \
                                          implemented on structs or enums");
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
