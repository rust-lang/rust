// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


/// Bounds Checkers
/// This runs after the type checker and enforces that built-in trait
/// implementations are fulfilled by the type they are implemented on.
/// For a `struct` to fulfill a built-in trait B, all its fields types
/// must have implement such trait.

use middle::ty;
use middle::typeck::{CrateCtxt};
use util::ppaux;
use syntax::ast;
use syntax::codemap::Span;
use syntax::visit::Visitor;
use syntax::visit;
use syntax::print::pprust;


struct BoundContext<'a> {
    tcx: &'a ty::ctxt
}

impl<'a> BoundContext<'a> {
    fn report_error(&self, i: &ast::Item, note_span: Span, error: &str, msg: &str) {
        self.tcx.sess.span_err(i.span, error);
        self.tcx.sess.span_note(note_span, msg);
    }
}

impl<'a> Visitor<()> for BoundContext<'a> {

    fn visit_item(&mut self, i: &ast::Item, _: ()) {
        match i.node {
            ast::ItemImpl(_, Some(ref trait_ref), _, _) => {
                let tref = ty::node_id_to_trait_ref(self.tcx, trait_ref.ref_id);
                if !ty::is_built_in_trait(self.tcx, &*tref) {
                    return
                }

                debug!("built-in trait implementation found: item={}",
                       pprust::item_to_str(i))

                // `nty` is `Type` in `impl Trait for Type`
                let nty = ty::node_id_to_type(self.tcx, i.id);
                match ty::get(nty).sty {
                    ty::ty_struct(did, ref substs) => {
                        let fields = ty::lookup_struct_fields(self.tcx, did);
                        let span = self.tcx.map.span(did.node);

                        for field in fields.iter() {
                            let fty = ty::lookup_field_type(self.tcx, did, field.id, substs);
                            if !ty::type_fulfills_trait(self.tcx, fty, tref.clone()) {
                                self.report_error(i, span,
                                        format!("cannot implement the trait `{}` \
                                                 on type `{}` because the field with type \
                                                 `{}` doesn't fulfill such trait.",
                                                 ppaux::trait_ref_to_str(self.tcx,
                                                                         &*tref),
                                                ppaux::ty_to_str(self.tcx, nty),
                                                ppaux::ty_to_str(self.tcx, fty)).as_slice(),
                                        format!("field `{}`, declared in this struct",
                                                   ppaux::ty_to_str(self.tcx, fty)).as_slice());
                            }
                        }
                    }
                    ty::ty_enum(did, ref substs) => {
                        let variants = ty::substd_enum_variants(self.tcx, did, substs);
                        for variant in variants.iter() {
                            let span = self.tcx.map.span(variant.id.node);
                            for arg in variant.args.iter() {
                                if !ty::type_fulfills_trait(self.tcx, *arg, tref.clone()) {
                                    self.report_error(i, span,
                                          format!("cannot implement the trait `{}` \
                                                   on type `{}` because variant arg with type \
                                                   `{}` doesn't fulfill such trait.",
                                                   ppaux::trait_ref_to_str(self.tcx,
                                                                           &*tref),
                                                  ppaux::ty_to_str(self.tcx, nty),
                                                  ppaux::ty_to_str(self.tcx, *arg)).as_slice(),
                                          format!("variant arg {}, declared here",
                                                  ppaux::ty_to_str(self.tcx, *arg)).as_slice());
                                }
                            }
                        }
                    }
                    _ => {
                        self.tcx.sess.span_err(i.span,
                                             format!("can only implement built-in trait \
                                                     `{}` on a struct or enum",
                                                     ppaux::trait_ref_to_str(self.tcx,
                                                                              &*tref)).as_slice());

                    }
                }
            }
            _ => {}
        }
    }
}

pub fn check_bounds(ccx: &CrateCtxt, krate: &ast::Crate) {
    let mut visitor = BoundContext {tcx: ccx.tcx};
    visit::walk_crate(&mut visitor, krate, ());
}
