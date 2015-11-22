// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use session::Session;

use syntax::ast;
use syntax::attr::AttrMetaMethods;
use syntax::visit;
use syntax::visit::Visitor;

#[derive(Copy, Clone, PartialEq)]
enum Target {
    Fn,
    Struct,
    Enum,
    Other,
}

impl Target {
    fn from_item(item: &ast::Item) -> Target {
        match item.node {
            ast::ItemFn(..) => Target::Fn,
            ast::ItemStruct(..) => Target::Struct,
            ast::ItemEnum(..) => Target::Enum,
            _ => Target::Other,
        }
    }
}

struct CheckAttrVisitor<'a> {
    sess: &'a Session,
}

impl<'a> CheckAttrVisitor<'a> {
    fn check_inline(&self, attr: &ast::Attribute, target: Target) {
        if target != Target::Fn {
            span_err!(self.sess, attr.span, E0518, "attribute should be applied to function");
        }
    }

    fn check_repr(&self, attr: &ast::Attribute, target: Target) {
        let words = match attr.meta_item_list() {
            Some(words) => words,
            None => {
                return;
            }
        };
        for word in words {
            let word: &str = &word.name();
            let message = match word {
                "C" => {
                    if target != Target::Struct && target != Target::Enum {
                            "attribute should be applied to struct or enum"
                    } else {
                        continue
                    }
                }
                "packed" |
                "simd" => {
                    if target != Target::Struct {
                        "attribute should be applied to struct"
                    } else {
                        continue
                    }
                }
                "i8" | "u8" | "i16" | "u16" |
                "i32" | "u32" | "i64" | "u64" |
                "isize" | "usize" => {
                    if target != Target::Enum {
                            "attribute should be applied to enum"
                    } else {
                        continue
                    }
                }
                _ => continue,
            };
            span_err!(self.sess, attr.span, E0517, "{}", message);
        }
    }

    fn check_attribute(&self, attr: &ast::Attribute, target: Target) {
        let name: &str = &attr.name();
        match name {
            "inline" => self.check_inline(attr, target),
            "repr" => self.check_repr(attr, target),
            _ => (),
        }
    }
}

impl<'a, 'v> Visitor<'v> for CheckAttrVisitor<'a> {
    fn visit_item(&mut self, item: &ast::Item) {
        let target = Target::from_item(item);
        for attr in &item.attrs {
            self.check_attribute(attr, target);
        }
    }
}

pub fn check_crate(sess: &Session, krate: &ast::Crate) {
    visit::walk_crate(&mut CheckAttrVisitor { sess: sess }, krate);
}
