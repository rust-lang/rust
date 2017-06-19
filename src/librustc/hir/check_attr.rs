// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module implements some validity checks for attributes.
//! In particular it verifies that `#[inline]` and `#[repr]` attributes are
//! attached to items that actually support them and if there are
//! conflicts between multiple such attributes attached to the same
//! item.

use session::Session;

use syntax::ast;
use syntax::visit;
use syntax::visit::Visitor;

#[derive(Copy, Clone, PartialEq)]
enum Target {
    Fn,
    Struct,
    Union,
    Enum,
    Other,
}

impl Target {
    fn from_item(item: &ast::Item) -> Target {
        match item.node {
            ast::ItemKind::Fn(..) => Target::Fn,
            ast::ItemKind::Struct(..) => Target::Struct,
            ast::ItemKind::Union(..) => Target::Union,
            ast::ItemKind::Enum(..) => Target::Enum,
            _ => Target::Other,
        }
    }
}

struct CheckAttrVisitor<'a> {
    sess: &'a Session,
}

impl<'a> CheckAttrVisitor<'a> {
    /// Check any attribute.
    fn check_attribute(&self, attr: &ast::Attribute, target: Target) {
        if let Some(name) = attr.name() {
            match &*name.as_str() {
                "inline" => self.check_inline(attr, target),
                "repr" => self.check_repr(attr, target),
                _ => (),
            }
        }
    }

    /// Check if an `#[inline]` is applied to a function.
    fn check_inline(&self, attr: &ast::Attribute, target: Target) {
        if target != Target::Fn {
            struct_span_err!(self.sess, attr.span, E0518, "attribute should be applied to function")
                .span_label(attr.span, "requires a function")
                .emit();
        }
    }

    /// Check if an `#[repr]` attr is valid.
    fn check_repr(&self, attr: &ast::Attribute, target: Target) {
        let words = match attr.meta_item_list() {
            Some(words) => words,
            None => {
                return;
            }
        };

        let mut conflicting_reprs = 0;
        let mut found_packed = false;
        let mut found_align = false;

        for word in words {

            let name = match word.name() {
                Some(word) => word,
                None => continue,
            };

            let (message, label) = match &*name.as_str() {
                "C" => {
                    conflicting_reprs += 1;
                    if target != Target::Struct &&
                            target != Target::Union &&
                            target != Target::Enum {
                                ("attribute should be applied to struct, enum or union",
                                 "a struct, enum or union")
                    } else {
                        continue
                    }
                }
                "packed" => {
                    // Do not increment conflicting_reprs here, because "packed"
                    // can be used to modify another repr hint
                    if target != Target::Struct &&
                            target != Target::Union {
                                ("attribute should be applied to struct or union",
                                 "a struct or union")
                    } else {
                        found_packed = true;
                        continue
                    }
                }
                "simd" => {
                    conflicting_reprs += 1;
                    if target != Target::Struct {
                        ("attribute should be applied to struct",
                         "a struct")
                    } else {
                        continue
                    }
                }
                "align" => {
                    found_align = true;
                    if target != Target::Struct {
                        ("attribute should be applied to struct",
                         "a struct")
                    } else {
                        continue
                    }
                }
                "i8" | "u8" | "i16" | "u16" |
                "i32" | "u32" | "i64" | "u64" |
                "isize" | "usize" => {
                    conflicting_reprs += 1;
                    if target != Target::Enum {
                        ("attribute should be applied to enum",
                         "an enum")
                    } else {
                        continue
                    }
                }
                _ => continue,
            };
            struct_span_err!(self.sess, attr.span, E0517, "{}", message)
                .span_label(attr.span, format!("requires {}", label))
                .emit();
        }
        if conflicting_reprs > 1 {
            span_warn!(self.sess, attr.span, E0566,
                       "conflicting representation hints");
        }
        if found_align && found_packed {
            struct_span_err!(self.sess, attr.span, E0587,
                             "conflicting packed and align representation hints").emit();
        }
    }
}

impl<'a> Visitor<'a> for CheckAttrVisitor<'a> {
    fn visit_item(&mut self, item: &'a ast::Item) {
        let target = Target::from_item(item);
        for attr in &item.attrs {
            self.check_attribute(attr, target);
        }
        visit::walk_item(self, item);
    }
}

pub fn check_crate(sess: &Session, krate: &ast::Crate) {
    visit::walk_crate(&mut CheckAttrVisitor { sess: sess }, krate);
}
