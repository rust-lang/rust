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
    fn check_attributes(&self, item: &ast::Item, target: Target) {
        for attr in &item.attrs {
            if let Some(name) = attr.name() {
                if name == "inline" {
                    self.check_inline(attr, item, target)
                }
            }
        }
        self.check_repr(item, target);
    }

    /// Check if an `#[inline]` is applied to a function.
    fn check_inline(&self, attr: &ast::Attribute, item: &ast::Item, target: Target) {
        if target != Target::Fn {
            struct_span_err!(self.sess, attr.span, E0518, "attribute should be applied to function")
                .span_label(item.span, "not a function")
                .emit();
        }
    }

    /// Check if the `#[repr]` attributes on `item` are valid.
    fn check_repr(&self, item: &ast::Item, target: Target) {
        // Extract the names of all repr hints, e.g., [foo, bar, align] for:
        // ```
        // #[repr(foo)]
        // #[repr(bar, align(8))]
        // ```
        let hints: Vec<_> = item.attrs
            .iter()
            .filter(|attr| match attr.name() {
                Some(name) => name == "repr",
                None => false,
            })
            .filter_map(|attr| attr.meta_item_list())
            .flat_map(|hints| hints)
            .collect();

        let mut int_reprs = 0;
        let mut is_c = false;
        let mut is_simd = false;

        for hint in &hints {
            let name = if let Some(name) = hint.name() {
                name
            } else {
                // Invalid repr hint like repr(42). We don't check for unrecognized hints here
                // (libsyntax does that), so just ignore it.
                continue;
            };

            let (article, allowed_targets) = match &*name.as_str() {
                "C" => {
                    is_c = true;
                    if target != Target::Struct &&
                            target != Target::Union &&
                            target != Target::Enum {
                                ("a", "struct, enum or union")
                    } else {
                        continue
                    }
                }
                "packed" => {
                    if target != Target::Struct &&
                            target != Target::Union {
                                ("a", "struct or union")
                    } else {
                        continue
                    }
                }
                "simd" => {
                    is_simd = true;
                    if target != Target::Struct {
                        ("a", "struct")
                    } else {
                        continue
                    }
                }
                "align" => {
                    if target != Target::Struct &&
                            target != Target::Union {
                        ("a", "struct or union")
                    } else {
                        continue
                    }
                }
                "i8" | "u8" | "i16" | "u16" |
                "i32" | "u32" | "i64" | "u64" |
                "isize" | "usize" => {
                    int_reprs += 1;
                    if target != Target::Enum {
                        ("an", "enum")
                    } else {
                        continue
                    }
                }
                _ => continue,
            };
            struct_span_err!(self.sess, hint.span, E0517,
                             "attribute should be applied to {}", allowed_targets)
                .span_label(item.span, format!("not {} {}", article, allowed_targets))
                .emit();
        }

        // Warn on repr(u8, u16), repr(C, simd), and c-like-enum-repr(C, u8)
        if (int_reprs > 1)
           || (is_simd && is_c)
           || (int_reprs == 1 && is_c && is_c_like_enum(item)) {
            // Just point at all repr hints. This is not ideal, but tracking precisely which ones
            // are at fault is a huge hassle.
            let spans: Vec<_> = hints.iter().map(|hint| hint.span).collect();
            span_warn!(self.sess, spans, E0566,
                       "conflicting representation hints");
        }
    }
}

impl<'a> Visitor<'a> for CheckAttrVisitor<'a> {
    fn visit_item(&mut self, item: &'a ast::Item) {
        let target = Target::from_item(item);
        self.check_attributes(item, target);
        visit::walk_item(self, item);
    }
}

pub fn check_crate(sess: &Session, krate: &ast::Crate) {
    visit::walk_crate(&mut CheckAttrVisitor { sess: sess }, krate);
}

fn is_c_like_enum(item: &ast::Item) -> bool {
    if let ast::ItemKind::Enum(ref def, _) = item.node {
        for variant in &def.variants {
            match variant.node.data {
                ast::VariantData::Unit(_) => { /* continue */ }
                _ => { return false; }
            }
        }
        true
    } else {
        false
    }
}
