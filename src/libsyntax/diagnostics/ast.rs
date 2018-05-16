// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::*;
use errors::{SpanContextResolver, SpanContextKind, SpanContext};
use rustc_data_structures::sync::Lrc;
use syntax_pos::Span;
use visit::{self, Visitor, FnKind};

pub struct ContextResolver {
    krate: Lrc<Crate>
}

impl ContextResolver {
    pub fn new(krate: Lrc<Crate>) -> Self {
        Self { krate }
    }
}

struct SpanResolver {
    idents: Vec<Ident>,
    kind: Option<SpanContextKind>,
    span: Span,
}

impl<'ast> Visitor<'ast> for SpanResolver {
    fn visit_trait_item(&mut self, ti: &'ast TraitItem) {
        if ti.span.proper_contains(self.span) {
            self.idents.push(ti.ident);
            self.kind = Some(SpanContextKind::Trait);
            visit::walk_trait_item(self, ti)
        }
    }

    fn visit_impl_item(&mut self, ii: &'ast ImplItem) {
        if ii.span.proper_contains(self.span) {
            self.idents.push(ii.ident);
            self.kind = Some(SpanContextKind::Impl);
            visit::walk_impl_item(self, ii)
        }
    }

    fn visit_item(&mut self, i: &'ast Item) {
        if i.span.proper_contains(self.span) {
            let kind = match i.node {
                ItemKind::Enum(..) => Some(SpanContextKind::Enum),
                ItemKind::Struct(..) => Some(SpanContextKind::Struct),
                ItemKind::Union(..) => Some(SpanContextKind::Union),
                ItemKind::Trait(..) => Some(SpanContextKind::Trait),
                ItemKind::Mod(..) => Some(SpanContextKind::Module),
                _ => None,
            };

            if kind.is_some() {
                self.idents.push(i.ident);
                self.kind = kind;
            }

            visit::walk_item(self, i);
        }
    }

    fn visit_fn(&mut self, fk: FnKind<'ast>, fd: &'ast FnDecl, s: Span, _: NodeId) {
        if s.proper_contains(self.span) {
            match fk {
                FnKind::ItemFn(ref ident, ..) => {
                    self.idents.push(*ident);
                    self.kind = Some(SpanContextKind::Function);
                }
                FnKind::Method(ref ident, ..) => {
                    self.idents.push(*ident);
                    self.kind = Some(SpanContextKind::Method);
                }
                _ => {}
            }

            visit::walk_fn(self, fk, fd, s)
        }
    }
}

impl SpanContextResolver for ContextResolver {
    fn span_to_context(&self, sp: Span) -> Option<SpanContext> {
        let mut sr = SpanResolver {
            idents: Vec::new(),
            span: sp,
            kind: None,
        };
        visit::walk_crate(&mut sr, &*self.krate);

        let SpanResolver { kind, idents, .. } = sr;

        match kind {
            None => None,
            Some(kind) => {
                let path = idents.iter().map(
                    |x| x.to_string()).collect::<Vec<String>>().join("::");
                Some(SpanContext::new(kind, path))
            }
        }
    }
}
