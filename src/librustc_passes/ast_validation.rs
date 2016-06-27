// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Validate AST before lowering it to HIR
//
// This pass is supposed to catch things that fit into AST data structures,
// but not permitted by the language. It runs after expansion when AST is frozen,
// so it can check for erroneous constructions produced by syntax extensions.
// This pass is supposed to perform only simple checks not requiring name resolution
// or type checking or some other kind of complex analysis.

use rustc::lint;
use rustc::session::Session;
use syntax::ast::*;
use syntax::parse::token::{self, keywords};
use syntax::visit::{self, Visitor};
use syntax_pos::Span;
use errors;

struct AstValidator<'a> {
    session: &'a Session,
}

impl<'a> AstValidator<'a> {
    fn err_handler(&self) -> &errors::Handler {
        &self.session.parse_sess.span_diagnostic
    }

    fn check_label(&self, label: Ident, span: Span, id: NodeId) {
        if label.name == keywords::StaticLifetime.name() {
            self.err_handler().span_err(span, &format!("invalid label name `{}`", label.name));
        }
        if label.name.as_str() == "'_" {
            self.session.add_lint(
                lint::builtin::LIFETIME_UNDERSCORE, id, span,
                format!("invalid label name `{}`", label.name)
            );
        }
    }

    fn invalid_visibility(&self, vis: &Visibility, span: Span, note: Option<&str>) {
        if vis != &Visibility::Inherited {
            let mut err = struct_span_err!(self.session, span, E0449,
                                           "unnecessary visibility qualifier");
            if let Some(note) = note {
                err.span_note(span, note);
            }
            err.emit();
        }
    }
}

impl<'a> Visitor for AstValidator<'a> {
    fn visit_lifetime(&mut self, lt: &Lifetime) {
        if lt.name.as_str() == "'_" {
            self.session.add_lint(
                lint::builtin::LIFETIME_UNDERSCORE, lt.id, lt.span,
                format!("invalid lifetime name `{}`", lt.name)
            );
        }

        visit::walk_lifetime(self, lt)
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match expr.node {
            ExprKind::While(_, _, Some(ident)) | ExprKind::Loop(_, Some(ident)) |
            ExprKind::WhileLet(_, _, _, Some(ident)) | ExprKind::ForLoop(_, _, _, Some(ident)) |
            ExprKind::Break(Some(ident)) | ExprKind::Continue(Some(ident)) => {
                self.check_label(ident.node, ident.span, expr.id);
            }
            _ => {}
        }

        visit::walk_expr(self, expr)
    }

    fn visit_path(&mut self, path: &Path, id: NodeId) {
        if path.global && path.segments.len() > 0 {
            let ident = path.segments[0].identifier;
            if token::Ident(ident).is_path_segment_keyword() {
                self.session.add_lint(
                    lint::builtin::SUPER_OR_SELF_IN_GLOBAL_PATH, id, path.span,
                    format!("global paths cannot start with `{}`", ident)
                );
            }
        }

        visit::walk_path(self, path)
    }

    fn visit_item(&mut self, item: &Item) {
        match item.node {
            ItemKind::Use(ref view_path) => {
                let path = view_path.node.path();
                if !path.segments.iter().all(|segment| segment.parameters.is_empty()) {
                    self.err_handler().span_err(path.span, "type or lifetime parameters \
                                                            in import path");
                }
            }
            ItemKind::Impl(_, _, _, Some(..), _, ref impl_items) => {
                self.invalid_visibility(&item.vis, item.span, None);
                for impl_item in impl_items {
                    self.invalid_visibility(&impl_item.vis, impl_item.span, None);
                }
            }
            ItemKind::Impl(_, _, _, None, _, _) => {
                self.invalid_visibility(&item.vis, item.span, Some("place qualifiers on individual \
                                                                    impl items instead"));
            }
            ItemKind::DefaultImpl(..) => {
                self.invalid_visibility(&item.vis, item.span, None);
            }
            ItemKind::ForeignMod(..) => {
                self.invalid_visibility(&item.vis, item.span, Some("place qualifiers on individual \
                                                                    foreign items instead"));
            }
            ItemKind::Enum(ref def, _) => {
                for variant in &def.variants {
                    for field in variant.node.data.fields() {
                        self.invalid_visibility(&field.vis, field.span, None);
                    }
                }
            }
            _ => {}
        }

        visit::walk_item(self, item)
    }

    fn visit_variant_data(&mut self, vdata: &VariantData, _: Ident,
                          _: &Generics, _: NodeId, span: Span) {
        if vdata.fields().is_empty() {
            if vdata.is_tuple() {
                self.err_handler().struct_span_err(span, "empty tuple structs and enum variants \
                                                          are not allowed, use unit structs and \
                                                          enum variants instead")
                                         .span_help(span, "remove trailing `()` to make a unit \
                                                           struct or unit enum variant")
                                         .emit();
            }
        }

        visit::walk_struct_def(self, vdata)
    }

    fn visit_vis(&mut self, vis: &Visibility) {
        match *vis {
            Visibility::Restricted{ref path, ..} => {
                if !path.segments.iter().all(|segment| segment.parameters.is_empty()) {
                    self.err_handler().span_err(path.span, "type or lifetime parameters \
                                                            in visibility path");
                }
            }
            _ => {}
        }

        visit::walk_vis(self, vis)
    }
}

pub fn check_crate(session: &Session, krate: &Crate) {
    visit::walk_crate(&mut AstValidator { session: session }, krate)
}
