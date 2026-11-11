use rustc_ast::visit::{self, Visitor};
use rustc_ast::{self as ast, ItemKind};
use rustc_attr_ir::{Attribute, AttributeKind, RustcShowSpansKind};
use rustc_attr_parsing::AttributeParser;
use rustc_session::Session;
use rustc_span::sym;

use crate::diagnostics;

struct ShowSpans<'a> {
    sess: &'a Session,
    expr: bool,
    ty: bool,
    path: bool,
}

impl<'a> Visitor<'a> for ShowSpans<'a> {
    fn visit_item(&mut self, item: &'a ast::Item) {
        let saved = (self.expr, self.ty, self.path);
        (self.expr, self.ty, self.path) = (false, false, false);

        if let ItemKind::Fn(_) = &item.kind
            && let Some(Attribute::Parsed(AttributeKind::RustcShowSpans(kinds))) =
                AttributeParser::parse_limited_sym(self.sess, &item.attrs, &[sym::rustc_show_spans])
        {
            for kind in kinds {
                match kind {
                    RustcShowSpansKind::Expr => self.expr = true,
                    RustcShowSpansKind::Ty => self.ty = true,
                    RustcShowSpansKind::Path => self.path = true,
                }
            }
        }
        visit::walk_item(self, item);

        // Recover the state. Needed if we are looking into nested items.
        (self.expr, self.ty, self.path) = saved;
    }

    // We override it as to not report attribute paths as paths.
    fn visit_attribute(&mut self, _: &'a ast::Attribute) {}

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        if self.expr {
            self.sess.dcx().emit_note(diagnostics::ShowSpan { span: e.span, kind: "expr" });
        }
        visit::walk_expr(self, e);
    }

    fn visit_ty(&mut self, t: &'a ast::Ty) {
        if self.ty {
            self.sess.dcx().emit_note(diagnostics::ShowSpan { span: t.span, kind: "ty" });
        }
        visit::walk_ty(self, t);
    }

    fn visit_path(&mut self, p: &'a ast::Path) {
        if self.path {
            self.sess.dcx().emit_note(diagnostics::ShowSpan { span: p.span, kind: "path" });
        }
        visit::walk_path(self, p);
    }
}

pub fn check_crate(sess: &Session, krate: &ast::Crate) {
    visit::walk_crate(&mut ShowSpans { sess, expr: false, ty: false, path: false }, krate);
}
