use std::mem;

use rustc_ast::visit::{self, Visitor};
use rustc_ast::{self as ast, ItemKind};
use rustc_attr_ir::{Attribute, AttributeKind, RustcShowSpansKind};
use rustc_attr_parsing::AttributeParser;
use rustc_session::Session;
use rustc_span::sym;

use crate::diagnostics;

struct ShowSpans<'a> {
    sess: &'a Session,
    kinds: ShowSpanKinds,
}

#[derive(Clone, Copy, Default)]
struct ShowSpanKinds {
    expr: bool,
    ty: bool,
    path: bool,
    generic_args: bool,
}

impl<'a> Visitor<'a> for ShowSpans<'a> {
    fn visit_item(&mut self, item: &'a ast::Item) {
        let saved = mem::take(&mut self.kinds);

        if let ItemKind::Fn(_) = &item.kind
            && let Some(Attribute::Parsed(AttributeKind::RustcShowSpans(kinds))) =
                AttributeParser::parse_limited_sym(self.sess, &item.attrs, &[sym::rustc_show_spans])
        {
            for kind in kinds {
                match kind {
                    RustcShowSpansKind::Expr => self.kinds.expr = true,
                    RustcShowSpansKind::Ty => self.kinds.ty = true,
                    RustcShowSpansKind::Path => self.kinds.path = true,
                    RustcShowSpansKind::GenericArgs => self.kinds.generic_args = true,
                }
            }
        }
        visit::walk_item(self, item);

        // Recover the state. Needed if we are looking into nested items.
        self.kinds = saved;
    }

    // We override it as to not report attribute paths as paths.
    fn visit_attribute(&mut self, _: &'a ast::Attribute) {}

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        if self.kinds.expr {
            self.sess.dcx().emit_note(diagnostics::ShowSpan { span: e.span, kind: "expr" });
        }
        visit::walk_expr(self, e);
    }

    fn visit_ty(&mut self, t: &'a ast::Ty) {
        if self.kinds.ty {
            self.sess.dcx().emit_note(diagnostics::ShowSpan { span: t.span, kind: "ty" });
        }
        visit::walk_ty(self, t);
    }

    fn visit_path(&mut self, p: &'a ast::Path) {
        if self.kinds.path {
            self.sess.dcx().emit_note(diagnostics::ShowSpan { span: p.span, kind: "path" });
        }
        visit::walk_path(self, p);
    }

    fn visit_generic_args(&mut self, g: &'a ast::GenericArgs) {
        if self.kinds.generic_args {
            self.sess
                .dcx()
                .emit_note(diagnostics::ShowSpan { span: g.span(), kind: "generic_args" });
        }
        visit::walk_generic_args(self, g);
    }
}

pub fn check_crate(sess: &Session, krate: &ast::Crate) {
    visit::walk_crate(&mut ShowSpans { sess, kinds: ShowSpanKinds::default() }, krate);
}
