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
use syntax::attr;
use syntax::codemap::Spanned;
use syntax::parse::token;
use syntax::symbol::keywords;
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
        if label.name == "'_" {
            self.session.add_lint(lint::builtin::LIFETIME_UNDERSCORE,
                                  id,
                                  span,
                                  format!("invalid label name `{}`", label.name));
        }
    }

    fn invalid_visibility(&self, vis: &Visibility, span: Span, note: Option<&str>) {
        if vis != &Visibility::Inherited {
            let mut err = struct_span_err!(self.session,
                                           span,
                                           E0449,
                                           "unnecessary visibility qualifier");
            if vis == &Visibility::Public {
                err.span_label(span, &format!("`pub` not needed here"));
            }
            if let Some(note) = note {
                err.note(note);
            }
            err.emit();
        }
    }

    fn check_decl_no_pat<ReportFn: Fn(Span, bool)>(&self, decl: &FnDecl, report_err: ReportFn) {
        for arg in &decl.inputs {
            match arg.pat.node {
                PatKind::Ident(BindingMode::ByValue(Mutability::Immutable), _, None) |
                PatKind::Wild => {}
                PatKind::Ident(..) => report_err(arg.pat.span, true),
                _ => report_err(arg.pat.span, false),
            }
        }
    }

    fn check_trait_fn_not_const(&self, constness: Spanned<Constness>) {
        match constness.node {
            Constness::Const => {
                struct_span_err!(self.session, constness.span, E0379,
                                 "trait fns cannot be declared const")
                    .span_label(constness.span, &format!("trait fns cannot be const"))
                    .emit();
            }
            _ => {}
        }
    }

    fn no_questions_in_bounds(&self, bounds: &TyParamBounds, where_: &str, is_trait: bool) {
        for bound in bounds {
            if let TraitTyParamBound(ref poly, TraitBoundModifier::Maybe) = *bound {
                let mut err = self.err_handler().struct_span_err(poly.span,
                                    &format!("`?Trait` is not permitted in {}", where_));
                if is_trait {
                    err.note(&format!("traits are `?{}` by default", poly.trait_ref.path));
                }
                err.emit();
            }
        }
    }
}

impl<'a> Visitor<'a> for AstValidator<'a> {
    fn visit_lifetime(&mut self, lt: &'a Lifetime) {
        if lt.name == "'_" {
            self.session.add_lint(lint::builtin::LIFETIME_UNDERSCORE,
                                  lt.id,
                                  lt.span,
                                  format!("invalid lifetime name `{}`", lt.name));
        }

        visit::walk_lifetime(self, lt)
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr.node {
            ExprKind::While(.., Some(ident)) |
            ExprKind::Loop(_, Some(ident)) |
            ExprKind::WhileLet(.., Some(ident)) |
            ExprKind::ForLoop(.., Some(ident)) |
            ExprKind::Break(Some(ident), _) |
            ExprKind::Continue(Some(ident)) => {
                self.check_label(ident.node, ident.span, expr.id);
            }
            _ => {}
        }

        visit::walk_expr(self, expr)
    }

    fn visit_ty(&mut self, ty: &'a Ty) {
        match ty.node {
            TyKind::BareFn(ref bfty) => {
                self.check_decl_no_pat(&bfty.decl, |span, _| {
                    let mut err = struct_span_err!(self.session,
                                                   span,
                                                   E0561,
                                                   "patterns aren't allowed in function pointer \
                                                    types");
                    err.span_note(span,
                                  "this is a recent error, see issue #35203 for more details");
                    err.emit();
                });
            }
            TyKind::TraitObject(ref bounds) => {
                self.no_questions_in_bounds(bounds, "trait object types", false);
            }
            _ => {}
        }

        visit::walk_ty(self, ty)
    }

    fn visit_path(&mut self, path: &'a Path, id: NodeId) {
        if path.segments.len() >= 2 && path.is_global() {
            let ident = path.segments[1].identifier;
            if token::Ident(ident).is_path_segment_keyword() {
                self.session.add_lint(lint::builtin::SUPER_OR_SELF_IN_GLOBAL_PATH,
                                      id,
                                      path.span,
                                      format!("global paths cannot start with `{}`", ident));
            }
        }

        visit::walk_path(self, path)
    }

    fn visit_item(&mut self, item: &'a Item) {
        match item.node {
            ItemKind::Use(ref view_path) => {
                let path = view_path.node.path();
                if path.segments.iter().any(|segment| segment.parameters.is_some()) {
                    self.err_handler()
                        .span_err(path.span, "type or lifetime parameters in import path");
                }
            }
            ItemKind::Impl(.., Some(..), _, ref impl_items) => {
                self.invalid_visibility(&item.vis, item.span, None);
                for impl_item in impl_items {
                    self.invalid_visibility(&impl_item.vis, impl_item.span, None);
                    if let ImplItemKind::Method(ref sig, _) = impl_item.node {
                        self.check_trait_fn_not_const(sig.constness);
                    }
                }
            }
            ItemKind::Impl(.., None, _, _) => {
                self.invalid_visibility(&item.vis,
                                        item.span,
                                        Some("place qualifiers on individual impl items instead"));
            }
            ItemKind::DefaultImpl(..) => {
                self.invalid_visibility(&item.vis, item.span, None);
            }
            ItemKind::ForeignMod(..) => {
                self.invalid_visibility(&item.vis,
                                        item.span,
                                        Some("place qualifiers on individual foreign items \
                                              instead"));
            }
            ItemKind::Enum(ref def, _) => {
                for variant in &def.variants {
                    for field in variant.node.data.fields() {
                        self.invalid_visibility(&field.vis, field.span, None);
                    }
                }
            }
            ItemKind::Trait(.., ref bounds, ref trait_items) => {
                self.no_questions_in_bounds(bounds, "supertraits", true);
                for trait_item in trait_items {
                    if let TraitItemKind::Method(ref sig, ref block) = trait_item.node {
                        self.check_trait_fn_not_const(sig.constness);
                        if block.is_none() {
                            self.check_decl_no_pat(&sig.decl, |span, _| {
                                self.session.add_lint(lint::builtin::PATTERNS_IN_FNS_WITHOUT_BODY,
                                                      trait_item.id, span,
                                                      "patterns aren't allowed in methods \
                                                       without bodies".to_string());
                            });
                        }
                    }
                }
            }
            ItemKind::Mod(_) => {
                // Ensure that `path` attributes on modules are recorded as used (c.f. #35584).
                attr::first_attr_value_str_by_name(&item.attrs, "path");
                if let Some(attr) =
                        item.attrs.iter().find(|attr| attr.name() == "warn_directory_ownership") {
                    let lint = lint::builtin::LEGACY_DIRECTORY_OWNERSHIP;
                    let msg = "cannot declare a new module at this location";
                    self.session.add_lint(lint, item.id, item.span, msg.to_string());
                    attr::mark_used(attr);
                }
            }
            ItemKind::Union(ref vdata, _) => {
                if !vdata.is_struct() {
                    self.err_handler().span_err(item.span,
                                                "tuple and unit unions are not permitted");
                }
                if vdata.fields().len() == 0 {
                    self.err_handler().span_err(item.span,
                                                "unions cannot have zero fields");
                }
            }
            _ => {}
        }

        visit::walk_item(self, item)
    }

    fn visit_foreign_item(&mut self, fi: &'a ForeignItem) {
        match fi.node {
            ForeignItemKind::Fn(ref decl, _) => {
                self.check_decl_no_pat(decl, |span, is_recent| {
                    let mut err = struct_span_err!(self.session,
                                                   span,
                                                   E0130,
                                                   "patterns aren't allowed in foreign function \
                                                    declarations");
                    err.span_label(span, &format!("pattern not allowed in foreign function"));
                    if is_recent {
                        err.span_note(span,
                                      "this is a recent error, see issue #35203 for more details");
                    }
                    err.emit();
                });
            }
            ForeignItemKind::Static(..) => {}
        }

        visit::walk_foreign_item(self, fi)
    }

    fn visit_vis(&mut self, vis: &'a Visibility) {
        match *vis {
            Visibility::Restricted { ref path, .. } => {
                if !path.segments.iter().all(|segment| segment.parameters.is_none()) {
                    self.err_handler()
                        .span_err(path.span, "type or lifetime parameters in visibility path");
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
