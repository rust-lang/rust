// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use attr::HasAttrs;
use feature_gate::{feature_err, EXPLAIN_STMT_ATTR_SYNTAX, Features, get_features, GateIssue};
use {fold, attr};
use ast;
use source_map::Spanned;
use edition::Edition;
use parse::{token, ParseSess};
use OneVector;

use ptr::P;

/// A folder that strips out items that do not belong in the current configuration.
pub struct StripUnconfigured<'a> {
    pub should_test: bool,
    pub sess: &'a ParseSess,
    pub features: Option<&'a Features>,
}

// `cfg_attr`-process the crate's attributes and compute the crate's features.
pub fn features(mut krate: ast::Crate, sess: &ParseSess, should_test: bool, edition: Edition)
                -> (ast::Crate, Features) {
    let features;
    {
        let mut strip_unconfigured = StripUnconfigured {
            should_test,
            sess,
            features: None,
        };

        let unconfigured_attrs = krate.attrs.clone();
        let err_count = sess.span_diagnostic.err_count();
        if let Some(attrs) = strip_unconfigured.configure(krate.attrs) {
            krate.attrs = attrs;
        } else { // the entire crate is unconfigured
            krate.attrs = Vec::new();
            krate.module.items = Vec::new();
            return (krate, Features::new());
        }

        features = get_features(&sess.span_diagnostic, &krate.attrs, edition);

        // Avoid reconfiguring malformed `cfg_attr`s
        if err_count == sess.span_diagnostic.err_count() {
            strip_unconfigured.features = Some(&features);
            strip_unconfigured.configure(unconfigured_attrs);
        }
    }

    (krate, features)
}

macro_rules! configure {
    ($this:ident, $node:ident) => {
        match $this.configure($node) {
            Some(node) => node,
            None => return Default::default(),
        }
    }
}

impl<'a> StripUnconfigured<'a> {
    pub fn configure<T: HasAttrs>(&mut self, node: T) -> Option<T> {
        let node = self.process_cfg_attrs(node);
        if self.in_cfg(node.attrs()) { Some(node) } else { None }
    }

    pub fn process_cfg_attrs<T: HasAttrs>(&mut self, node: T) -> T {
        node.map_attrs(|attrs| {
            attrs.into_iter().filter_map(|attr| self.process_cfg_attr(attr)).collect()
        })
    }

    fn process_cfg_attr(&mut self, attr: ast::Attribute) -> Option<ast::Attribute> {
        if !attr.check_name("cfg_attr") {
            return Some(attr);
        }

        let (cfg, path, tokens, span) = match attr.parse(self.sess, |parser| {
            parser.expect(&token::OpenDelim(token::Paren))?;
            let cfg = parser.parse_meta_item()?;
            parser.expect(&token::Comma)?;
            let lo = parser.span.lo();
            let (path, tokens) = parser.parse_meta_item_unrestricted()?;
            parser.expect(&token::CloseDelim(token::Paren))?;
            Ok((cfg, path, tokens, parser.prev_span.with_lo(lo)))
        }) {
            Ok(result) => result,
            Err(mut e) => {
                e.emit();
                return None;
            }
        };

        if attr::cfg_matches(&cfg, self.sess, self.features) {
            self.process_cfg_attr(ast::Attribute {
                id: attr::mk_attr_id(),
                style: attr.style,
                path,
                tokens,
                is_sugared_doc: false,
                span,
            })
        } else {
            None
        }
    }

    // Determine if a node with the given attributes should be included in this configuration.
    pub fn in_cfg(&mut self, attrs: &[ast::Attribute]) -> bool {
        attrs.iter().all(|attr| {
            // When not compiling with --test we should not compile the #[test] functions
            if !self.should_test && is_test_or_bench(attr) {
                return false;
            }

            let mis = if !is_cfg(attr) {
                return true;
            } else if let Some(mis) = attr.meta_item_list() {
                mis
            } else {
                return true;
            };

            if mis.len() != 1 {
                self.sess.span_diagnostic.span_err(attr.span, "expected 1 cfg-pattern");
                return true;
            }

            if !mis[0].is_meta_item() {
                self.sess.span_diagnostic.span_err(mis[0].span, "unexpected literal");
                return true;
            }

            attr::cfg_matches(mis[0].meta_item().unwrap(), self.sess, self.features)
        })
    }

    // Visit attributes on expression and statements (but not attributes on items in blocks).
    fn visit_expr_attrs(&mut self, attrs: &[ast::Attribute]) {
        // flag the offending attributes
        for attr in attrs.iter() {
            self.maybe_emit_expr_attr_err(attr);
        }
    }

    /// If attributes are not allowed on expressions, emit an error for `attr`
    pub fn maybe_emit_expr_attr_err(&self, attr: &ast::Attribute) {
        if !self.features.map(|features| features.stmt_expr_attributes).unwrap_or(true) {
            let mut err = feature_err(self.sess,
                                      "stmt_expr_attributes",
                                      attr.span,
                                      GateIssue::Language,
                                      EXPLAIN_STMT_ATTR_SYNTAX);

            if attr.is_sugared_doc {
                err.help("`///` is for documentation comments. For a plain comment, use `//`.");
            }

            err.emit();
        }
    }

    pub fn configure_foreign_mod(&mut self, foreign_mod: ast::ForeignMod) -> ast::ForeignMod {
        ast::ForeignMod {
            abi: foreign_mod.abi,
            items: foreign_mod.items.into_iter().filter_map(|item| self.configure(item)).collect(),
        }
    }

    fn configure_variant_data(&mut self, vdata: ast::VariantData) -> ast::VariantData {
        match vdata {
            ast::VariantData::Struct(fields, id) => {
                let fields = fields.into_iter().filter_map(|field| self.configure(field));
                ast::VariantData::Struct(fields.collect(), id)
            }
            ast::VariantData::Tuple(fields, id) => {
                let fields = fields.into_iter().filter_map(|field| self.configure(field));
                ast::VariantData::Tuple(fields.collect(), id)
            }
            ast::VariantData::Unit(id) => ast::VariantData::Unit(id)
        }
    }

    pub fn configure_item_kind(&mut self, item: ast::ItemKind) -> ast::ItemKind {
        match item {
            ast::ItemKind::Struct(def, generics) => {
                ast::ItemKind::Struct(self.configure_variant_data(def), generics)
            }
            ast::ItemKind::Union(def, generics) => {
                ast::ItemKind::Union(self.configure_variant_data(def), generics)
            }
            ast::ItemKind::Enum(def, generics) => {
                let variants = def.variants.into_iter().filter_map(|v| {
                    self.configure(v).map(|v| {
                        Spanned {
                            node: ast::Variant_ {
                                ident: v.node.ident,
                                attrs: v.node.attrs,
                                data: self.configure_variant_data(v.node.data),
                                disr_expr: v.node.disr_expr,
                            },
                            span: v.span
                        }
                    })
                });
                ast::ItemKind::Enum(ast::EnumDef {
                    variants: variants.collect(),
                }, generics)
            }
            item => item,
        }
    }

    pub fn configure_expr_kind(&mut self, expr_kind: ast::ExprKind) -> ast::ExprKind {
        match expr_kind {
            ast::ExprKind::Match(m, arms) => {
                let arms = arms.into_iter().filter_map(|a| self.configure(a)).collect();
                ast::ExprKind::Match(m, arms)
            }
            ast::ExprKind::Struct(path, fields, base) => {
                let fields = fields.into_iter()
                    .filter_map(|field| {
                        self.configure(field)
                    })
                    .collect();
                ast::ExprKind::Struct(path, fields, base)
            }
            _ => expr_kind,
        }
    }

    pub fn configure_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        self.visit_expr_attrs(expr.attrs());

        // If an expr is valid to cfg away it will have been removed by the
        // outer stmt or expression folder before descending in here.
        // Anything else is always required, and thus has to error out
        // in case of a cfg attr.
        //
        // NB: This is intentionally not part of the fold_expr() function
        //     in order for fold_opt_expr() to be able to avoid this check
        if let Some(attr) = expr.attrs().iter().find(|a| is_cfg(a) || is_test_or_bench(a)) {
            let msg = "removing an expression is not supported in this position";
            self.sess.span_diagnostic.span_err(attr.span, msg);
        }

        self.process_cfg_attrs(expr)
    }

    pub fn configure_stmt(&mut self, stmt: ast::Stmt) -> Option<ast::Stmt> {
        self.configure(stmt)
    }

    pub fn configure_struct_expr_field(&mut self, field: ast::Field) -> Option<ast::Field> {
        self.configure(field)
    }

    pub fn configure_pat(&mut self, pattern: P<ast::Pat>) -> P<ast::Pat> {
        pattern.map(|mut pattern| {
            if let ast::PatKind::Struct(path, fields, etc) = pattern.node {
                let fields = fields.into_iter()
                    .filter_map(|field| {
                        self.configure(field)
                    })
                    .collect();
                pattern.node = ast::PatKind::Struct(path, fields, etc);
            }
            pattern
        })
    }

    // deny #[cfg] on generic parameters until we decide what to do with it.
    // see issue #51279.
    pub fn disallow_cfg_on_generic_param(&mut self, param: &ast::GenericParam) {
        for attr in param.attrs() {
            let offending_attr = if attr.check_name("cfg") {
                "cfg"
            } else if attr.check_name("cfg_attr") {
                "cfg_attr"
            } else {
                continue;
            };
            let msg = format!("#[{}] cannot be applied on a generic parameter", offending_attr);
            self.sess.span_diagnostic.span_err(attr.span, &msg);
        }
    }
}

impl<'a> fold::Folder for StripUnconfigured<'a> {
    fn fold_foreign_mod(&mut self, foreign_mod: ast::ForeignMod) -> ast::ForeignMod {
        let foreign_mod = self.configure_foreign_mod(foreign_mod);
        fold::noop_fold_foreign_mod(foreign_mod, self)
    }

    fn fold_item_kind(&mut self, item: ast::ItemKind) -> ast::ItemKind {
        let item = self.configure_item_kind(item);
        fold::noop_fold_item_kind(item, self)
    }

    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        let mut expr = self.configure_expr(expr).into_inner();
        expr.node = self.configure_expr_kind(expr.node);
        P(fold::noop_fold_expr(expr, self))
    }

    fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let mut expr = configure!(self, expr).into_inner();
        expr.node = self.configure_expr_kind(expr.node);
        Some(P(fold::noop_fold_expr(expr, self)))
    }

    fn fold_stmt(&mut self, stmt: ast::Stmt) -> OneVector<ast::Stmt> {
        match self.configure_stmt(stmt) {
            Some(stmt) => fold::noop_fold_stmt(stmt, self),
            None => return OneVector::new(),
        }
    }

    fn fold_item(&mut self, item: P<ast::Item>) -> OneVector<P<ast::Item>> {
        fold::noop_fold_item(configure!(self, item), self)
    }

    fn fold_impl_item(&mut self, item: ast::ImplItem) -> OneVector<ast::ImplItem> {
        fold::noop_fold_impl_item(configure!(self, item), self)
    }

    fn fold_trait_item(&mut self, item: ast::TraitItem) -> OneVector<ast::TraitItem> {
        fold::noop_fold_trait_item(configure!(self, item), self)
    }

    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        // Don't configure interpolated AST (c.f. #34171).
        // Interpolated AST will get configured once the surrounding tokens are parsed.
        mac
    }

    fn fold_pat(&mut self, pattern: P<ast::Pat>) -> P<ast::Pat> {
        fold::noop_fold_pat(self.configure_pat(pattern), self)
    }
}

fn is_cfg(attr: &ast::Attribute) -> bool {
    attr.check_name("cfg")
}

pub fn is_test_or_bench(attr: &ast::Attribute) -> bool {
    attr.check_name("test") || attr.check_name("bench")
}
