// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use attr::{AttrMetaMethods, HasAttrs};
use feature_gate::{emit_feature_err, EXPLAIN_STMT_ATTR_SYNTAX, Features, get_features, GateIssue};
use fold::Folder;
use {fold, attr};
use ast;
use codemap::{Spanned, respan};
use parse::{ParseSess, token};
use ptr::P;

use util::small_vector::SmallVector;

/// A folder that strips out items that do not belong in the current configuration.
pub struct StripUnconfigured<'a> {
    pub config: &'a ast::CrateConfig,
    pub should_test: bool,
    pub sess: &'a ParseSess,
    pub features: Option<&'a Features>,
}

impl<'a> StripUnconfigured<'a> {
    fn configure<T: HasAttrs>(&mut self, node: T) -> Option<T> {
        let node = self.process_cfg_attrs(node);
        if self.in_cfg(node.attrs()) { Some(node) } else { None }
    }

    fn process_cfg_attrs<T: HasAttrs>(&mut self, node: T) -> T {
        node.map_attrs(|attrs| {
            attrs.into_iter().filter_map(|attr| self.process_cfg_attr(attr)).collect()
        })
    }

    fn process_cfg_attr(&mut self, attr: ast::Attribute) -> Option<ast::Attribute> {
        if !attr.check_name("cfg_attr") {
            return Some(attr);
        }

        let attr_list = match attr.meta_item_list() {
            Some(attr_list) => attr_list,
            None => {
                let msg = "expected `#[cfg_attr(<cfg pattern>, <attr>)]`";
                self.sess.span_diagnostic.span_err(attr.span, msg);
                return None;
            }
        };
        let (cfg, mi) = match (attr_list.len(), attr_list.get(0), attr_list.get(1)) {
            (2, Some(cfg), Some(mi)) => (cfg, mi),
            _ => {
                let msg = "expected `#[cfg_attr(<cfg pattern>, <attr>)]`";
                self.sess.span_diagnostic.span_err(attr.span, msg);
                return None;
            }
        };

        if attr::cfg_matches(self.config, &cfg, self.sess, self.features) {
            self.process_cfg_attr(respan(mi.span, ast::Attribute_ {
                id: attr::mk_attr_id(),
                style: attr.node.style,
                value: mi.clone(),
                is_sugared_doc: false,
            }))
        } else {
            None
        }
    }

    // Determine if a node with the given attributes should be included in this configuation.
    fn in_cfg(&mut self, attrs: &[ast::Attribute]) -> bool {
        attrs.iter().all(|attr| {
            // When not compiling with --test we should not compile the #[test] functions
            if !self.should_test && is_test_or_bench(attr) {
                return false;
            }

            let mis = match attr.node.value.node {
                ast::MetaItemKind::List(_, ref mis) if is_cfg(&attr) => mis,
                _ => return true
            };

            if mis.len() != 1 {
                self.sess.span_diagnostic.span_err(attr.span, "expected 1 cfg-pattern");
                return true;
            }

            attr::cfg_matches(self.config, &mis[0], self.sess, self.features)
        })
    }

    // Visit attributes on expression and statements (but not attributes on items in blocks).
    fn visit_stmt_or_expr_attrs(&mut self, attrs: &[ast::Attribute]) {
        // flag the offending attributes
        for attr in attrs.iter() {
            if !self.features.map(|features| features.stmt_expr_attributes).unwrap_or(true) {
                emit_feature_err(&self.sess.span_diagnostic,
                                 "stmt_expr_attributes",
                                 attr.span,
                                 GateIssue::Language,
                                 EXPLAIN_STMT_ATTR_SYNTAX);
            }
        }
    }
}

// Support conditional compilation by transforming the AST, stripping out
// any items that do not belong in the current configuration
pub fn strip_unconfigured_items(mut krate: ast::Crate, sess: &ParseSess, should_test: bool)
                                -> (ast::Crate, Features) {
    let features;
    {
        let mut strip_unconfigured = StripUnconfigured {
            config: &krate.config.clone(),
            should_test: should_test,
            sess: sess,
            features: None,
        };

        let err_count = sess.span_diagnostic.err_count();
        let krate_attrs = strip_unconfigured.process_cfg_attrs(krate.attrs.clone());
        features = get_features(&sess.span_diagnostic, &krate_attrs);
        if err_count < sess.span_diagnostic.err_count() {
            krate.attrs = krate_attrs.clone(); // Avoid reconfiguring malformed `cfg_attr`s
        }

        strip_unconfigured.features = Some(&features);
        krate = strip_unconfigured.fold_crate(krate);
        krate.attrs = krate_attrs;
    }

    (krate, features)
}

impl<'a> fold::Folder for StripUnconfigured<'a> {
    fn fold_foreign_mod(&mut self, foreign_mod: ast::ForeignMod) -> ast::ForeignMod {
        ast::ForeignMod {
            abi: foreign_mod.abi,
            items: foreign_mod.items.into_iter().filter_map(|item| {
                self.configure(item).map(|item| fold::noop_fold_foreign_item(item, self))
            }).collect(),
        }
    }

    fn fold_item_kind(&mut self, item: ast::ItemKind) -> ast::ItemKind {
        let fold_struct = |this: &mut Self, vdata| match vdata {
            ast::VariantData::Struct(fields, id) => {
                let fields = fields.into_iter().filter_map(|field| this.configure(field));
                ast::VariantData::Struct(fields.collect(), id)
            }
            ast::VariantData::Tuple(fields, id) => {
                let fields = fields.into_iter().filter_map(|field| this.configure(field));
                ast::VariantData::Tuple(fields.collect(), id)
            }
            ast::VariantData::Unit(id) => ast::VariantData::Unit(id)
        };

        let item = match item {
            ast::ItemKind::Struct(def, generics) => {
                ast::ItemKind::Struct(fold_struct(self, def), generics)
            }
            ast::ItemKind::Enum(def, generics) => {
                let variants = def.variants.into_iter().filter_map(|v| {
                    self.configure(v).map(|v| {
                        Spanned {
                            node: ast::Variant_ {
                                name: v.node.name,
                                attrs: v.node.attrs,
                                data: fold_struct(self, v.node.data),
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
        };

        fold::noop_fold_item_kind(item, self)
    }

    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        self.visit_stmt_or_expr_attrs(expr.attrs());

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

        let expr = self.process_cfg_attrs(expr);
        fold_expr(self, expr)
    }

    fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        self.configure(expr).map(|expr| fold_expr(self, expr))
    }

    fn fold_stmt(&mut self, stmt: ast::Stmt) -> SmallVector<ast::Stmt> {
        // avoid calling `visit_stmt_or_expr_attrs` on items
        match stmt.node {
            ast::StmtKind::Item(_) => {}
            _ => self.visit_stmt_or_expr_attrs(stmt.attrs()),
        }

        self.configure(stmt).map(|stmt| fold::noop_fold_stmt(stmt, self))
                            .unwrap_or(SmallVector::zero())
    }

    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        fold::noop_fold_mac(mac, self)
    }

    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        self.configure(item).map(|item| fold::noop_fold_item(item, self))
                            .unwrap_or(SmallVector::zero())
    }

    fn fold_impl_item(&mut self, item: ast::ImplItem) -> SmallVector<ast::ImplItem> {
        self.configure(item).map(|item| fold::noop_fold_impl_item(item, self))
                            .unwrap_or(SmallVector::zero())
    }

    fn fold_trait_item(&mut self, item: ast::TraitItem) -> SmallVector<ast::TraitItem> {
        self.configure(item).map(|item| fold::noop_fold_trait_item(item, self))
                            .unwrap_or(SmallVector::zero())
    }

    fn fold_interpolated(&mut self, nt: token::Nonterminal) -> token::Nonterminal {
        // Don't configure interpolated AST (c.f. #34171).
        // Interpolated AST will get configured once the surrounding tokens are parsed.
        nt
    }
}

fn fold_expr(folder: &mut StripUnconfigured, expr: P<ast::Expr>) -> P<ast::Expr> {
    expr.map(|ast::Expr {id, span, node, attrs}| {
        fold::noop_fold_expr(ast::Expr {
            id: id,
            node: match node {
                ast::ExprKind::Match(m, arms) => {
                    ast::ExprKind::Match(m, arms.into_iter()
                                        .filter_map(|a| folder.configure(a))
                                        .collect())
                }
                _ => node
            },
            span: span,
            attrs: attrs,
        }, folder)
    })
}

fn is_cfg(attr: &ast::Attribute) -> bool {
    attr.check_name("cfg")
}

fn is_test_or_bench(attr: &ast::Attribute) -> bool {
    attr.check_name("test") || attr.check_name("bench")
}
