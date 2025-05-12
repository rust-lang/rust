use core::ops::ControlFlow;

use rustc_ast as ast;
use rustc_ast::mut_visit::MutVisitor;
use rustc_ast::ptr::P;
use rustc_ast::visit::{AssocCtxt, Visitor};
use rustc_ast::{Attribute, HasAttrs, HasTokens, NodeId, mut_visit, visit};
use rustc_errors::PResult;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_expand::config::StripUnconfigured;
use rustc_expand::configure;
use rustc_feature::Features;
use rustc_parse::parser::{ForceCollect, Parser};
use rustc_session::Session;
use rustc_span::{Span, sym};
use smallvec::SmallVec;
use tracing::instrument;

use crate::util::{check_builtin_macro_attribute, warn_on_duplicate_attribute};

pub(crate) fn expand(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    annotatable: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::cfg_eval);
    warn_on_duplicate_attribute(ecx, &annotatable, sym::cfg_eval);
    vec![cfg_eval(ecx.sess, ecx.ecfg.features, annotatable, ecx.current_expansion.lint_node_id)]
}

pub(crate) fn cfg_eval(
    sess: &Session,
    features: &Features,
    annotatable: Annotatable,
    lint_node_id: NodeId,
) -> Annotatable {
    let features = Some(features);
    CfgEval(StripUnconfigured { sess, features, config_tokens: true, lint_node_id })
        .configure_annotatable(annotatable)
}

struct CfgEval<'a>(StripUnconfigured<'a>);

fn has_cfg_or_cfg_attr(annotatable: &Annotatable) -> bool {
    struct CfgFinder;

    impl<'ast> visit::Visitor<'ast> for CfgFinder {
        type Result = ControlFlow<()>;
        fn visit_attribute(&mut self, attr: &'ast Attribute) -> ControlFlow<()> {
            if attr
                .ident()
                .is_some_and(|ident| ident.name == sym::cfg || ident.name == sym::cfg_attr)
            {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }
    }

    let res = match annotatable {
        Annotatable::Item(item) => CfgFinder.visit_item(item),
        Annotatable::AssocItem(item, ctxt) => CfgFinder.visit_assoc_item(item, *ctxt),
        Annotatable::ForeignItem(item) => CfgFinder.visit_foreign_item(item),
        Annotatable::Stmt(stmt) => CfgFinder.visit_stmt(stmt),
        Annotatable::Expr(expr) => CfgFinder.visit_expr(expr),
        _ => unreachable!(),
    };
    res.is_break()
}

impl CfgEval<'_> {
    fn configure<T: HasAttrs + HasTokens>(&mut self, node: T) -> Option<T> {
        self.0.configure(node)
    }

    fn configure_annotatable(mut self, annotatable: Annotatable) -> Annotatable {
        // Tokenizing and re-parsing the `Annotatable` can have a significant
        // performance impact, so try to avoid it if possible
        if !has_cfg_or_cfg_attr(&annotatable) {
            return annotatable;
        }

        // The majority of parsed attribute targets will never need to have early cfg-expansion
        // run (e.g. they are not part of a `#[derive]` or `#[cfg_eval]` macro input).
        // Therefore, we normally do not capture the necessary information about `#[cfg]`
        // and `#[cfg_attr]` attributes during parsing.
        //
        // Therefore, when we actually *do* run early cfg-expansion, we need to tokenize
        // and re-parse the attribute target, this time capturing information about
        // the location of `#[cfg]` and `#[cfg_attr]` in the token stream. The tokenization
        // process is lossless, so this process is invisible to proc-macros.

        // Interesting cases:
        //
        // ```rust
        // #[cfg_eval] #[cfg] $item
        //```
        //
        // where `$item` is `#[cfg_attr] struct Foo {}`. We want to make
        // sure to evaluate *all* `#[cfg]` and `#[cfg_attr]` attributes - the simplest
        // way to do this is to do a single parse of the token stream.
        let orig_tokens = annotatable.to_tokens();

        // Re-parse the tokens, setting the `capture_cfg` flag to save extra information
        // to the captured `AttrTokenStream` (specifically, we capture
        // `AttrTokenTree::AttrsTarget` for all occurrences of `#[cfg]` and `#[cfg_attr]`)
        //
        // After that we have our re-parsed `AttrTokenStream`, recursively configuring
        // our attribute target will correctly configure the tokens as well.
        let mut parser = Parser::new(&self.0.sess.psess, orig_tokens, None);
        parser.capture_cfg = true;
        let res: PResult<'_, Annotatable> = try {
            match annotatable {
                Annotatable::Item(_) => {
                    let item = parser.parse_item(ForceCollect::Yes)?.unwrap();
                    Annotatable::Item(self.flat_map_item(item).pop().unwrap())
                }
                Annotatable::AssocItem(_, ctxt) => {
                    let item = parser.parse_trait_item(ForceCollect::Yes)?.unwrap().unwrap();
                    Annotatable::AssocItem(
                        self.flat_map_assoc_item(item, ctxt).pop().unwrap(),
                        ctxt,
                    )
                }
                Annotatable::ForeignItem(_) => {
                    let item = parser.parse_foreign_item(ForceCollect::Yes)?.unwrap().unwrap();
                    Annotatable::ForeignItem(self.flat_map_foreign_item(item).pop().unwrap())
                }
                Annotatable::Stmt(_) => {
                    let stmt = parser
                        .parse_stmt_without_recovery(false, ForceCollect::Yes, false)?
                        .unwrap();
                    Annotatable::Stmt(P(self.flat_map_stmt(stmt).pop().unwrap()))
                }
                Annotatable::Expr(_) => {
                    let mut expr = parser.parse_expr_force_collect()?;
                    self.visit_expr(&mut expr);
                    Annotatable::Expr(expr)
                }
                _ => unreachable!(),
            }
        };

        match res {
            Ok(ann) => ann,
            Err(err) => {
                err.emit();
                annotatable
            }
        }
    }
}

impl MutVisitor for CfgEval<'_> {
    #[instrument(level = "trace", skip(self))]
    fn visit_expr(&mut self, expr: &mut P<ast::Expr>) {
        self.0.configure_expr(expr, false);
        mut_visit::walk_expr(self, expr);
    }

    #[instrument(level = "trace", skip(self))]
    fn visit_method_receiver_expr(&mut self, expr: &mut P<ast::Expr>) {
        self.0.configure_expr(expr, true);
        mut_visit::walk_expr(self, expr);
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let mut expr = configure!(self, expr);
        mut_visit::walk_expr(self, &mut expr);
        Some(expr)
    }

    fn flat_map_generic_param(
        &mut self,
        param: ast::GenericParam,
    ) -> SmallVec<[ast::GenericParam; 1]> {
        let param = configure!(self, param);
        mut_visit::walk_flat_map_generic_param(self, param)
    }

    fn flat_map_stmt(&mut self, stmt: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        let stmt = configure!(self, stmt);
        mut_visit::walk_flat_map_stmt(self, stmt)
    }

    fn flat_map_item(&mut self, item: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        let item = configure!(self, item);
        mut_visit::walk_flat_map_item(self, item)
    }

    fn flat_map_assoc_item(
        &mut self,
        item: P<ast::AssocItem>,
        ctxt: AssocCtxt,
    ) -> SmallVec<[P<ast::AssocItem>; 1]> {
        let item = configure!(self, item);
        mut_visit::walk_flat_map_assoc_item(self, item, ctxt)
    }

    fn flat_map_foreign_item(
        &mut self,
        foreign_item: P<ast::ForeignItem>,
    ) -> SmallVec<[P<ast::ForeignItem>; 1]> {
        let foreign_item = configure!(self, foreign_item);
        mut_visit::walk_flat_map_foreign_item(self, foreign_item)
    }

    fn flat_map_arm(&mut self, arm: ast::Arm) -> SmallVec<[ast::Arm; 1]> {
        let arm = configure!(self, arm);
        mut_visit::walk_flat_map_arm(self, arm)
    }

    fn flat_map_expr_field(&mut self, field: ast::ExprField) -> SmallVec<[ast::ExprField; 1]> {
        let field = configure!(self, field);
        mut_visit::walk_flat_map_expr_field(self, field)
    }

    fn flat_map_pat_field(&mut self, fp: ast::PatField) -> SmallVec<[ast::PatField; 1]> {
        let fp = configure!(self, fp);
        mut_visit::walk_flat_map_pat_field(self, fp)
    }

    fn flat_map_param(&mut self, p: ast::Param) -> SmallVec<[ast::Param; 1]> {
        let p = configure!(self, p);
        mut_visit::walk_flat_map_param(self, p)
    }

    fn flat_map_field_def(&mut self, sf: ast::FieldDef) -> SmallVec<[ast::FieldDef; 1]> {
        let sf = configure!(self, sf);
        mut_visit::walk_flat_map_field_def(self, sf)
    }

    fn flat_map_variant(&mut self, variant: ast::Variant) -> SmallVec<[ast::Variant; 1]> {
        let variant = configure!(self, variant);
        mut_visit::walk_flat_map_variant(self, variant)
    }
}
