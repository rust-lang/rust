use crate::util::{check_builtin_macro_attribute, warn_on_duplicate_attribute};

use core::ops::ControlFlow;
use rustc_ast as ast;
use rustc_ast::mut_visit::MutVisitor;
use rustc_ast::ptr::P;
use rustc_ast::visit::{AssocCtxt, Visitor};
use rustc_ast::NodeId;
use rustc_ast::{mut_visit, visit};
use rustc_ast::{Attribute, HasAttrs, HasTokens};
use rustc_errors::PResult;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_expand::config::StripUnconfigured;
use rustc_expand::configure;
use rustc_feature::Features;
use rustc_parse::parser::{ForceCollect, Parser};
use rustc_session::Session;
use rustc_span::symbol::sym;
use rustc_span::Span;
use smallvec::SmallVec;
use tracing::instrument;

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
        // Since the item itself has already been configured by the `InvocationCollector`,
        // we know that fold result vector will contain exactly one element.
        .unwrap()
}

struct CfgEval<'a>(StripUnconfigured<'a>);

fn flat_map_annotatable(
    vis: &mut impl MutVisitor,
    annotatable: Annotatable,
) -> Option<Annotatable> {
    match annotatable {
        Annotatable::Item(item) => vis.flat_map_item(item).pop().map(Annotatable::Item),
        Annotatable::AssocItem(item, ctxt) => {
            Some(Annotatable::AssocItem(vis.flat_map_assoc_item(item, ctxt).pop()?, ctxt))
        }
        Annotatable::ForeignItem(item) => {
            vis.flat_map_foreign_item(item).pop().map(Annotatable::ForeignItem)
        }
        Annotatable::Stmt(stmt) => {
            vis.flat_map_stmt(stmt.into_inner()).pop().map(P).map(Annotatable::Stmt)
        }
        Annotatable::Expr(mut expr) => {
            vis.visit_expr(&mut expr);
            Some(Annotatable::Expr(expr))
        }
        Annotatable::Arm(arm) => vis.flat_map_arm(arm).pop().map(Annotatable::Arm),
        Annotatable::ExprField(field) => {
            vis.flat_map_expr_field(field).pop().map(Annotatable::ExprField)
        }
        Annotatable::PatField(fp) => vis.flat_map_pat_field(fp).pop().map(Annotatable::PatField),
        Annotatable::GenericParam(param) => {
            vis.flat_map_generic_param(param).pop().map(Annotatable::GenericParam)
        }
        Annotatable::Param(param) => vis.flat_map_param(param).pop().map(Annotatable::Param),
        Annotatable::FieldDef(sf) => vis.flat_map_field_def(sf).pop().map(Annotatable::FieldDef),
        Annotatable::Variant(v) => vis.flat_map_variant(v).pop().map(Annotatable::Variant),
        Annotatable::Crate(mut krate) => {
            vis.visit_crate(&mut krate);
            Some(Annotatable::Crate(krate))
        }
    }
}

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
        Annotatable::Arm(arm) => CfgFinder.visit_arm(arm),
        Annotatable::ExprField(field) => CfgFinder.visit_expr_field(field),
        Annotatable::PatField(field) => CfgFinder.visit_pat_field(field),
        Annotatable::GenericParam(param) => CfgFinder.visit_generic_param(param),
        Annotatable::Param(param) => CfgFinder.visit_param(param),
        Annotatable::FieldDef(field) => CfgFinder.visit_field_def(field),
        Annotatable::Variant(variant) => CfgFinder.visit_variant(variant),
        Annotatable::Crate(krate) => CfgFinder.visit_crate(krate),
    };
    res.is_break()
}

impl CfgEval<'_> {
    fn configure<T: HasAttrs + HasTokens>(&mut self, node: T) -> Option<T> {
        self.0.configure(node)
    }

    fn configure_annotatable(&mut self, mut annotatable: Annotatable) -> Option<Annotatable> {
        // Tokenizing and re-parsing the `Annotatable` can have a significant
        // performance impact, so try to avoid it if possible
        if !has_cfg_or_cfg_attr(&annotatable) {
            return Some(annotatable);
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

        let parse_annotatable_with: for<'a> fn(&mut Parser<'a>) -> PResult<'a, _> =
            match annotatable {
                Annotatable::Item(_) => {
                    |parser| Ok(Annotatable::Item(parser.parse_item(ForceCollect::Yes)?.unwrap()))
                }
                Annotatable::AssocItem(_, AssocCtxt::Trait) => |parser| {
                    Ok(Annotatable::AssocItem(
                        parser.parse_trait_item(ForceCollect::Yes)?.unwrap().unwrap(),
                        AssocCtxt::Trait,
                    ))
                },
                Annotatable::AssocItem(_, AssocCtxt::Impl) => |parser| {
                    Ok(Annotatable::AssocItem(
                        parser.parse_impl_item(ForceCollect::Yes)?.unwrap().unwrap(),
                        AssocCtxt::Impl,
                    ))
                },
                Annotatable::ForeignItem(_) => |parser| {
                    Ok(Annotatable::ForeignItem(
                        parser.parse_foreign_item(ForceCollect::Yes)?.unwrap().unwrap(),
                    ))
                },
                Annotatable::Stmt(_) => |parser| {
                    Ok(Annotatable::Stmt(P(parser
                        .parse_stmt_without_recovery(false, ForceCollect::Yes)?
                        .unwrap())))
                },
                Annotatable::Expr(_) => {
                    |parser| Ok(Annotatable::Expr(parser.parse_expr_force_collect()?))
                }
                _ => unreachable!(),
            };

        // 'Flatten' all nonterminals (i.e. `TokenKind::Interpolated`)
        // to `None`-delimited groups containing the corresponding tokens. This
        // is normally delayed until the proc-macro server actually needs to
        // provide a `TokenKind::Interpolated` to a proc-macro. We do this earlier,
        // so that we can handle cases like:
        //
        // ```rust
        // #[cfg_eval] #[cfg] $item
        //```
        //
        // where `$item` is `#[cfg_attr] struct Foo {}`. We want to make
        // sure to evaluate *all* `#[cfg]` and `#[cfg_attr]` attributes - the simplest
        // way to do this is to do a single parse of a stream without any nonterminals.
        let orig_tokens = annotatable.to_tokens().flattened();

        // Re-parse the tokens, setting the `capture_cfg` flag to save extra information
        // to the captured `AttrTokenStream` (specifically, we capture
        // `AttrTokenTree::AttrsTarget` for all occurrences of `#[cfg]` and `#[cfg_attr]`)
        let mut parser = Parser::new(&self.0.sess.psess, orig_tokens, None);
        parser.capture_cfg = true;
        match parse_annotatable_with(&mut parser) {
            Ok(a) => annotatable = a,
            Err(err) => {
                err.emit();
                return Some(annotatable);
            }
        }

        // Now that we have our re-parsed `AttrTokenStream`, recursively configuring
        // our attribute target will correctly the tokens as well.
        flat_map_annotatable(self, annotatable)
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
        _ctxt: AssocCtxt,
    ) -> SmallVec<[P<ast::AssocItem>; 1]> {
        let item = configure!(self, item);
        mut_visit::walk_flat_map_item(self, item)
    }

    fn flat_map_foreign_item(
        &mut self,
        foreign_item: P<ast::ForeignItem>,
    ) -> SmallVec<[P<ast::ForeignItem>; 1]> {
        let foreign_item = configure!(self, foreign_item);
        mut_visit::walk_flat_map_item(self, foreign_item)
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
