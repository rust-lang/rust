use crate::util::{check_builtin_macro_attribute, warn_on_duplicate_attribute};

use rustc_ast as ast;
use rustc_ast::mut_visit::MutVisitor;
use rustc_ast::ptr::P;
use rustc_ast::tokenstream::CanSynthesizeMissingTokens;
use rustc_ast::visit::Visitor;
use rustc_ast::{mut_visit, visit};
use rustc_ast::{AstLike, Attribute};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_expand::config::StripUnconfigured;
use rustc_expand::configure;
use rustc_feature::Features;
use rustc_parse::parser::{ForceCollect, Parser};
use rustc_session::utils::FlattenNonterminals;
use rustc_session::Session;
use rustc_span::symbol::sym;
use rustc_span::Span;
use smallvec::SmallVec;

crate fn expand(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    annotatable: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::cfg_eval);
    warn_on_duplicate_attribute(&ecx, &annotatable, sym::cfg_eval);
    vec![cfg_eval(ecx.sess, ecx.ecfg.features, annotatable)]
}

crate fn cfg_eval(
    sess: &Session,
    features: Option<&Features>,
    annotatable: Annotatable,
) -> Annotatable {
    CfgEval { cfg: &mut StripUnconfigured { sess, features, config_tokens: true } }
        .configure_annotatable(annotatable)
        // Since the item itself has already been configured by the `InvocationCollector`,
        // we know that fold result vector will contain exactly one element.
        .unwrap()
}

struct CfgEval<'a, 'b> {
    cfg: &'a mut StripUnconfigured<'b>,
}

fn flat_map_annotatable(
    vis: &mut impl MutVisitor,
    annotatable: Annotatable,
) -> Option<Annotatable> {
    match annotatable {
        Annotatable::Item(item) => vis.flat_map_item(item).pop().map(Annotatable::Item),
        Annotatable::TraitItem(item) => {
            vis.flat_map_trait_item(item).pop().map(Annotatable::TraitItem)
        }
        Annotatable::ImplItem(item) => {
            vis.flat_map_impl_item(item).pop().map(Annotatable::ImplItem)
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

struct CfgFinder {
    has_cfg_or_cfg_attr: bool,
}

impl CfgFinder {
    fn has_cfg_or_cfg_attr(annotatable: &Annotatable) -> bool {
        let mut finder = CfgFinder { has_cfg_or_cfg_attr: false };
        match annotatable {
            Annotatable::Item(item) => finder.visit_item(&item),
            Annotatable::TraitItem(item) => finder.visit_assoc_item(&item, visit::AssocCtxt::Trait),
            Annotatable::ImplItem(item) => finder.visit_assoc_item(&item, visit::AssocCtxt::Impl),
            Annotatable::ForeignItem(item) => finder.visit_foreign_item(&item),
            Annotatable::Stmt(stmt) => finder.visit_stmt(&stmt),
            Annotatable::Expr(expr) => finder.visit_expr(&expr),
            Annotatable::Arm(arm) => finder.visit_arm(&arm),
            Annotatable::ExprField(field) => finder.visit_expr_field(&field),
            Annotatable::PatField(field) => finder.visit_pat_field(&field),
            Annotatable::GenericParam(param) => finder.visit_generic_param(&param),
            Annotatable::Param(param) => finder.visit_param(&param),
            Annotatable::FieldDef(field) => finder.visit_field_def(&field),
            Annotatable::Variant(variant) => finder.visit_variant(&variant),
            Annotatable::Crate(krate) => finder.visit_crate(krate),
        };
        finder.has_cfg_or_cfg_attr
    }
}

impl<'ast> visit::Visitor<'ast> for CfgFinder {
    fn visit_attribute(&mut self, attr: &'ast Attribute) {
        // We want short-circuiting behavior, so don't use the '|=' operator.
        self.has_cfg_or_cfg_attr = self.has_cfg_or_cfg_attr
            || attr
                .ident()
                .map_or(false, |ident| ident.name == sym::cfg || ident.name == sym::cfg_attr);
    }
}

impl CfgEval<'_, '_> {
    fn configure<T: AstLike>(&mut self, node: T) -> Option<T> {
        self.cfg.configure(node)
    }

    fn configure_annotatable(&mut self, mut annotatable: Annotatable) -> Option<Annotatable> {
        // Tokenizing and re-parsing the `Annotatable` can have a significant
        // performance impact, so try to avoid it if possible
        if !CfgFinder::has_cfg_or_cfg_attr(&annotatable) {
            return Some(annotatable);
        }

        // The majority of parsed attribute targets will never need to have early cfg-expansion
        // run (e.g. they are not part of a `#[derive]` or `#[cfg_eval]` macro inoput).
        // Therefore, we normally do not capture the necessary information about `#[cfg]`
        // and `#[cfg_attr]` attributes during parsing.
        //
        // Therefore, when we actually *do* run early cfg-expansion, we need to tokenize
        // and re-parse the attribute target, this time capturing information about
        // the location of `#[cfg]` and `#[cfg_attr]` in the token stream. The tokenization
        // process is lossless, so this process is invisible to proc-macros.

        let parse_annotatable_with: fn(&mut Parser<'_>) -> _ = match annotatable {
            Annotatable::Item(_) => {
                |parser| Annotatable::Item(parser.parse_item(ForceCollect::Yes).unwrap().unwrap())
            }
            Annotatable::TraitItem(_) => |parser| {
                Annotatable::TraitItem(
                    parser.parse_trait_item(ForceCollect::Yes).unwrap().unwrap().unwrap(),
                )
            },
            Annotatable::ImplItem(_) => |parser| {
                Annotatable::ImplItem(
                    parser.parse_impl_item(ForceCollect::Yes).unwrap().unwrap().unwrap(),
                )
            },
            Annotatable::ForeignItem(_) => |parser| {
                Annotatable::ForeignItem(
                    parser.parse_foreign_item(ForceCollect::Yes).unwrap().unwrap().unwrap(),
                )
            },
            Annotatable::Stmt(_) => |parser| {
                Annotatable::Stmt(P(parser.parse_stmt(ForceCollect::Yes).unwrap().unwrap()))
            },
            Annotatable::Expr(_) => {
                |parser| Annotatable::Expr(parser.parse_expr_force_collect().unwrap())
            }
            _ => unreachable!(),
        };
        let nt = annotatable.into_nonterminal();

        let mut orig_tokens = rustc_parse::nt_to_tokenstream(
            &nt,
            &self.cfg.sess.parse_sess,
            CanSynthesizeMissingTokens::No,
        );

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
        let mut flatten = FlattenNonterminals {
            nt_to_tokenstream: rustc_parse::nt_to_tokenstream,
            parse_sess: &self.cfg.sess.parse_sess,
            synthesize_tokens: CanSynthesizeMissingTokens::No,
        };
        orig_tokens = flatten.process_token_stream(orig_tokens);

        // Re-parse the tokens, setting the `capture_cfg` flag to save extra information
        // to the captured `AttrAnnotatedTokenStream` (specifically, we capture
        // `AttrAnnotatedTokenTree::AttributesData` for all occurences of `#[cfg]` and `#[cfg_attr]`)
        let mut parser =
            rustc_parse::stream_to_parser(&self.cfg.sess.parse_sess, orig_tokens, None);
        parser.capture_cfg = true;
        annotatable = parse_annotatable_with(&mut parser);

        // Now that we have our re-parsed `AttrAnnotatedTokenStream`, recursively configuring
        // our attribute target will correctly the tokens as well.
        flat_map_annotatable(self, annotatable)
    }
}

impl MutVisitor for CfgEval<'_, '_> {
    fn visit_expr(&mut self, expr: &mut P<ast::Expr>) {
        self.cfg.configure_expr(expr);
        mut_visit::noop_visit_expr(expr, self);
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let mut expr = configure!(self, expr);
        mut_visit::noop_visit_expr(&mut expr, self);
        Some(expr)
    }

    fn flat_map_generic_param(
        &mut self,
        param: ast::GenericParam,
    ) -> SmallVec<[ast::GenericParam; 1]> {
        mut_visit::noop_flat_map_generic_param(configure!(self, param), self)
    }

    fn flat_map_stmt(&mut self, stmt: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        mut_visit::noop_flat_map_stmt(configure!(self, stmt), self)
    }

    fn flat_map_item(&mut self, item: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        mut_visit::noop_flat_map_item(configure!(self, item), self)
    }

    fn flat_map_impl_item(&mut self, item: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        mut_visit::noop_flat_map_assoc_item(configure!(self, item), self)
    }

    fn flat_map_trait_item(&mut self, item: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        mut_visit::noop_flat_map_assoc_item(configure!(self, item), self)
    }

    fn flat_map_foreign_item(
        &mut self,
        foreign_item: P<ast::ForeignItem>,
    ) -> SmallVec<[P<ast::ForeignItem>; 1]> {
        mut_visit::noop_flat_map_foreign_item(configure!(self, foreign_item), self)
    }

    fn flat_map_arm(&mut self, arm: ast::Arm) -> SmallVec<[ast::Arm; 1]> {
        mut_visit::noop_flat_map_arm(configure!(self, arm), self)
    }

    fn flat_map_expr_field(&mut self, field: ast::ExprField) -> SmallVec<[ast::ExprField; 1]> {
        mut_visit::noop_flat_map_expr_field(configure!(self, field), self)
    }

    fn flat_map_pat_field(&mut self, fp: ast::PatField) -> SmallVec<[ast::PatField; 1]> {
        mut_visit::noop_flat_map_pat_field(configure!(self, fp), self)
    }

    fn flat_map_param(&mut self, p: ast::Param) -> SmallVec<[ast::Param; 1]> {
        mut_visit::noop_flat_map_param(configure!(self, p), self)
    }

    fn flat_map_field_def(&mut self, sf: ast::FieldDef) -> SmallVec<[ast::FieldDef; 1]> {
        mut_visit::noop_flat_map_field_def(configure!(self, sf), self)
    }

    fn flat_map_variant(&mut self, variant: ast::Variant) -> SmallVec<[ast::Variant; 1]> {
        mut_visit::noop_flat_map_variant(configure!(self, variant), self)
    }
}
