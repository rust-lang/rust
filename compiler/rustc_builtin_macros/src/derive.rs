use crate::cfg_eval::cfg_eval;
use crate::errors::{NotApplicableDerive, PathRejected, TraitPathExpected};

use rustc_ast as ast;
use rustc_ast::{attr, token, GenericParamKind, ItemKind, MetaItemKind, NestedMetaItem, StmtKind};
use rustc_expand::base::{Annotatable, ExpandResult, ExtCtxt, Indeterminate, MultiItemModifier};
use rustc_feature::AttributeTemplate;
use rustc_parse::validate_attr;
use rustc_session::Session;
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;

pub(crate) struct Expander;

impl MultiItemModifier for Expander {
    fn expand(
        &self,
        ecx: &mut ExtCtxt<'_>,
        span: Span,
        meta_item: &ast::MetaItem,
        item: Annotatable,
    ) -> ExpandResult<Vec<Annotatable>, Annotatable> {
        let sess = ecx.sess;
        if report_bad_target(sess, &item, span) {
            // We don't want to pass inappropriate targets to derive macros to avoid
            // follow up errors, all other errors below are recoverable.
            return ExpandResult::Ready(vec![item]);
        }

        let (sess, features) = (ecx.sess, ecx.ecfg.features);
        let result =
            ecx.resolver.resolve_derives(ecx.current_expansion.id, ecx.force_mode, &|| {
                let template =
                    AttributeTemplate { list: Some("Trait1, Trait2, ..."), ..Default::default() };
                let attr = attr::mk_attr_outer(meta_item.clone());
                validate_attr::check_builtin_attribute(
                    &sess.parse_sess,
                    &attr,
                    sym::derive,
                    template,
                );

                let mut resolutions: Vec<_> = attr
                    .meta_item_list()
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|nested_meta| match nested_meta {
                        NestedMetaItem::MetaItem(meta) => Some(meta),
                        NestedMetaItem::Literal(lit) => {
                            // Reject `#[derive("Debug")]`.
                            report_unexpected_literal(sess, &lit);
                            None
                        }
                    })
                    .map(|meta| {
                        // Reject `#[derive(Debug = "value", Debug(abc))]`, but recover the paths.
                        report_path_args(sess, &meta);
                        meta.path
                    })
                    .map(|path| (path, dummy_annotatable(), None))
                    .collect();

                // Do not configure or clone items unless necessary.
                match &mut resolutions[..] {
                    [] => {}
                    [(_, first_item, _), others @ ..] => {
                        *first_item = cfg_eval(
                            sess,
                            features,
                            item.clone(),
                            ecx.current_expansion.lint_node_id,
                        );
                        for (_, item, _) in others {
                            *item = first_item.clone();
                        }
                    }
                }

                resolutions
            });

        match result {
            Ok(()) => ExpandResult::Ready(vec![item]),
            Err(Indeterminate) => ExpandResult::Retry(item),
        }
    }
}

// The cheapest `Annotatable` to construct.
fn dummy_annotatable() -> Annotatable {
    Annotatable::GenericParam(ast::GenericParam {
        id: ast::DUMMY_NODE_ID,
        ident: Ident::empty(),
        attrs: Default::default(),
        bounds: Default::default(),
        is_placeholder: false,
        kind: GenericParamKind::Lifetime,
        colon_span: None,
    })
}

fn report_bad_target(sess: &Session, item: &Annotatable, span: Span) -> bool {
    let item_kind = match item {
        Annotatable::Item(item) => Some(&item.kind),
        Annotatable::Stmt(stmt) => match &stmt.kind {
            StmtKind::Item(item) => Some(&item.kind),
            _ => None,
        },
        _ => None,
    };

    let bad_target =
        !matches!(item_kind, Some(ItemKind::Struct(..) | ItemKind::Enum(..) | ItemKind::Union(..)));
    if bad_target {
        sess.emit_err(NotApplicableDerive { span, item_span: item.span() });
    }
    bad_target
}

fn report_unexpected_literal(sess: &Session, lit: &ast::Lit) {
    let help_msg = match lit.token_lit.kind {
        token::Str if rustc_lexer::is_ident(lit.token_lit.symbol.as_str()) => {
            format!("try using `#[derive({})]`", lit.token_lit.symbol)
        }
        _ => "for example, write `#[derive(Debug)]` for `Debug`".to_string(),
    };
    sess.emit_err(TraitPathExpected { span: lit.span, help_msg: &help_msg });
}

fn report_path_args(sess: &Session, meta: &ast::MetaItem) {
    let report_error = |title, action| {
        let span = meta.span.with_lo(meta.path.span.hi());
        sess.emit_err(PathRejected { span, title, action });
    };
    match meta.kind {
        MetaItemKind::Word => {}
        MetaItemKind::List(..) => report_error(
            "traits in `#[derive(...)]` don't accept arguments",
            "remove the arguments",
        ),
        MetaItemKind::NameValue(..) => {
            report_error("traits in `#[derive(...)]` don't accept values", "remove the value")
        }
    }
}
