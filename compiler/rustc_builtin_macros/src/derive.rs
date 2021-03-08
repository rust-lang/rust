use crate::cfg_eval::cfg_eval;

use rustc_ast::{self as ast, token, ItemKind, MetaItemKind, NestedMetaItem, StmtKind};
use rustc_errors::{struct_span_err, Applicability};
use rustc_expand::base::{Annotatable, ExpandResult, ExtCtxt, Indeterminate, MultiItemModifier};
use rustc_feature::AttributeTemplate;
use rustc_parse::validate_attr;
use rustc_session::Session;
use rustc_span::symbol::sym;
use rustc_span::Span;

crate struct Expander;

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

        let template =
            AttributeTemplate { list: Some("Trait1, Trait2, ..."), ..Default::default() };
        let attr = ecx.attribute(meta_item.clone());
        validate_attr::check_builtin_attribute(&sess.parse_sess, &attr, sym::derive, template);

        let derives: Vec<_> = attr
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
            .collect();

        // FIXME: Try to cache intermediate results to avoid collecting same paths multiple times.
        match ecx.resolver.resolve_derives(ecx.current_expansion.id, derives, ecx.force_mode) {
            Ok(()) => ExpandResult::Ready(cfg_eval(ecx, item)),
            Err(Indeterminate) => ExpandResult::Retry(item),
        }
    }
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
        struct_span_err!(
            sess,
            span,
            E0774,
            "`derive` may only be applied to structs, enums and unions",
        )
        .emit();
    }
    bad_target
}

fn report_unexpected_literal(sess: &Session, lit: &ast::Lit) {
    let help_msg = match lit.token.kind {
        token::Str if rustc_lexer::is_ident(&lit.token.symbol.as_str()) => {
            format!("try using `#[derive({})]`", lit.token.symbol)
        }
        _ => "for example, write `#[derive(Debug)]` for `Debug`".to_string(),
    };
    struct_span_err!(sess, lit.span, E0777, "expected path to a trait, found literal",)
        .help(&help_msg)
        .emit();
}

fn report_path_args(sess: &Session, meta: &ast::MetaItem) {
    let report_error = |title, action| {
        let span = meta.span.with_lo(meta.path.span.hi());
        sess.struct_span_err(span, title)
            .span_suggestion(span, action, String::new(), Applicability::MachineApplicable)
            .emit();
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
