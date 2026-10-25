//! The expansion from a function when the code would not compile with rustdoc: so if `cfg(doc)`,
//! expands to a `loop {}`.

use rustc_ast::token::{self, Delimiter, IdentKind};
use rustc_ast::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast::{
    AssocItemKind, AttrArgs, AttrItem, Attribute, DelimArgs, ExprKind, Fn, ItemKind, MetaItem,
    Safety,
};
use rustc_expand::base::*;
use rustc_span::{Ident, Span, sym};
use thin_vec::ThinVec;

pub(crate) fn expand(
    cx: &mut ExtCtxt<'_>,
    attr_path_sp: Span,
    meta_item: &MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    if !meta_item.is_word() {
        cx.dcx().struct_span_err(attr_path_sp, "`rustdoc_skip_body` doesn't take arguments").emit();
    }
    let mut is_expected_item = false;
    match item {
        Annotatable::Item(ref mut item) => {
            if let ItemKind::Fn(ref mut fn_) = item.kind {
                expand_macro(cx, attr_path_sp, fn_);
                is_expected_item = true;
            }
        }
        Annotatable::AssocItem(ref mut item, _) => {
            if let AssocItemKind::Fn(ref mut fn_) = item.kind {
                expand_macro(cx, attr_path_sp, fn_);
                is_expected_item = true;
            }
        }
        _ => {}
    }
    if !is_expected_item {
        cx.dcx()
            .struct_span_err(attr_path_sp, "`rustdoc_skip_body` can only be used on functions")
            .emit();
    }
    vec![item]
}

fn cfg_not_doc(cx: &mut ExtCtxt<'_>, sp: Span) -> Attribute {
    // `not(doc)`
    let not_doc = TokenStream::new(vec![
        TokenTree::token_alone(token::Ident(sym::not, IdentKind::Normal), sp),
        TokenTree::Delimited(
            DelimSpan::from_single(sp),
            DelimSpacing::new(Spacing::JointHidden, Spacing::Alone),
            Delimiter::Parenthesis,
            TokenStream::token_alone(token::Ident(sym::doc, IdentKind::Normal), sp),
        ),
    ]);

    cx.attr_nested(
        AttrItem {
            unsafety: Safety::Default,
            path: cx.path_ident(sp, Ident::new(sym::cfg, sp)),
            args: AttrArgs::Delimited(DelimArgs {
                dspan: DelimSpan::from_single(sp),
                delim: Delimiter::Parenthesis,
                tokens: not_doc,
            }),
            span: sp,
        },
        sp,
    )
}

fn expand_macro(cx: &mut ExtCtxt<'_>, sp: Span, fn_: &mut Fn) {
    let Some(old_body) = fn_.body.take() else {
        // No body... Well, no possibility to not compile I guess so all good.
        return;
    };
    let mut old_body = cx.expr_block(old_body);
    // We add on the function's body `#[cfg(not(doc))]`.
    old_body.attrs.push(cfg_not_doc(cx, sp));

    // We create the `#[cfg(doc)] loop {}` "block".
    let mut loop_expr = cx.expr(sp, ExprKind::Loop(cx.block(sp, ThinVec::new()), None, sp));
    loop_expr.attrs.push(cx.attr_nested_word(sym::cfg, sym::doc, sp));

    let mut stmts = ThinVec::new();
    stmts.push(cx.stmt_expr(loop_expr));
    stmts.push(cx.stmt_expr(old_body));

    // We create a new function body containing both blocks.
    fn_.body = Some(cx.block(sp, stmts));
}
