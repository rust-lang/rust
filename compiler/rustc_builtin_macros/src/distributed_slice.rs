use rustc_ast::ptr::P;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{
    AssocItemKind, ConstItem, DUMMY_NODE_ID, Defaultness, DistributedSlice, Expr, ForeignItemKind,
    Generics, Item, ItemKind, Path, Ty, TyKind, ast,
};
use rustc_errors::PResult;
use rustc_expand::base::{
    Annotatable, DummyResult, ExpandResult, ExtCtxt, MacEager, MacroExpanderResult,
};
use rustc_parse::exp;
use rustc_parse::parser::{Parser, PathStyle};
use rustc_span::{Ident, Span, kw};
use smallvec::smallvec;
use thin_vec::ThinVec;

use crate::errors::{
    DistributedSliceAssocItem, DistributedSliceExpectedConstStatic, DistributedSliceExpectedCrate,
    DistributedSliceForeignItem, DistributedSliceGeneric,
};

/// ```rust
/// #[distributed_slice(crate)]
/// const MEOWS: [&str; _];
/// ```
pub(crate) fn distributed_slice(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    meta_item: &ast::MetaItem,
    mut orig_item: Annotatable,
) -> Vec<Annotatable> {
    // TODO: FIXME(gr)

    if let Some([ast::MetaItemInner::MetaItem(mi)]) = meta_item.meta_item_list() {
        if !mi.is_word() || !mi.path.is_ident(kw::Crate) {
            ecx.dcx().emit_err(DistributedSliceExpectedCrate { span: meta_item.span });
        }
    } else {
        ecx.dcx().emit_err(DistributedSliceExpectedCrate { span: meta_item.span });
    };

    let item_span = orig_item.span();

    let Annotatable::Item(item) = &mut orig_item else {
        if let Annotatable::ForeignItem(fi) = &mut orig_item {
            let eg = ecx.dcx().emit_err(DistributedSliceForeignItem {
                span: item_span,
                attr_span: meta_item.span,
            });

            if let ForeignItemKind::Static(static_item) = &mut fi.kind {
                static_item.distributed_slice = DistributedSlice::Err(eg);
            }
        } else if let Annotatable::AssocItem(ai, ..) = &mut orig_item {
            let eg = ecx
                .dcx()
                .emit_err(DistributedSliceAssocItem { span: item_span, attr_span: meta_item.span });

            if let AssocItemKind::Const(const_item) = &mut ai.kind {
                const_item.distributed_slice = DistributedSlice::Err(eg);
            }
        } else {
            ecx.dcx().emit_err(DistributedSliceExpectedConstStatic {
                span: orig_item.span(),
                attr_span: meta_item.span,
            });
        }

        return vec![orig_item];
    };

    match &mut item.kind {
        ItemKind::Static(static_item) => {
            static_item.distributed_slice = DistributedSlice::Declaration(span, DUMMY_NODE_ID);
        }
        ItemKind::Const(const_item) => {
            if !const_item.generics.params.is_empty()
                || !const_item.generics.where_clause.is_empty()
            {
                ecx.dcx().emit_err(DistributedSliceGeneric { span: item_span });
            }

            const_item.distributed_slice = DistributedSlice::Declaration(span, DUMMY_NODE_ID);
        }
        _ => {
            ecx.dcx().emit_err(DistributedSliceExpectedConstStatic {
                span: item.span,
                attr_span: meta_item.span,
            });
            return vec![orig_item];
        }
    }

    vec![orig_item]
}

fn parse_element(mut p: Parser<'_>) -> PResult<'_, (Path, P<Expr>)> {
    let path = p.parse_path(PathStyle::Expr)?;
    p.expect(exp![Comma])?;
    let expr = p.parse_expr()?;

    // optional trailing comma
    let _ = p.eat(exp![Comma]);

    Ok((path, expr))
}

/// ```rust
/// distributed_slice_element!(MEOWS, "mrow");
/// ```
pub(crate) fn distributed_slice_element(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let (path, expr) = match parse_element(cx.new_parser_from_tts(tts)) {
        Ok((ident, expr)) => (ident, expr),
        Err(mut err) => {
            if err.span.is_dummy() {
                err.span(span);
            }
            let guar = err.emit();
            return ExpandResult::Ready(DummyResult::any(span, guar));
        }
    };

    ExpandResult::Ready(MacEager::items(smallvec![P(Item {
        attrs: ThinVec::new(),
        id: DUMMY_NODE_ID,
        span,
        vis: ast::Visibility { kind: ast::VisibilityKind::Inherited, span, tokens: None },
        kind: ItemKind::Const(Box::new(ConstItem {
            defaultness: Defaultness::Final,
            ident: Ident { name: kw::Underscore, span },
            generics: Generics::default(),
            // leave out the ty, we discover it when
            // when name-resolving to the registry definition
            ty: P(Ty { id: DUMMY_NODE_ID, kind: TyKind::Infer, span, tokens: None }),
            expr: Some(expr),
            define_opaque: None,
            distributed_slice: DistributedSlice::Addition { declaration: path, id: DUMMY_NODE_ID }
        })),
        tokens: None
    })]))
}

/// ```rust
/// distributed_slice_elements!(MEOWS, ["mrow"]);
/// ```
pub(crate) fn distributed_slice_elements(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let (path, expr) = match parse_element(cx.new_parser_from_tts(tts)) {
        Ok((ident, expr)) => (ident, expr),
        Err(mut err) => {
            if err.span.is_dummy() {
                err.span(span);
            }
            let guar = err.emit();
            return ExpandResult::Ready(DummyResult::any(span, guar));
        }
    };

    ExpandResult::Ready(MacEager::items(smallvec![P(Item {
        attrs: ThinVec::new(),
        id: DUMMY_NODE_ID,
        span,
        vis: ast::Visibility { kind: ast::VisibilityKind::Inherited, span, tokens: None },
        kind: ItemKind::Const(Box::new(ConstItem {
            defaultness: Defaultness::Final,
            ident: Ident { name: kw::Underscore, span },
            generics: Generics::default(),
            // leave out the ty, we discover it when
            // when name-resolving to the registry definition
            ty: P(Ty { id: DUMMY_NODE_ID, kind: TyKind::Infer, span, tokens: None }),
            expr: Some(expr),
            define_opaque: None,
            distributed_slice: DistributedSlice::AdditionMany {
                declaration: path,
                id: DUMMY_NODE_ID
            }
        })),
        tokens: None
    })]))
}
