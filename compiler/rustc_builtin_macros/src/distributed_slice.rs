use rustc_ast::ptr::P;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{
    ConstItem, DUMMY_NODE_ID, Defaultness, DistributedSlice, Expr, Generics, Item, ItemKind, Path,
    Ty, TyKind, ast,
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

/// ```rust
/// #[distributed_slice(crate)]
/// const MEOWS: [&str; _];
/// ```
pub(crate) fn distributed_slice(
    _ecx: &mut ExtCtxt<'_>,
    span: Span,
    _meta_item: &ast::MetaItem,
    mut orig_item: Annotatable,
) -> Vec<Annotatable> {
    // TODO: FIXME(gr)
    // FIXME(gr): check item

    let Annotatable::Item(item) = &mut orig_item else {
        panic!("expected `#[distributed_slice(crate)]` on an item")
    };

    match &mut item.kind {
        ItemKind::Static(static_item) => {
            static_item.distributed_slice = DistributedSlice::Declaration(span, DUMMY_NODE_ID);
        }
        ItemKind::Const(const_item) => {
            const_item.distributed_slice = DistributedSlice::Declaration(span, DUMMY_NODE_ID);
        }
        other => {
            panic!(
                "expected `#[distributed_slice(crate)]` on a const or static item, not {other:?}"
            );
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
        Err(err) => {
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
