//! The compiler code necessary to implement the `#[derive]` extensions.

use rustc_ast as ast;
use rustc_ast::ptr::P;
use rustc_ast::{ItemKind, MetaItem};
use rustc_expand::base::{Annotatable, ExpandResult, ExtCtxt, MultiItemModifier};
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::Span;

macro path_local($x:ident) {
    generic::ty::Path::new_local(sym::$x)
}

macro pathvec_std($($rest:ident)::+) {{
    vec![ $( sym::$rest ),+ ]
}}

macro path_std($($x:tt)*) {
    generic::ty::Path::new( pathvec_std!( $($x)* ) )
}

pub mod bounds;
pub mod clone;
pub mod debug;
pub mod decodable;
pub mod default;
pub mod encodable;
pub mod hash;

#[path = "cmp/eq.rs"]
pub mod eq;
#[path = "cmp/ord.rs"]
pub mod ord;
#[path = "cmp/partial_eq.rs"]
pub mod partial_eq;
#[path = "cmp/partial_ord.rs"]
pub mod partial_ord;

pub mod generic;

crate struct BuiltinDerive(
    crate fn(&mut ExtCtxt<'_>, Span, &MetaItem, &Annotatable, &mut dyn FnMut(Annotatable)),
);

impl MultiItemModifier for BuiltinDerive {
    fn expand(
        &self,
        ecx: &mut ExtCtxt<'_>,
        span: Span,
        meta_item: &MetaItem,
        item: Annotatable,
    ) -> ExpandResult<Vec<Annotatable>, Annotatable> {
        // FIXME: Built-in derives often forget to give spans contexts,
        // so we are doing it here in a centralized way.
        let span = ecx.with_def_site_ctxt(span);
        let mut items = Vec::new();
        match item {
            Annotatable::Stmt(stmt) => {
                if let ast::StmtKind::Item(item) = stmt.into_inner().kind {
                    (self.0)(ecx, span, meta_item, &Annotatable::Item(item), &mut |a| {
                        // Cannot use 'ecx.stmt_item' here, because we need to pass 'ecx'
                        // to the function
                        items.push(Annotatable::Stmt(P(ast::Stmt {
                            id: ast::DUMMY_NODE_ID,
                            kind: ast::StmtKind::Item(a.expect_item()),
                            span,
                            tokens: None,
                        })));
                    });
                } else {
                    unreachable!("should have already errored on non-item statement")
                }
            }
            _ => {
                (self.0)(ecx, span, meta_item, &item, &mut |a| items.push(a));
            }
        }
        ExpandResult::Ready(items)
    }
}

/// Constructs an expression that calls an intrinsic
fn call_intrinsic(
    cx: &ExtCtxt<'_>,
    span: Span,
    intrinsic: Symbol,
    args: Vec<P<ast::Expr>>,
) -> P<ast::Expr> {
    let span = cx.with_def_site_ctxt(span);
    let path = cx.std_path(&[sym::intrinsics, intrinsic]);
    cx.expr_call_global(span, path, args)
}

/// Constructs an expression that calls the `unreachable` intrinsic.
fn call_unreachable(cx: &ExtCtxt<'_>, span: Span) -> P<ast::Expr> {
    let span = cx.with_def_site_ctxt(span);
    let path = cx.std_path(&[sym::intrinsics, sym::unreachable]);
    let call = cx.expr_call_global(span, path, vec![]);

    cx.expr_block(P(ast::Block {
        stmts: vec![cx.stmt_expr(call)],
        id: ast::DUMMY_NODE_ID,
        rules: ast::BlockCheckMode::Unsafe(ast::CompilerGenerated),
        span,
        tokens: None,
    }))
}

// Injects `impl<...> Structural for ItemType<...> { }`. In particular,
// does *not* add `where T: Structural` for parameters `T` in `...`.
// (That's the main reason we cannot use TraitDef here.)
fn inject_impl_of_structural_trait(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    item: &Annotatable,
    structural_path: generic::ty::Path,
    push: &mut dyn FnMut(Annotatable),
) {
    let item = match *item {
        Annotatable::Item(ref item) => item,
        _ => unreachable!(),
    };

    let generics = match item.kind {
        ItemKind::Struct(_, ref generics) | ItemKind::Enum(_, ref generics) => generics,
        // Do not inject `impl Structural for Union`. (`PartialEq` does not
        // support unions, so we will see error downstream.)
        ItemKind::Union(..) => return,
        _ => unreachable!(),
    };

    // Create generics param list for where clauses and impl headers
    let mut generics = generics.clone();

    // Create the type of `self`.
    //
    // in addition, remove defaults from type params (impls cannot have them).
    let self_params: Vec<_> = generics
        .params
        .iter_mut()
        .map(|param| match &mut param.kind {
            ast::GenericParamKind::Lifetime => {
                ast::GenericArg::Lifetime(cx.lifetime(span, param.ident))
            }
            ast::GenericParamKind::Type { default } => {
                *default = None;
                ast::GenericArg::Type(cx.ty_ident(span, param.ident))
            }
            ast::GenericParamKind::Const { ty: _, kw_span: _ } => {
                ast::GenericArg::Const(cx.const_ident(span, param.ident))
            }
        })
        .collect();

    let type_ident = item.ident;

    let trait_ref = cx.trait_ref(structural_path.to_path(cx, span, type_ident, &generics));
    let self_type = cx.ty_path(cx.path_all(span, false, vec![type_ident], self_params));

    // It would be nice to also encode constraint `where Self: Eq` (by adding it
    // onto `generics` cloned above). Unfortunately, that strategy runs afoul of
    // rust-lang/rust#48214. So we perform that additional check in the compiler
    // itself, instead of encoding it here.

    // Keep the lint and stability attributes of the original item, to control
    // how the generated implementation is linted.
    let mut attrs = Vec::new();
    attrs.extend(
        item.attrs
            .iter()
            .filter(|a| {
                [sym::allow, sym::warn, sym::deny, sym::forbid, sym::stable, sym::unstable]
                    .contains(&a.name_or_empty())
            })
            .cloned(),
    );

    let newitem = cx.item(
        span,
        Ident::invalid(),
        attrs,
        ItemKind::Impl {
            unsafety: ast::Unsafe::No,
            polarity: ast::ImplPolarity::Positive,
            defaultness: ast::Defaultness::Final,
            constness: ast::Const::No,
            generics,
            of_trait: Some(trait_ref),
            self_ty: self_type,
            items: Vec::new(),
        },
    );

    push(Annotatable::Item(newitem));
}
