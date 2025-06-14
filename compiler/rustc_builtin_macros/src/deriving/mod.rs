//! The compiler code necessary to implement the `#[derive]` extensions.

use rustc_ast as ast;
use rustc_ast::ptr::P;
use rustc_ast::{GenericArg, MetaItem};
use rustc_expand::base::{Annotatable, ExpandResult, ExtCtxt, MultiItemModifier};
use rustc_span::{Span, Symbol, sym};
use thin_vec::{ThinVec, thin_vec};

macro path_local($x:ident) {
    generic::ty::Path::new_local(sym::$x)
}

macro pathvec_std($($rest:ident)::+) {{
    vec![ $( sym::$rest ),+ ]
}}

macro path_std($($x:tt)*) {
    generic::ty::Path::new( pathvec_std!( $($x)* ) )
}

pub(crate) mod bounds;
pub(crate) mod clone;
pub(crate) mod coerce_pointee;
pub(crate) mod debug;
pub(crate) mod default;
pub(crate) mod hash;

#[path = "cmp/eq.rs"]
pub(crate) mod eq;
#[path = "cmp/ord.rs"]
pub(crate) mod ord;
#[path = "cmp/partial_eq.rs"]
pub(crate) mod partial_eq;
#[path = "cmp/partial_ord.rs"]
pub(crate) mod partial_ord;

pub(crate) mod generic;

pub(crate) type BuiltinDeriveFn =
    fn(&ExtCtxt<'_>, Span, &MetaItem, &Annotatable, &mut dyn FnMut(Annotatable), bool);

pub(crate) struct BuiltinDerive(pub(crate) BuiltinDeriveFn);

impl MultiItemModifier for BuiltinDerive {
    fn expand(
        &self,
        ecx: &mut ExtCtxt<'_>,
        span: Span,
        meta_item: &MetaItem,
        item: Annotatable,
        is_derive_const: bool,
    ) -> ExpandResult<Vec<Annotatable>, Annotatable> {
        // FIXME: Built-in derives often forget to give spans contexts,
        // so we are doing it here in a centralized way.
        let span = ecx.with_def_site_ctxt(span);
        let mut items = Vec::new();
        match item {
            Annotatable::Stmt(stmt) => {
                if let ast::StmtKind::Item(item) = stmt.kind {
                    (self.0)(
                        ecx,
                        span,
                        meta_item,
                        &Annotatable::Item(item),
                        &mut |a| {
                            // Cannot use 'ecx.stmt_item' here, because we need to pass 'ecx'
                            // to the function
                            items.push(Annotatable::Stmt(P(ast::Stmt {
                                id: ast::DUMMY_NODE_ID,
                                kind: ast::StmtKind::Item(a.expect_item()),
                                span,
                            })));
                        },
                        is_derive_const,
                    );
                } else {
                    unreachable!("should have already errored on non-item statement")
                }
            }
            _ => {
                (self.0)(ecx, span, meta_item, &item, &mut |a| items.push(a), is_derive_const);
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
    args: ThinVec<P<ast::Expr>>,
) -> P<ast::Expr> {
    let span = cx.with_def_site_ctxt(span);
    let path = cx.std_path(&[sym::intrinsics, intrinsic]);
    cx.expr_call_global(span, path, args)
}

/// Constructs an expression that calls the `unreachable` intrinsic.
fn call_unreachable(cx: &ExtCtxt<'_>, span: Span) -> P<ast::Expr> {
    let span = cx.with_def_site_ctxt(span);
    let path = cx.std_path(&[sym::intrinsics, sym::unreachable]);
    let call = cx.expr_call_global(span, path, ThinVec::new());

    cx.expr_block(P(ast::Block {
        stmts: thin_vec![cx.stmt_expr(call)],
        id: ast::DUMMY_NODE_ID,
        rules: ast::BlockCheckMode::Unsafe(ast::CompilerGenerated),
        span,
        tokens: None,
    }))
}

fn assert_ty_bounds(
    cx: &ExtCtxt<'_>,
    stmts: &mut ThinVec<ast::Stmt>,
    ty: P<ast::Ty>,
    span: Span,
    assert_path: &[Symbol],
) {
    // Generate statement `let _: assert_path<ty>;`.
    let span = cx.with_def_site_ctxt(span);
    let assert_path = cx.path_all(span, true, cx.std_path(assert_path), vec![GenericArg::Type(ty)]);
    stmts.push(cx.stmt_let_type_only(span, cx.ty_path(assert_path)));
}
