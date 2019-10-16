//! The compiler code necessary to implement the `#[derive]` extensions.

use syntax::ast::{self, MetaItem};
use syntax_expand::base::{Annotatable, ExtCtxt, MultiItemModifier};
use syntax::ptr::P;
use syntax::symbol::{Symbol, sym};
use syntax_pos::Span;

macro path_local($x:ident) {
    generic::ty::Path::new_local(stringify!($x))
}

macro pathvec_std($cx:expr, $($rest:ident)::+) {{
    vec![ $( stringify!($rest) ),+ ]
}}

macro path_std($($x:tt)*) {
    generic::ty::Path::new( pathvec_std!( $($x)* ) )
}

pub mod bounds;
pub mod clone;
pub mod encodable;
pub mod decodable;
pub mod hash;
pub mod debug;
pub mod default;

#[path="cmp/partial_eq.rs"]
pub mod partial_eq;
#[path="cmp/eq.rs"]
pub mod eq;
#[path="cmp/partial_ord.rs"]
pub mod partial_ord;
#[path="cmp/ord.rs"]
pub mod ord;

pub mod generic;

crate struct BuiltinDerive(
    crate fn(&mut ExtCtxt<'_>, Span, &MetaItem, &Annotatable, &mut dyn FnMut(Annotatable))
);

impl MultiItemModifier for BuiltinDerive {
    fn expand(&self,
              ecx: &mut ExtCtxt<'_>,
              span: Span,
              meta_item: &MetaItem,
              item: Annotatable)
              -> Vec<Annotatable> {
        // FIXME: Built-in derives often forget to give spans contexts,
        // so we are doing it here in a centralized way.
        let span = ecx.with_def_site_ctxt(span);
        let mut items = Vec::new();
        (self.0)(ecx, span, meta_item, &item, &mut |a| items.push(a));
        items
    }
}

/// Constructs an expression that calls an intrinsic
fn call_intrinsic(cx: &ExtCtxt<'_>,
                  span: Span,
                  intrinsic: &str,
                  args: Vec<P<ast::Expr>>)
                  -> P<ast::Expr> {
    let span = cx.with_def_site_ctxt(span);
    let path = cx.std_path(&[sym::intrinsics, Symbol::intern(intrinsic)]);
    let call = cx.expr_call_global(span, path, args);

    cx.expr_block(P(ast::Block {
        stmts: vec![cx.stmt_expr(call)],
        id: ast::DUMMY_NODE_ID,
        rules: ast::BlockCheckMode::Unsafe(ast::CompilerGenerated),
        span,
    }))
}
