//! The compiler code necessary to implement the `#[derive]` extensions.

use syntax::ast::{self, MetaItem};
use syntax::ext::base::{Annotatable, ExtCtxt, MultiItemModifier};
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
        let mut items = Vec::new();
        (self.0)(ecx, span, meta_item, &item, &mut |a| items.push(a));
        items
    }
}

/// Construct a name for the inner type parameter that can't collide with any type parameters of
/// the item. This is achieved by starting with a base and then concatenating the names of all
/// other type parameters.
// FIXME(aburka): use real hygiene when that becomes possible
fn hygienic_type_parameter(item: &Annotatable, base: &str) -> String {
    let mut typaram = String::from(base);
    if let Annotatable::Item(ref item) = *item {
        match item.node {
            ast::ItemKind::Struct(_, ast::Generics { ref params, .. }) |
            ast::ItemKind::Enum(_, ast::Generics { ref params, .. }) => {
                for param in params {
                    match param.kind {
                        ast::GenericParamKind::Type { .. } => {
                            typaram.push_str(&param.ident.as_str());
                        }
                        _ => {}
                    }
                }
            }

            _ => {}
        }
    }

    typaram
}

/// Constructs an expression that calls an intrinsic
fn call_intrinsic(cx: &ExtCtxt<'_>,
                  span: Span,
                  intrinsic: &str,
                  args: Vec<P<ast::Expr>>)
                  -> P<ast::Expr> {
    let span = span.with_ctxt(cx.backtrace());
    let path = cx.std_path(&[sym::intrinsics, Symbol::intern(intrinsic)]);
    let call = cx.expr_call_global(span, path, args);

    cx.expr_block(P(ast::Block {
        stmts: vec![cx.stmt_expr(call)],
        id: ast::DUMMY_NODE_ID,
        rules: ast::BlockCheckMode::Unsafe(ast::CompilerGenerated),
        span,
    }))
}
