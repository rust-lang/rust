//! The compiler code necessary to implement the `#[derive]` extensions.

use rustc_data_structures::sync::Lrc;
use syntax::ast;
use syntax::ext::base::{Annotatable, ExtCtxt, SyntaxExtension, Resolver};
use syntax::ext::build::AstBuilder;
use syntax::ext::hygiene::{Mark, SyntaxContext};
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
pub mod custom;

#[path="cmp/partial_eq.rs"]
pub mod partial_eq;
#[path="cmp/eq.rs"]
pub mod eq;
#[path="cmp/partial_ord.rs"]
pub mod partial_ord;
#[path="cmp/ord.rs"]
pub mod ord;


pub mod generic;

macro_rules! derive_traits {
    ($( $name:expr => $func:path, )+) => {
        pub fn is_builtin_trait(name: ast::Name) -> bool {
            match &*name.as_str() {
                $( $name )|+ => true,
                _ => false,
            }
        }

        pub fn register_builtin_derives(resolver: &mut dyn Resolver) {
            $(
                resolver.add_builtin(
                    ast::Ident::with_empty_ctxt(Symbol::intern($name)),
                    Lrc::new(SyntaxExtension::BuiltinDerive($func))
                );
            )*
        }
    }
}

derive_traits! {
    "Clone" => clone::expand_deriving_clone,

    "Hash" => hash::expand_deriving_hash,

    "RustcEncodable" => encodable::expand_deriving_rustc_encodable,

    "RustcDecodable" => decodable::expand_deriving_rustc_decodable,

    "PartialEq" => partial_eq::expand_deriving_partial_eq,
    "Eq" => eq::expand_deriving_eq,
    "PartialOrd" => partial_ord::expand_deriving_partial_ord,
    "Ord" => ord::expand_deriving_ord,

    "Debug" => debug::expand_deriving_debug,

    "Default" => default::expand_deriving_default,

    "Send" => bounds::expand_deriving_unsafe_bound,
    "Sync" => bounds::expand_deriving_unsafe_bound,
    "Copy" => bounds::expand_deriving_copy,

    // deprecated
    "Encodable" => encodable::expand_deriving_encodable,
    "Decodable" => decodable::expand_deriving_decodable,
}

#[inline] // because `name` is a compile-time constant
fn warn_if_deprecated(ecx: &mut ExtCtxt<'_>, sp: Span, name: &str) {
    if let Some(replacement) = match name {
        "Encodable" => Some("RustcEncodable"),
        "Decodable" => Some("RustcDecodable"),
        _ => None,
    } {
        ecx.span_warn(sp,
                      &format!("derive({}) is deprecated in favor of derive({})",
                               name,
                               replacement));
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
                  mut span: Span,
                  intrinsic: &str,
                  args: Vec<P<ast::Expr>>)
                  -> P<ast::Expr> {
    let intrinsic_allowed_via_allow_internal_unstable = cx
        .current_expansion.mark.expn_info().unwrap()
        .allow_internal_unstable.map_or(false, |features| features.iter().any(|&s|
            s == sym::core_intrinsics
        ));
    if intrinsic_allowed_via_allow_internal_unstable {
        span = span.with_ctxt(cx.backtrace());
    } else { // Avoid instability errors with user defined curstom derives, cc #36316
        let mut info = cx.current_expansion.mark.expn_info().unwrap();
        info.allow_internal_unstable = Some(vec![Symbol::intern("core_intrinsics")].into());
        let mark = Mark::fresh(Mark::root());
        mark.set_expn_info(info);
        span = span.with_ctxt(SyntaxContext::empty().apply_mark(mark));
    }
    let path = cx.std_path(&["intrinsics", intrinsic]);
    let call = cx.expr_call_global(span, path, args);

    cx.expr_block(P(ast::Block {
        stmts: vec![cx.stmt_expr(call)],
        id: ast::DUMMY_NODE_ID,
        rules: ast::BlockCheckMode::Unsafe(ast::CompilerGenerated),
        span,
    }))
}
