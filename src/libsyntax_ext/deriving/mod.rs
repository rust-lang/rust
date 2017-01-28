// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The compiler code necessary to implement the `#[derive]` extensions.

use std::rc::Rc;
use syntax::ast;
use syntax::codemap;
use syntax::ext::base::{Annotatable, ExtCtxt, SyntaxExtension, Resolver};
use syntax::ext::build::AstBuilder;
use syntax::ptr::P;
use syntax::symbol::Symbol;
use syntax_pos::Span;

macro_rules! pathvec {
    ($($x:ident)::+) => (
        vec![ $( stringify!($x) ),+ ]
    )
}

macro_rules! path {
    ($($x:tt)*) => (
        ::ext::deriving::generic::ty::Path::new( pathvec![ $($x)* ] )
    )
}

macro_rules! path_local {
    ($x:ident) => (
        ::deriving::generic::ty::Path::new_local(stringify!($x))
    )
}

macro_rules! pathvec_std {
    ($cx:expr, $first:ident :: $($rest:ident)::+) => ({
        let mut v = pathvec![$($rest)::+];
        if let Some(s) = $cx.crate_root {
            v.insert(0, s);
        }
        v
    })
}

macro_rules! path_std {
    ($($x:tt)*) => (
        ::deriving::generic::ty::Path::new( pathvec_std!( $($x)* ) )
    )
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

fn allow_unstable(cx: &mut ExtCtxt, span: Span, attr_name: &str) -> Span {
    Span {
        expn_id: cx.codemap().record_expansion(codemap::ExpnInfo {
            call_site: span,
            callee: codemap::NameAndSpan {
                format: codemap::MacroAttribute(Symbol::intern(attr_name)),
                span: Some(span),
                allow_internal_unstable: true,
            },
        }),
        ..span
    }
}

macro_rules! derive_traits {
    ($( $name:expr => $func:path, )+) => {
        pub fn is_builtin_trait(name: ast::Name) -> bool {
            match &*name.as_str() {
                $( $name )|+ => true,
                _ => false,
            }
        }

        pub fn register_builtin_derives(resolver: &mut Resolver) {
            $(
                resolver.add_ext(
                    ast::Ident::with_empty_ctxt(Symbol::intern($name)),
                    Rc::new(SyntaxExtension::BuiltinDerive($func))
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
fn warn_if_deprecated(ecx: &mut ExtCtxt, sp: Span, name: &str) {
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
            ast::ItemKind::Struct(_, ast::Generics { ref ty_params, .. }) |
            ast::ItemKind::Enum(_, ast::Generics { ref ty_params, .. }) => {
                for ty in ty_params.iter() {
                    typaram.push_str(&ty.ident.name.as_str());
                }
            }

            _ => {}
        }
    }

    typaram
}

/// Constructs an expression that calls an intrinsic
fn call_intrinsic(cx: &ExtCtxt,
                  mut span: Span,
                  intrinsic: &str,
                  args: Vec<P<ast::Expr>>)
                  -> P<ast::Expr> {
    span.expn_id = cx.codemap().record_expansion(codemap::ExpnInfo {
        call_site: span,
        callee: codemap::NameAndSpan {
            format: codemap::MacroAttribute(Symbol::intern("derive")),
            span: Some(span),
            allow_internal_unstable: true,
        },
    });

    let path = cx.std_path(&["intrinsics", intrinsic]);
    let call = cx.expr_call_global(span, path, args);

    cx.expr_block(P(ast::Block {
        stmts: vec![cx.stmt_expr(call)],
        id: ast::DUMMY_NODE_ID,
        rules: ast::BlockCheckMode::Unsafe(ast::CompilerGenerated),
        span: span,
    }))
}
