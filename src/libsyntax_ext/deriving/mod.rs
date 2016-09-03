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

use syntax::ast::{self, MetaItem};
use syntax::ext::base::{Annotatable, ExtCtxt, SyntaxEnv};
use syntax::ext::base::MultiModifier;
use syntax::ext::build::AstBuilder;
use syntax::feature_gate;
use syntax::codemap;
use syntax::parse::token::{intern, intern_and_get_ident};
use syntax::ptr::P;
use syntax_pos::Span;

macro_rules! pathvec {
    ($($x:ident)::+) => (
        vec![ $( stringify!($x) ),+ ]
    )
}

macro_rules! path {
    ($($x:tt)*) => (
        ::ext::deriving::generic::ty::Path::new( pathvec!( $($x)* ) )
    )
}

macro_rules! path_local {
    ($x:ident) => (
        ::deriving::generic::ty::Path::new_local(stringify!($x))
    )
}

macro_rules! pathvec_std {
    ($cx:expr, $first:ident :: $($rest:ident)::+) => ({
        let mut v = pathvec!($($rest)::+);
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
                format: codemap::MacroAttribute(intern(attr_name)),
                span: Some(span),
                allow_internal_unstable: true,
            },
        }),
        ..span
    }
}

fn expand_derive(cx: &mut ExtCtxt,
                 span: Span,
                 mitem: &MetaItem,
                 annotatable: Annotatable)
                 -> Vec<Annotatable> {
    debug!("expand_derive: span = {:?}", span);
    debug!("expand_derive: mitem = {:?}", mitem);
    debug!("expand_derive: annotatable input  = {:?}", annotatable);
    let mut item = match annotatable {
        Annotatable::Item(item) => item,
        other => {
            cx.span_err(span, "`derive` can only be applied to items");
            return vec![other]
        }
    };

    if mitem.value_str().is_some() {
        cx.span_err(mitem.span, "unexpected value in `derive`");
    }

    let traits = mitem.meta_item_list().unwrap_or(&[]);
    if traits.is_empty() {
        cx.span_warn(mitem.span, "empty trait list in `derive`");
    }

    // RFC #1445. `#[derive(PartialEq, Eq)]` adds a (trusted)
    // `#[structural_match]` attribute.
    if traits.iter().filter_map(|t| t.name()).any(|t| t == "PartialEq") &&
       traits.iter().filter_map(|t| t.name()).any(|t| t == "Eq") {
        let structural_match = intern_and_get_ident("structural_match");
        let span = allow_unstable(cx, span, "derive(PartialEq, Eq)");
        let meta = cx.meta_word(span, structural_match);
        item = item.map(|mut i| {
            i.attrs.push(cx.attribute(span, meta));
            i
        });
    }

    // RFC #1521. `Clone` can assume that `Copy` types' clone implementation is
    // the same as the copy implementation.
    //
    // Add a marker attribute here picked up during #[derive(Clone)]
    if traits.iter().filter_map(|t| t.name()).any(|t| t == "Clone") &&
       traits.iter().filter_map(|t| t.name()).any(|t| t == "Copy") {
        let marker = intern_and_get_ident("rustc_copy_clone_marker");
        let span = allow_unstable(cx, span, "derive(Copy, Clone)");
        let meta = cx.meta_word(span, marker);
        item = item.map(|mut i| {
            i.attrs.push(cx.attribute(span, meta));
            i
        });
    }

    let mut other_items = Vec::new();

    let mut iter = traits.iter();
    while let Some(titem) = iter.next() {

        let tword = match titem.word() {
            Some(name) => name,
            None => {
                cx.span_err(titem.span, "malformed `derive` entry");
                continue
            }
        };
        let tname = tword.name();

        // If this is a built-in derive mode, then we expand it immediately
        // here.
        if is_builtin_trait(&tname) {
            let name = intern_and_get_ident(&format!("derive({})", tname));
            let mitem = cx.meta_word(titem.span, name);

            let span = Span {
                expn_id: cx.codemap().record_expansion(codemap::ExpnInfo {
                    call_site: titem.span,
                    callee: codemap::NameAndSpan {
                        format: codemap::MacroAttribute(intern(&format!("derive({})", tname))),
                        span: Some(titem.span),
                        allow_internal_unstable: true,
                    },
                }),
                ..titem.span
            };

            let my_item = Annotatable::Item(item);
            expand_builtin(&tname, cx, span, &mitem, &my_item, &mut |a| {
                other_items.push(a);
            });
            item = my_item.expect_item();

        // Otherwise if this is a `rustc_macro`-style derive mode, we process it
        // here. The logic here is to:
        //
        // 1. Collect the remaining `#[derive]` annotations into a list. If
        //    there are any left, attach a `#[derive]` attribute to the item
        //    that we're currently expanding with the remaining derive modes.
        // 2. Manufacture a `#[derive(Foo)]` attribute to pass to the expander.
        // 3. Expand the current item we're expanding, getting back a list of
        //    items that replace it.
        // 4. Extend the returned list with the current list of items we've
        //    collected so far.
        // 5. Return everything!
        //
        // If custom derive extensions end up threading through the `#[derive]`
        // attribute, we'll get called again later on to continue expanding
        // those modes.
        } else if let Some(ext) = cx.derive_modes.remove(&tname) {
            let remaining_derives = iter.cloned().collect::<Vec<_>>();
            if remaining_derives.len() > 0 {
                let list = cx.meta_list(titem.span,
                                        intern_and_get_ident("derive"),
                                        remaining_derives);
                let attr = cx.attribute(titem.span, list);
                item = item.map(|mut i| {
                    i.attrs.push(attr);
                    i
                });
            }
            let titem = cx.meta_list_item_word(titem.span, tname.clone());
            let mitem = cx.meta_list(titem.span,
                                     intern_and_get_ident("derive"),
                                     vec![titem]);
            let item = Annotatable::Item(item);
            let mut items = ext.expand(cx, mitem.span, &mitem, item);
            items.extend(other_items);
            cx.derive_modes.insert(tname.clone(), ext);
            return items

        // If we've gotten this far then it means that we're in the territory of
        // the old custom derive mechanism. If the feature isn't enabled, we
        // issue an error, otherwise manufacture the `derive_Foo` attribute.
        } else if !cx.ecfg.enable_custom_derive() {
            feature_gate::emit_feature_err(&cx.parse_sess.span_diagnostic,
                                           "custom_derive",
                                           titem.span,
                                           feature_gate::GateIssue::Language,
                                           feature_gate::EXPLAIN_CUSTOM_DERIVE);
        } else {
            let name = intern_and_get_ident(&format!("derive_{}", tname));
            let mitem = cx.meta_word(titem.span, name);
            item = item.map(|mut i| {
                i.attrs.push(cx.attribute(mitem.span, mitem));
                i
            });
        }
    }

    other_items.insert(0, Annotatable::Item(item));
    return other_items
}

macro_rules! derive_traits {
    ($( $name:expr => $func:path, )+) => {
        pub fn register_all(env: &mut SyntaxEnv) {
            env.insert(intern("derive"), MultiModifier(Box::new(expand_derive)));
        }

        pub fn is_builtin_trait(name: &str) -> bool {
            match name {
                $( $name )|+ => true,
                _ => false,
            }
        }

        fn expand_builtin(name: &str,
                          ecx: &mut ExtCtxt,
                          span: Span,
                          mitem: &MetaItem,
                          item: &Annotatable,
                          push: &mut FnMut(Annotatable)) {
            match name {
                $(
                    $name => {
                        warn_if_deprecated(ecx, span, $name);
                        $func(ecx, span, mitem, item, push);
                    }
                )*
                _ => panic!("not a builtin derive mode: {}", name),
            }
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
                  span: Span,
                  intrinsic: &str,
                  args: Vec<P<ast::Expr>>)
                  -> P<ast::Expr> {
    let path = cx.std_path(&["intrinsics", intrinsic]);
    let call = cx.expr_call_global(span, path, args);

    cx.expr_block(P(ast::Block {
        stmts: vec![cx.stmt_expr(call)],
        id: ast::DUMMY_NODE_ID,
        rules: ast::BlockCheckMode::Unsafe(ast::CompilerGenerated),
        span: span,
    }))
}
