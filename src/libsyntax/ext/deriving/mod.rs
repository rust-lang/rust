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
//!
//! FIXME (#2810): hygiene. Search for "__" strings (in other files too). We also assume "extra" is
//! the standard library, and "std" is the core library.

use ast::{Item, MetaItem, MetaWord};
use attr::AttrMetaMethods;
use ext::base::{ExtCtxt, SyntaxEnv, Decorator, ItemDecorator, Modifier};
use ext::build::AstBuilder;
use feature_gate;
use codemap::Span;
use parse::token::{intern, intern_and_get_ident};
use ptr::P;

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
        ::ext::deriving::generic::ty::Path::new_local(stringify!($x))
    )
}

macro_rules! pathvec_std {
    ($cx:expr, $first:ident :: $($rest:ident)::+) => (
        if $cx.use_std {
            pathvec!(std :: $($rest)::+)
        } else {
            pathvec!($first :: $($rest)::+)
        }
    )
}

macro_rules! path_std {
    ($($x:tt)*) => (
        ::ext::deriving::generic::ty::Path::new( pathvec_std!( $($x)* ) )
    )
}

pub mod bounds;
pub mod clone;
pub mod encodable;
pub mod decodable;
pub mod hash;
pub mod show;
pub mod default;
pub mod primitive;

#[path="cmp/eq.rs"]
pub mod eq;
#[path="cmp/totaleq.rs"]
pub mod totaleq;
#[path="cmp/ord.rs"]
pub mod ord;
#[path="cmp/totalord.rs"]
pub mod totalord;


pub mod generic;

fn expand_derive(cx: &mut ExtCtxt,
                 _: Span,
                 mitem: &MetaItem,
                 item: P<Item>) -> P<Item> {
    item.map(|mut item| {
        if mitem.value_str().is_some() {
            cx.span_err(mitem.span, "unexpected value in `derive`");
        }

        let traits = mitem.meta_item_list().unwrap_or(&[]);
        if traits.is_empty() {
            cx.span_warn(mitem.span, "empty trait list in `derive`");
        }

        for titem in traits.iter().rev() {
            let tname = match titem.node {
                MetaWord(ref tname) => tname,
                _ => {
                    cx.span_err(titem.span, "malformed `derive` entry");
                    continue;
                }
            };

            if !(is_builtin_trait(tname) || cx.ecfg.enable_custom_derive()) {
                feature_gate::emit_feature_err(&cx.parse_sess.span_diagnostic,
                                               "custom_derive",
                                               titem.span,
                                               feature_gate::EXPLAIN_CUSTOM_DERIVE);
                continue;
            }

            // #[derive(Foo, Bar)] expands to #[derive_Foo] #[derive_Bar]
            item.attrs.push(cx.attribute(titem.span, cx.meta_word(titem.span,
                intern_and_get_ident(&format!("derive_{}", tname)))));
        }

        item
    })
}

macro_rules! derive_traits {
    ($( $name:expr => $func:path, )*) => {
        pub fn register_all(env: &mut SyntaxEnv) {
            // Define the #[derive_*] extensions.
            $({
                struct DeriveExtension;

                impl ItemDecorator for DeriveExtension {
                    fn expand(&self,
                              ecx: &mut ExtCtxt,
                              sp: Span,
                              mitem: &MetaItem,
                              item: &Item,
                              push: &mut FnMut(P<Item>)) {
                        warn_if_deprecated(ecx, sp, $name);
                        $func(ecx, sp, mitem, item, |i| push(i));
                    }
                }

                env.insert(intern(concat!("derive_", $name)),
                           Decorator(Box::new(DeriveExtension)));
            })*

            env.insert(intern("derive"),
                       Modifier(Box::new(expand_derive)));
        }

        fn is_builtin_trait(name: &str) -> bool {
            match name {
                $( $name )|* => true,
                _ => false,
            }
        }
    }
}

derive_traits! {
    "Clone" => clone::expand_deriving_clone,

    "Hash" => hash::expand_deriving_hash,

    "RustcEncodable" => encodable::expand_deriving_rustc_encodable,

    "RustcDecodable" => decodable::expand_deriving_rustc_decodable,

    "PartialEq" => eq::expand_deriving_eq,
    "Eq" => totaleq::expand_deriving_totaleq,
    "PartialOrd" => ord::expand_deriving_ord,
    "Ord" => totalord::expand_deriving_totalord,

    "Debug" => show::expand_deriving_show,

    "Default" => default::expand_deriving_default,

    "FromPrimitive" => primitive::expand_deriving_from_primitive,

    "Send" => bounds::expand_deriving_unsafe_bound,
    "Sync" => bounds::expand_deriving_unsafe_bound,
    "Copy" => bounds::expand_deriving_copy,

    // deprecated
    "Show" => show::expand_deriving_show,
    "Encodable" => encodable::expand_deriving_encodable,
    "Decodable" => decodable::expand_deriving_decodable,
}

#[inline] // because `name` is a compile-time constant
fn warn_if_deprecated(ecx: &mut ExtCtxt, sp: Span, name: &str) {
    if let Some(replacement) = match name {
        "Show" => Some("Debug"),
        "Encodable" => Some("RustcEncodable"),
        "Decodable" => Some("RustcDecodable"),
        _ => None,
    } {
        ecx.span_warn(sp, &format!("derive({}) is deprecated in favor of derive({})",
                                   name, replacement));
    }
}
