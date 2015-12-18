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

use syntax::ast::{MetaItem, MetaWord};
use syntax::attr::AttrMetaMethods;
use syntax::ext::base::{ExtCtxt, SyntaxEnv, Annotatable};
use syntax::ext::base::{MultiDecorator, MultiItemDecorator, MultiModifier};
use syntax::ext::build::AstBuilder;
use syntax::feature_gate;
use syntax::codemap::Span;
use syntax::parse::token::{intern, intern_and_get_ident};

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
pub mod primitive;

#[path="cmp/partial_eq.rs"]
pub mod partial_eq;
#[path="cmp/eq.rs"]
pub mod eq;
#[path="cmp/partial_ord.rs"]
pub mod partial_ord;
#[path="cmp/ord.rs"]
pub mod ord;


pub mod generic;

fn expand_derive(cx: &mut ExtCtxt,
                 span: Span,
                 mitem: &MetaItem,
                 annotatable: Annotatable)
                 -> Annotatable {
    annotatable.map_item_or(|item| {
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
                                                   feature_gate::GateIssue::Language,
                                                   feature_gate::EXPLAIN_CUSTOM_DERIVE);
                    continue;
                }

                // #[derive(Foo, Bar)] expands to #[derive_Foo] #[derive_Bar]
                item.attrs.push(cx.attribute(titem.span, cx.meta_word(titem.span,
                    intern_and_get_ident(&format!("derive_{}", tname)))));
            }

            item
        })
    }, |a| {
        cx.span_err(span, "`derive` can only be applied to items");
        a
    })
}

macro_rules! derive_traits {
    ($( $name:expr => $func:path, )+) => {
        pub fn register_all(env: &mut SyntaxEnv) {
            // Define the #[derive_*] extensions.
            $({
                struct DeriveExtension;

                impl MultiItemDecorator for DeriveExtension {
                    fn expand(&self,
                              ecx: &mut ExtCtxt,
                              sp: Span,
                              mitem: &MetaItem,
                              annotatable: &Annotatable,
                              push: &mut FnMut(Annotatable)) {
                        warn_if_deprecated(ecx, sp, $name);
                        $func(ecx, sp, mitem, annotatable, push);
                    }
                }

                env.insert(intern(concat!("derive_", $name)),
                           MultiDecorator(Box::new(DeriveExtension)));
            })+

            env.insert(intern("derive"),
                       MultiModifier(Box::new(expand_derive)));
        }

        fn is_builtin_trait(name: &str) -> bool {
            match name {
                $( $name )|+ => true,
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

    "PartialEq" => partial_eq::expand_deriving_partial_eq,
    "Eq" => eq::expand_deriving_eq,
    "PartialOrd" => partial_ord::expand_deriving_partial_ord,
    "Ord" => ord::expand_deriving_ord,

    "Debug" => debug::expand_deriving_debug,

    "Default" => default::expand_deriving_default,

    "FromPrimitive" => primitive::expand_deriving_from_primitive,

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
        ecx.span_warn(sp, &format!("derive({}) is deprecated in favor of derive({})",
                                   name, replacement));
    }
}
