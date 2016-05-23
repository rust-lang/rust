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

use syntax::ast::{MetaItem, MetaItemKind, self};
use syntax::attr::AttrMetaMethods;
use syntax::ext::base::{ExtCtxt, SyntaxEnv, Annotatable};
use syntax::ext::base::{MultiDecorator, MultiItemDecorator, MultiModifier};
use syntax::ext::build::AstBuilder;
use syntax::feature_gate;
use syntax::codemap::{self, Span};
use syntax::parse::token::{intern, intern_and_get_ident};
use syntax::ptr::P;

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
    debug!("expand_derive: span = {:?}", span);
    debug!("expand_derive: mitem = {:?}", mitem);
    debug!("expand_derive: annotatable input  = {:?}", annotatable);
    let annot = annotatable.map_item_or(|item| {
        item.map(|mut item| {
            if mitem.value_str().is_some() {
                cx.span_err(mitem.span, "unexpected value in `derive`");
            }

            let traits = mitem.meta_item_list().unwrap_or(&[]);
            if traits.is_empty() {
                cx.span_warn(mitem.span, "empty trait list in `derive`");
            }

            let mut found_partial_eq = false;
            let mut found_eq = false;

            for titem in traits.iter().rev() {
                let tname = match titem.node {
                    MetaItemKind::Word(ref tname) => tname,
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

                if &tname[..] == "Eq" {
                    found_eq = true;
                } else if &tname[..] == "PartialEq" {
                    found_partial_eq = true;
                }

                let span = Span {
                    expn_id: cx.codemap().record_expansion(codemap::ExpnInfo {
                        call_site: titem.span,
                        callee: codemap::NameAndSpan {
                            format: codemap::MacroAttribute(intern(&format!("derive({})", tname))),
                            span: Some(titem.span),
                            allow_internal_unstable: true,
                        },
                    }), ..titem.span
                };

                // #[derive(Foo, Bar)] expands to #[derive_Foo] #[derive_Bar]
                item.attrs.push(cx.attribute(span, cx.meta_word(titem.span,
                    intern_and_get_ident(&format!("derive_{}", tname)))));
            }

            // RFC #1445. `#[derive(PartialEq, Eq)]` adds a (trusted)
            // `#[structural_match]` attribute.
            if found_partial_eq && found_eq {
                // This span is **very** sensitive and crucial to
                // getting the stability behavior we want. What we are
                // doing is marking `#[structural_match]` with the
                // span of the `#[deriving(...)]` attribute (the
                // entire attribute, not just the `PartialEq` or `Eq`
                // part), but with the current backtrace. The current
                // backtrace will contain a topmost entry that IS this
                // `#[deriving(...)]` attribute and with the
                // "allow-unstable" flag set to true.
                //
                // Note that we do NOT use the span of the `Eq`
                // text itself. You might think this is
                // equivalent, because the `Eq` appears within the
                // `#[deriving(Eq)]` attribute, and hence we would
                // inherit the "allows unstable" from the
                // backtrace.  But in fact this is not always the
                // case. The actual source text that led to
                // deriving can be `#[$attr]`, for example, where
                // `$attr == deriving(Eq)`. In that case, the
                // "#[structural_match]" would be considered to
                // originate not from the deriving call but from
                // text outside the deriving call, and hence would
                // be forbidden from using unstable
                // content.
                //
                // See tests src/run-pass/rfc1445 for
                // examples. --nmatsakis
                let span = Span { expn_id: cx.backtrace(), .. span };
                assert!(cx.parse_sess.codemap().span_allows_unstable(span));
                debug!("inserting structural_match with span {:?}", span);
                let structural_match = intern_and_get_ident("structural_match");
                item.attrs.push(cx.attribute(span,
                                             cx.meta_word(span,
                                                          structural_match)));
            }

            item
        })
    }, |a| {
        cx.span_err(span, "`derive` can only be applied to items");
        a
    });
    debug!("expand_derive: annotatable output = {:?}", annot);
    annot
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
                        if !ecx.parse_sess.codemap().span_allows_unstable(sp)
                            && !ecx.ecfg.features.unwrap().custom_derive {
                            // FIXME:
                            // https://github.com/rust-lang/rust/pull/32671#issuecomment-206245303
                            // This is just to avoid breakage with syntex.
                            // Remove that to spawn an error instead.
                            let cm = ecx.parse_sess.codemap();
                            let parent = cm.with_expn_info(ecx.backtrace(),
                                                           |info| info.unwrap().call_site.expn_id);
                            cm.with_expn_info(parent, |info| {
                                if info.is_some() {
                                    let mut w = ecx.parse_sess.span_diagnostic.struct_span_warn(
                                        sp, feature_gate::EXPLAIN_DERIVE_UNDERSCORE,
                                    );
                                    if option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_none() {
                                        w.help(
                                            &format!("add #![feature(custom_derive)] to \
                                                      the crate attributes to enable")
                                        );
                                    }
                                    w.emit();
                                } else {
                                    feature_gate::emit_feature_err(
                                        &ecx.parse_sess.span_diagnostic,
                                        "custom_derive", sp, feature_gate::GateIssue::Language,
                                        feature_gate::EXPLAIN_DERIVE_UNDERSCORE
                                    );

                                    return;
                                }
                            })
                        }

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
                  args: Vec<P<ast::Expr>>) -> P<ast::Expr> {
    let path = cx.std_path(&["intrinsics", intrinsic]);
    let call = cx.expr_call_global(span, path, args);

    cx.expr_block(P(ast::Block {
        stmts: vec![],
        expr: Some(call),
        id: ast::DUMMY_NODE_ID,
        rules: ast::BlockCheckMode::Unsafe(ast::CompilerGenerated),
        span: span }))
}

