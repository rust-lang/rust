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
use syntax::attr::HasAttrs;
use syntax::codemap;
use syntax::ext::base::{Annotatable, ExtCtxt, SyntaxExtension};
use syntax::ext::build::AstBuilder;
use syntax::feature_gate;
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

pub fn expand_derive(cx: &mut ExtCtxt,
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

    let derive = Symbol::intern("derive");
    let mut derive_attrs = Vec::new();
    item = item.map_attrs(|attrs| {
        let partition = attrs.into_iter().partition(|attr| attr.name() == derive);
        derive_attrs = partition.0;
        partition.1
    });

    // Expand `#[derive]`s after other attribute macro invocations.
    if cx.resolver.find_attr_invoc(&mut item.attrs.clone()).is_some() {
        return vec![Annotatable::Item(item.map_attrs(|mut attrs| {
            attrs.push(cx.attribute(span, mitem.clone()));
            attrs.extend(derive_attrs);
            attrs
        }))];
    }

    let get_traits = |mitem: &MetaItem, cx: &ExtCtxt| {
        if mitem.value_str().is_some() {
            cx.span_err(mitem.span, "unexpected value in `derive`");
        }

        let traits = mitem.meta_item_list().unwrap_or(&[]).to_owned();
        if traits.is_empty() {
            cx.span_warn(mitem.span, "empty trait list in `derive`");
        }
        traits
    };

    let mut traits = get_traits(mitem, cx);
    for derive_attr in derive_attrs {
        traits.extend(get_traits(&derive_attr.value, cx));
    }

    // First, weed out malformed #[derive]
    traits.retain(|titem| {
        if titem.word().is_none() {
            cx.span_err(titem.span, "malformed `derive` entry");
            false
        } else {
            true
        }
    });

    // Next, check for old-style #[derive(Foo)]
    //
    // These all get expanded to `#[derive_Foo]` and will get expanded first. If
    // we actually add any attributes here then we return to get those expanded
    // and then eventually we'll come back to finish off the other derive modes.
    let mut new_attributes = Vec::new();
    traits.retain(|titem| {
        let tword = titem.word().unwrap();
        let tname = tword.name();

        if is_builtin_trait(tname) || {
            let derive_mode = ast::Path::from_ident(titem.span, ast::Ident::with_empty_ctxt(tname));
            cx.resolver.resolve_macro(cx.current_expansion.mark, &derive_mode, false).map(|ext| {
                if let SyntaxExtension::CustomDerive(_) = *ext { true } else { false }
            }).unwrap_or(false)
        } {
            return true;
        }

        if !cx.ecfg.enable_custom_derive() {
            feature_gate::emit_feature_err(&cx.parse_sess,
                                           "custom_derive",
                                           titem.span,
                                           feature_gate::GateIssue::Language,
                                           feature_gate::EXPLAIN_CUSTOM_DERIVE);
        } else {
            let name = Symbol::intern(&format!("derive_{}", tname));
            if !cx.resolver.is_whitelisted_legacy_custom_derive(name) {
                cx.span_warn(titem.span, feature_gate::EXPLAIN_DEPR_CUSTOM_DERIVE);
            }
            let mitem = cx.meta_word(titem.span, name);
            new_attributes.push(cx.attribute(mitem.span, mitem));
        }
        false
    });
    if new_attributes.len() > 0 {
        item = item.map(|mut i| {
            i.attrs.extend(new_attributes);
            if traits.len() > 0 {
                let list = cx.meta_list(mitem.span, derive, traits);
                i.attrs.push(cx.attribute(mitem.span, list));
            }
            i
        });
        return vec![Annotatable::Item(item)]
    }

    // Now check for macros-1.1 style custom #[derive].
    //
    // Expand each of them in order given, but *before* we expand any built-in
    // derive modes. The logic here is to:
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
    let macros_11_derive = traits.iter()
                                 .cloned()
                                 .enumerate()
                                 .filter(|&(_, ref name)| !is_builtin_trait(name.name().unwrap()))
                                 .next();
    if let Some((i, titem)) = macros_11_derive {
        let tname = ast::Ident::with_empty_ctxt(titem.name().unwrap());
        let path = ast::Path::from_ident(titem.span, tname);
        let ext = cx.resolver.resolve_macro(cx.current_expansion.mark, &path, false).unwrap();

        traits.remove(i);
        if traits.len() > 0 {
            item = item.map(|mut i| {
                let list = cx.meta_list(mitem.span, derive, traits);
                i.attrs.push(cx.attribute(mitem.span, list));
                i
            });
        }
        let titem = cx.meta_list_item_word(titem.span, titem.name().unwrap());
        let mitem = cx.meta_list(titem.span, derive, vec![titem]);
        let item = Annotatable::Item(item);

        let span = Span {
            expn_id: cx.codemap().record_expansion(codemap::ExpnInfo {
                call_site: mitem.span,
                callee: codemap::NameAndSpan {
                    format: codemap::MacroAttribute(Symbol::intern(&format!("derive({})", tname))),
                    span: None,
                    allow_internal_unstable: false,
                },
            }),
            ..mitem.span
        };

        if let SyntaxExtension::CustomDerive(ref ext) = *ext {
            return ext.expand(cx, span, &mitem, item);
        } else {
            unreachable!()
        }
    }

    // Ok, at this point we know that there are no old-style `#[derive_Foo]` nor
    // any macros-1.1 style `#[derive(Foo)]`. Expand all built-in traits here.

    // RFC #1445. `#[derive(PartialEq, Eq)]` adds a (trusted)
    // `#[structural_match]` attribute.
    let (partial_eq, eq) = (Symbol::intern("PartialEq"), Symbol::intern("Eq"));
    if traits.iter().any(|t| t.name() == Some(partial_eq)) &&
       traits.iter().any(|t| t.name() == Some(eq)) {
        let structural_match = Symbol::intern("structural_match");
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
    let (copy, clone) = (Symbol::intern("Copy"), Symbol::intern("Clone"));
    if traits.iter().any(|t| t.name() == Some(clone)) &&
       traits.iter().any(|t| t.name() == Some(copy)) {
        let marker = Symbol::intern("rustc_copy_clone_marker");
        let span = allow_unstable(cx, span, "derive(Copy, Clone)");
        let meta = cx.meta_word(span, marker);
        item = item.map(|mut i| {
            i.attrs.push(cx.attribute(span, meta));
            i
        });
    }

    let mut items = Vec::new();
    for titem in traits.iter() {
        let tname = titem.word().unwrap().name();
        let name = Symbol::intern(&format!("derive({})", tname));
        let mitem = cx.meta_word(titem.span, name);

        let span = Span {
            expn_id: cx.codemap().record_expansion(codemap::ExpnInfo {
                call_site: titem.span,
                callee: codemap::NameAndSpan {
                    format: codemap::MacroAttribute(name),
                    span: None,
                    allow_internal_unstable: true,
                },
            }),
            ..titem.span
        };

        let my_item = Annotatable::Item(item);
        expand_builtin(&tname.as_str(), cx, span, &mitem, &my_item, &mut |a| {
            items.push(a);
        });
        item = my_item.expect_item();
    }

    items.insert(0, Annotatable::Item(item));
    return items
}

macro_rules! derive_traits {
    ($( $name:expr => $func:path, )+) => {
        pub fn is_builtin_trait(name: ast::Name) -> bool {
            match &*name.as_str() {
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
