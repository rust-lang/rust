// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Syntax extensions in the Rust compiler.

#![crate_name = "syntax_ext"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]
#![cfg_attr(not(stage0), deny(warnings))]

#![feature(dotdot_in_tuple_patterns)]
#![feature(rustc_macro_lib)]
#![feature(rustc_macro_internals)]
#![feature(rustc_private)]
#![feature(staged_api)]

extern crate fmt_macros;
#[macro_use]
extern crate log;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
extern crate rustc_macro;
extern crate rustc_errors as errors;

mod asm;
mod cfg;
mod concat;
mod concat_idents;
mod env;
mod format;
mod log_syntax;
mod trace_macros;

pub mod rustc_macro_registrar;

// for custom_derive
pub mod deriving;

use std::rc::Rc;
use syntax::ast;
use syntax::ext::base::{MacroExpanderFn, NormalTT, IdentTT, MultiModifier};
use syntax::ext::hygiene::Mark;
use syntax::ext::tt::macro_rules::MacroRulesExpander;
use syntax::parse::token::intern;

pub fn register_builtins(resolver: &mut syntax::ext::base::Resolver, enable_quotes: bool) {
    let mut register = |name, ext| {
        resolver.add_macro(Mark::root(), ast::Ident::with_empty_ctxt(intern(name)), Rc::new(ext));
    };

    register("macro_rules", IdentTT(Box::new(MacroRulesExpander), None, false));

    macro_rules! register {
        ($( $name:ident: $f:expr, )*) => { $(
            register(stringify!($name), NormalTT(Box::new($f as MacroExpanderFn), None, false));
        )* }
    }

    if enable_quotes {
        use syntax::ext::quote::*;
        register! {
            quote_tokens: expand_quote_tokens,
            quote_expr: expand_quote_expr,
            quote_ty: expand_quote_ty,
            quote_item: expand_quote_item,
            quote_pat: expand_quote_pat,
            quote_arm: expand_quote_arm,
            quote_stmt: expand_quote_stmt,
            quote_matcher: expand_quote_matcher,
            quote_attr: expand_quote_attr,
            quote_arg: expand_quote_arg,
            quote_block: expand_quote_block,
            quote_meta_item: expand_quote_meta_item,
            quote_path: expand_quote_path,
        }
    }

    use syntax::ext::source_util::*;
    register! {
        line: expand_line,
        column: expand_column,
        file: expand_file,
        stringify: expand_stringify,
        include: expand_include,
        include_str: expand_include_str,
        include_bytes: expand_include_bytes,
        module_path: expand_mod,

        asm: asm::expand_asm,
        cfg: cfg::expand_cfg,
        concat: concat::expand_syntax_ext,
        concat_idents: concat_idents::expand_syntax_ext,
        env: env::expand_env,
        option_env: env::expand_option_env,
        log_syntax: log_syntax::expand_syntax_ext,
        trace_macros: trace_macros::expand_trace_macros,
    }

    // format_args uses `unstable` things internally.
    register("format_args", NormalTT(Box::new(format::expand_format_args), None, true));

    register("derive", MultiModifier(Box::new(deriving::expand_derive)));
}
