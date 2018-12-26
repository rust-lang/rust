//! Syntax extensions in the Rust compiler.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(in_band_lifetimes)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_internals)]
#![feature(proc_macro_span)]
#![feature(decl_macro)]
#![feature(nll)]
#![feature(str_escape)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]

#![recursion_limit="256"]

extern crate fmt_macros;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
extern crate proc_macro;
extern crate rustc_data_structures;
extern crate rustc_errors as errors;
extern crate rustc_target;
#[macro_use]
extern crate smallvec;
#[macro_use]
extern crate log;

mod diagnostics;

mod asm;
mod assert;
mod cfg;
mod compile_error;
mod concat;
mod concat_idents;
mod env;
mod format;
mod format_foreign;
mod global_asm;
mod log_syntax;
mod proc_macro_server;
mod test;
mod test_case;
mod trace_macros;

pub mod deriving;
pub mod proc_macro_decls;
pub mod proc_macro_impl;

use rustc_data_structures::sync::Lrc;
use syntax::ast;
use syntax::ext::base::{MacroExpanderFn, NormalTT, NamedSyntaxExtension, MultiModifier};
use syntax::ext::hygiene;
use syntax::symbol::Symbol;

pub fn register_builtins(resolver: &mut dyn syntax::ext::base::Resolver,
                         user_exts: Vec<NamedSyntaxExtension>,
                         enable_quotes: bool) {
    deriving::register_builtin_derives(resolver);

    let mut register = |name, ext| {
        resolver.add_builtin(ast::Ident::with_empty_ctxt(name), Lrc::new(ext));
    };

    macro_rules! register {
        ($( $name:ident: $f:expr, )*) => { $(
            register(Symbol::intern(stringify!($name)),
                     NormalTT {
                        expander: Box::new($f as MacroExpanderFn),
                        def_info: None,
                        allow_internal_unstable: false,
                        allow_internal_unsafe: false,
                        local_inner_macros: false,
                        unstable_feature: None,
                        edition: hygiene::default_edition(),
                    });
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
        __rust_unstable_column: expand_column_gated,
        column: expand_column,
        file: expand_file,
        stringify: expand_stringify,
        include: expand_include,
        include_str: expand_include_str,
        include_bytes: expand_include_bytes,
        module_path: expand_mod,

        asm: asm::expand_asm,
        global_asm: global_asm::expand_global_asm,
        cfg: cfg::expand_cfg,
        concat: concat::expand_syntax_ext,
        concat_idents: concat_idents::expand_syntax_ext,
        env: env::expand_env,
        option_env: env::expand_option_env,
        log_syntax: log_syntax::expand_syntax_ext,
        trace_macros: trace_macros::expand_trace_macros,
        compile_error: compile_error::expand_compile_error,
        assert: assert::expand_assert,
    }

    register(Symbol::intern("test_case"), MultiModifier(Box::new(test_case::expand)));
    register(Symbol::intern("test"), MultiModifier(Box::new(test::expand_test)));
    register(Symbol::intern("bench"), MultiModifier(Box::new(test::expand_bench)));

    // format_args uses `unstable` things internally.
    register(Symbol::intern("format_args"),
             NormalTT {
                expander: Box::new(format::expand_format_args),
                def_info: None,
                allow_internal_unstable: true,
                allow_internal_unsafe: false,
                local_inner_macros: false,
                unstable_feature: None,
                edition: hygiene::default_edition(),
            });
    register(Symbol::intern("format_args_nl"),
             NormalTT {
                 expander: Box::new(format::expand_format_args_nl),
                 def_info: None,
                 allow_internal_unstable: true,
                 allow_internal_unsafe: false,
                 local_inner_macros: false,
                 unstable_feature: None,
                 edition: hygiene::default_edition(),
             });

    for (name, ext) in user_exts {
        register(name, ext);
    }
}
