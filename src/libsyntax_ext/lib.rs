//! Syntax extensions in the Rust compiler.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![deny(rust_2018_idioms)]
#![deny(unused_lifetimes)]

#![feature(in_band_lifetimes)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_internals)]
#![feature(proc_macro_span)]
#![feature(decl_macro)]
#![feature(nll)]
#![feature(rustc_diagnostic_macros)]

#![recursion_limit="256"]

extern crate proc_macro;

mod error_codes;

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
use syntax::attr::Stability;
use syntax::ext::base::MacroExpanderFn;
use syntax::ext::base::{NamedSyntaxExtension, SyntaxExtension, SyntaxExtensionKind};
use syntax::edition::Edition;
use syntax::symbol::{sym, Symbol};

const EXPLAIN_ASM: &str =
    "inline assembly is not stable enough for use and is subject to change";
const EXPLAIN_GLOBAL_ASM: &str =
    "`global_asm!` is not stable enough for use and is subject to change";
const EXPLAIN_CUSTOM_TEST_FRAMEWORKS: &str =
    "custom test frameworks are an unstable feature";
const EXPLAIN_LOG_SYNTAX: &str =
    "`log_syntax!` is not stable enough for use and is subject to change";
const EXPLAIN_CONCAT_IDENTS: &str =
    "`concat_idents` is not stable enough for use and is subject to change";
const EXPLAIN_FORMAT_ARGS_NL: &str =
    "`format_args_nl` is only for internal language use and is subject to change";
const EXPLAIN_TRACE_MACROS: &str =
    "`trace_macros` is not stable enough for use and is subject to change";
const EXPLAIN_UNSTABLE_COLUMN: &str =
    "internal implementation detail of the `column` macro";

pub fn register_builtins(resolver: &mut dyn syntax::ext::base::Resolver,
                         user_exts: Vec<NamedSyntaxExtension>,
                         edition: Edition) {
    deriving::register_builtin_derives(resolver, edition);

    let mut register = |name, ext| {
        resolver.add_builtin(ast::Ident::with_empty_ctxt(name), Lrc::new(ext));
    };
    macro_rules! register {
        ($( $name:ident: $f:expr, )*) => { $(
            register(sym::$name, SyntaxExtension::default(
                SyntaxExtensionKind::LegacyBang(Box::new($f as MacroExpanderFn)), edition
            ));
        )* }
    }
    macro_rules! register_unstable {
        ($( [$feature:expr, $reason:expr, $issue:expr] $name:ident: $f:expr, )*) => { $(
            register(sym::$name, SyntaxExtension {
                stability: Some(Stability::unstable(
                    $feature, Some(Symbol::intern($reason)), $issue
                )),
                ..SyntaxExtension::default(
                    SyntaxExtensionKind::LegacyBang(Box::new($f as MacroExpanderFn)), edition
                )
            });
        )* }
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
        cfg: cfg::expand_cfg,
        concat: concat::expand_syntax_ext,
        env: env::expand_env,
        option_env: env::expand_option_env,
        compile_error: compile_error::expand_compile_error,
        assert: assert::expand_assert,
    }

    register_unstable! {
        [sym::__rust_unstable_column, EXPLAIN_UNSTABLE_COLUMN, 0]
        __rust_unstable_column: expand_column,
        [sym::asm, EXPLAIN_ASM, 29722]
        asm: asm::expand_asm,
        [sym::global_asm, EXPLAIN_GLOBAL_ASM, 35119]
        global_asm: global_asm::expand_global_asm,
        [sym::concat_idents, EXPLAIN_CONCAT_IDENTS, 29599]
        concat_idents: concat_idents::expand_syntax_ext,
        [sym::log_syntax, EXPLAIN_LOG_SYNTAX, 29598]
        log_syntax: log_syntax::expand_syntax_ext,
        [sym::trace_macros, EXPLAIN_TRACE_MACROS, 29598]
        trace_macros: trace_macros::expand_trace_macros,
    }

    register(sym::test_case, SyntaxExtension {
        stability: Some(Stability::unstable(
            sym::custom_test_frameworks,
            Some(Symbol::intern(EXPLAIN_CUSTOM_TEST_FRAMEWORKS)),
            50297,
        )),
        ..SyntaxExtension::default(
            SyntaxExtensionKind::LegacyAttr(Box::new(test_case::expand)), edition
        )
    });
    register(sym::test, SyntaxExtension::default(
        SyntaxExtensionKind::LegacyAttr(Box::new(test::expand_test)), edition
    ));
    register(sym::bench, SyntaxExtension::default(
        SyntaxExtensionKind::LegacyAttr(Box::new(test::expand_bench)), edition
    ));

    // format_args uses `unstable` things internally.
    let allow_internal_unstable = Some([sym::fmt_internals][..].into());
    register(sym::format_args, SyntaxExtension {
        allow_internal_unstable: allow_internal_unstable.clone(),
        ..SyntaxExtension::default(
            SyntaxExtensionKind::LegacyBang(Box::new(format::expand_format_args)), edition
        )
    });
    register(sym::format_args_nl, SyntaxExtension {
        stability: Some(Stability::unstable(
            sym::format_args_nl,
            Some(Symbol::intern(EXPLAIN_FORMAT_ARGS_NL)),
            0,
        )),
        allow_internal_unstable,
        ..SyntaxExtension::default(
            SyntaxExtensionKind::LegacyBang(Box::new(format::expand_format_args_nl)), edition
        )
    });

    for (name, ext) in user_exts {
        register(name, ext);
    }
}
