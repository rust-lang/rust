//! Syntax extensions in the Rust compiler.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![deny(rust_2018_idioms)]
#![deny(internal)]
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

use syntax::ext::base::MacroExpanderFn;
use syntax::ext::base::{NamedSyntaxExtension, SyntaxExtension, SyntaxExtensionKind};
use syntax::edition::Edition;
use syntax::symbol::{sym, Symbol};

pub fn register_builtins(resolver: &mut dyn syntax::ext::base::Resolver,
                         user_exts: Vec<NamedSyntaxExtension>,
                         edition: Edition) {
    deriving::register_builtin_derives(resolver, edition);

    let mut register = |name, ext| {
        resolver.add_builtin(ast::Ident::with_empty_ctxt(name), Lrc::new(ext));
    };
    macro_rules! register {
        ($( $name:ident: $f:expr, )*) => { $(
            register(Symbol::intern(stringify!($name)), SyntaxExtension::default(
                SyntaxExtensionKind::LegacyBang(Box::new($f as MacroExpanderFn)), edition
            ));
        )* }
    }
    macro_rules! register_attr {
        ($( $name:ident: $f:expr, )*) => { $(
            register(Symbol::intern(stringify!($name)), SyntaxExtension::default(
                SyntaxExtensionKind::LegacyAttr(Box::new($f)), edition
            ));
        )* }
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

    register_attr! {
        test_case: test_case::expand,
        test: test::expand_test,
        bench: test::expand_bench,
    }

    // format_args uses `unstable` things internally.
    let allow_internal_unstable = Some([sym::fmt_internals][..].into());
    register(Symbol::intern("format_args"), SyntaxExtension {
        allow_internal_unstable: allow_internal_unstable.clone(),
        ..SyntaxExtension::default(
            SyntaxExtensionKind::LegacyBang(Box::new(format::expand_format_args)), edition
        )
    });
    register(sym::format_args_nl, SyntaxExtension {
        allow_internal_unstable,
        ..SyntaxExtension::default(
            SyntaxExtensionKind::LegacyBang(Box::new(format::expand_format_args_nl)), edition
        )
    });

    for (name, ext) in user_exts {
        register(name, ext);
    }
}
