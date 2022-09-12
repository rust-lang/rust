//! This crate contains implementations of built-in macros and other code generating facilities
//! injecting code into the crate before it is lowered to HIR.

#![allow(rustc::potential_query_instability)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(array_windows)]
#![feature(box_patterns)]
#![feature(decl_macro)]
#![feature(if_let_guard)]
#![feature(is_sorted)]
#![feature(let_chains)]
#![feature(let_else)]
#![feature(proc_macro_internals)]
#![feature(proc_macro_quote)]
#![recursion_limit = "256"]

extern crate proc_macro;

#[macro_use]
extern crate tracing;

use crate::deriving::*;

use rustc_expand::base::{MacroExpanderFn, ResolverExpand, SyntaxExtensionKind};
use rustc_expand::proc_macro::BangProcMacro;
use rustc_span::symbol::sym;

mod assert;
mod cfg;
mod cfg_accessible;
mod cfg_eval;
mod compile_error;
mod concat;
mod concat_bytes;
mod concat_idents;
mod derive;
mod deriving;
mod edition_panic;
mod env;
mod format;
mod format_foreign;
mod global_allocator;
mod log_syntax;
mod source_util;
mod test;
mod trace_macros;
mod util;

pub mod asm;
pub mod cmdline_attrs;
pub mod proc_macro_harness;
pub mod standard_library_imports;
pub mod test_harness;

pub fn register_builtin_macros(resolver: &mut dyn ResolverExpand) {
    let mut register = |name, kind| resolver.register_builtin_macro(name, kind);
    macro register_bang($($name:ident: $f:expr,)*) {
        $(register(sym::$name, SyntaxExtensionKind::LegacyBang(Box::new($f as MacroExpanderFn)));)*
    }
    macro register_attr($($name:ident: $f:expr,)*) {
        $(register(sym::$name, SyntaxExtensionKind::LegacyAttr(Box::new($f)));)*
    }
    macro register_derive($($name:ident: $f:expr,)*) {
        $(register(sym::$name, SyntaxExtensionKind::LegacyDerive(Box::new(BuiltinDerive($f))));)*
    }

    register_bang! {
        asm: asm::expand_asm,
        assert: assert::expand_assert,
        cfg: cfg::expand_cfg,
        column: source_util::expand_column,
        compile_error: compile_error::expand_compile_error,
        concat_bytes: concat_bytes::expand_concat_bytes,
        concat_idents: concat_idents::expand_concat_idents,
        concat: concat::expand_concat,
        env: env::expand_env,
        file: source_util::expand_file,
        format_args_nl: format::expand_format_args_nl,
        format_args: format::expand_format_args,
        const_format_args: format::expand_format_args,
        global_asm: asm::expand_global_asm,
        include_bytes: source_util::expand_include_bytes,
        include_str: source_util::expand_include_str,
        include: source_util::expand_include,
        line: source_util::expand_line,
        log_syntax: log_syntax::expand_log_syntax,
        module_path: source_util::expand_mod,
        option_env: env::expand_option_env,
        core_panic: edition_panic::expand_panic,
        std_panic: edition_panic::expand_panic,
        unreachable: edition_panic::expand_unreachable,
        stringify: source_util::expand_stringify,
        trace_macros: trace_macros::expand_trace_macros,
    }

    register_attr! {
        bench: test::expand_bench,
        cfg_accessible: cfg_accessible::Expander,
        cfg_eval: cfg_eval::expand,
        derive: derive::Expander,
        global_allocator: global_allocator::expand,
        test: test::expand_test,
        test_case: test::expand_test_case,
    }

    register_derive! {
        Clone: clone::expand_deriving_clone,
        Copy: bounds::expand_deriving_copy,
        Debug: debug::expand_deriving_debug,
        Default: default::expand_deriving_default,
        Eq: eq::expand_deriving_eq,
        Hash: hash::expand_deriving_hash,
        Ord: ord::expand_deriving_ord,
        PartialEq: partial_eq::expand_deriving_partial_eq,
        PartialOrd: partial_ord::expand_deriving_partial_ord,
        RustcDecodable: decodable::expand_deriving_rustc_decodable,
        RustcEncodable: encodable::expand_deriving_rustc_encodable,
    }

    let client = proc_macro::bridge::client::Client::expand1(proc_macro::quote);
    register(sym::quote, SyntaxExtensionKind::Bang(Box::new(BangProcMacro { client })));
}
