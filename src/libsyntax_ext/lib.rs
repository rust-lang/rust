//! This crate contains implementations of built-in macros and other code generating facilities
//! injecting code into the crate before it is lowered to HIR.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(crate_visibility_modifier)]
#![feature(decl_macro)]
#![feature(mem_take)]
#![feature(nll)]
#![feature(rustc_diagnostic_macros)]

use crate::deriving::*;

use syntax::ast::Ident;
use syntax::edition::Edition;
use syntax::ext::base::{SyntaxExtension, SyntaxExtensionKind, MacroExpanderFn};
use syntax::symbol::sym;

mod error_codes;

mod asm;
mod assert;
mod cfg;
mod compile_error;
mod concat;
mod concat_idents;
mod deriving;
mod env;
mod format;
mod format_foreign;
mod global_allocator;
mod global_asm;
mod log_syntax;
mod source_util;
mod test;
mod trace_macros;

pub mod plugin_macro_defs;
pub mod proc_macro_harness;
pub mod standard_library_imports;
pub mod test_harness;

pub fn register_builtin_macros(resolver: &mut dyn syntax::ext::base::Resolver, edition: Edition) {
    let mut register = |name, kind| resolver.register_builtin_macro(
        Ident::with_empty_ctxt(name), SyntaxExtension {
            is_builtin: true, ..SyntaxExtension::default(kind, edition)
        },
    );
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
        __rust_unstable_column: source_util::expand_column,
        asm: asm::expand_asm,
        assert: assert::expand_assert,
        cfg: cfg::expand_cfg,
        column: source_util::expand_column,
        compile_error: compile_error::expand_compile_error,
        concat_idents: concat_idents::expand_syntax_ext,
        concat: concat::expand_syntax_ext,
        env: env::expand_env,
        file: source_util::expand_file,
        format_args_nl: format::expand_format_args_nl,
        format_args: format::expand_format_args,
        global_asm: global_asm::expand_global_asm,
        include_bytes: source_util::expand_include_bytes,
        include_str: source_util::expand_include_str,
        include: source_util::expand_include,
        line: source_util::expand_line,
        log_syntax: log_syntax::expand_syntax_ext,
        module_path: source_util::expand_mod,
        option_env: env::expand_option_env,
        stringify: source_util::expand_stringify,
        trace_macros: trace_macros::expand_trace_macros,
    }

    register_attr! {
        bench: test::expand_bench,
        global_allocator: global_allocator::expand,
        test: test::expand_test,
        test_case: test::expand_test_case,
    }

    register_derive! {
        Clone: clone::expand_deriving_clone,
        Copy: bounds::expand_deriving_copy,
        Debug: debug::expand_deriving_debug,
        Decodable: decodable::expand_deriving_decodable,
        Default: default::expand_deriving_default,
        Encodable: encodable::expand_deriving_encodable,
        Eq: eq::expand_deriving_eq,
        Hash: hash::expand_deriving_hash,
        Ord: ord::expand_deriving_ord,
        PartialEq: partial_eq::expand_deriving_partial_eq,
        PartialOrd: partial_ord::expand_deriving_partial_ord,
        RustcDecodable: decodable::expand_deriving_rustc_decodable,
        RustcEncodable: encodable::expand_deriving_rustc_encodable,
    }
}
