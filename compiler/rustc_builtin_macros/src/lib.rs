//! This crate contains implementations of built-in macros and other code generating facilities
//! injecting code into the crate before it is lowered to HIR.

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(autodiff)]
#![feature(box_patterns)]
#![feature(decl_macro)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(proc_macro_internals)]
#![feature(proc_macro_quote)]
#![feature(rustdoc_internals)]
#![feature(try_blocks)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

extern crate proc_macro;

use rustc_expand::base::{MacroExpanderFn, ResolverExpand, SyntaxExtensionKind};
use rustc_expand::proc_macro::BangProcMacro;
use rustc_span::symbol::sym;

use crate::deriving::*;

mod alloc_error_handler;
mod assert;
mod autodiff;
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
mod errors;
mod format;
mod format_foreign;
mod global_allocator;
mod log_syntax;
mod pattern_type;
mod source_util;
mod test;
mod trace_macros;

pub mod asm;
pub mod cmdline_attrs;
pub mod proc_macro_harness;
pub mod standard_library_imports;
pub mod test_harness;
pub mod util;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

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
        // tidy-alphabetical-start
        asm: asm::expand_asm,
        assert: assert::expand_assert,
        cfg: cfg::expand_cfg,
        column: source_util::expand_column,
        compile_error: compile_error::expand_compile_error,
        concat: concat::expand_concat,
        concat_bytes: concat_bytes::expand_concat_bytes,
        concat_idents: concat_idents::expand_concat_idents,
        const_format_args: format::expand_format_args,
        core_panic: edition_panic::expand_panic,
        env: env::expand_env,
        file: source_util::expand_file,
        format_args: format::expand_format_args,
        format_args_nl: format::expand_format_args_nl,
        global_asm: asm::expand_global_asm,
        include: source_util::expand_include,
        include_bytes: source_util::expand_include_bytes,
        include_str: source_util::expand_include_str,
        line: source_util::expand_line,
        log_syntax: log_syntax::expand_log_syntax,
        module_path: source_util::expand_mod,
        naked_asm: asm::expand_naked_asm,
        option_env: env::expand_option_env,
        pattern_type: pattern_type::expand,
        std_panic: edition_panic::expand_panic,
        stringify: source_util::expand_stringify,
        trace_macros: trace_macros::expand_trace_macros,
        unreachable: edition_panic::expand_unreachable,
        // tidy-alphabetical-end
    }

    register_attr! {
        alloc_error_handler: alloc_error_handler::expand,
        autodiff: autodiff::expand,
        bench: test::expand_bench,
        cfg_accessible: cfg_accessible::Expander,
        cfg_eval: cfg_eval::expand,
        derive: derive::Expander { is_const: false },
        derive_const: derive::Expander { is_const: true },
        global_allocator: global_allocator::expand,
        test: test::expand_test,
        test_case: test::expand_test_case,
    }

    register_derive! {
        Clone: clone::expand_deriving_clone,
        Copy: bounds::expand_deriving_copy,
        ConstParamTy: bounds::expand_deriving_const_param_ty,
        UnsizedConstParamTy: bounds::expand_deriving_unsized_const_param_ty,
        Debug: debug::expand_deriving_debug,
        Default: default::expand_deriving_default,
        Eq: eq::expand_deriving_eq,
        Hash: hash::expand_deriving_hash,
        Ord: ord::expand_deriving_ord,
        PartialEq: partial_eq::expand_deriving_partial_eq,
        PartialOrd: partial_ord::expand_deriving_partial_ord,
        RustcDecodable: decodable::expand_deriving_rustc_decodable,
        RustcEncodable: encodable::expand_deriving_rustc_encodable,
        CoercePointee: coerce_pointee::expand_deriving_coerce_pointee,
    }

    let client = proc_macro::bridge::client::Client::expand1(proc_macro::quote);
    register(sym::quote, SyntaxExtensionKind::Bang(Box::new(BangProcMacro { client })));
}
