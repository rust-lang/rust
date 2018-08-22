// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Feature gating
//!
//! This module implements the gating necessary for preventing certain compiler
//! features from being used by default. This module will crawl a pre-expanded
//! AST to ensure that there are no features which are used that are not
//! enabled.
//!
//! Features are enabled in programs via the crate-level attributes of
//! `#![feature(...)]` with a comma-separated list of features.
//!
//! For the purpose of future feature-tracking, once code for detection of feature
//! gate usage is added, *do not remove it again* even once the feature
//! becomes stable.

use self::AttributeType::*;
use self::AttributeGate::*;

use rustc_data_structures::fx::FxHashMap;
use rustc_target::spec::abi::Abi;
use ast::{self, NodeId, PatKind, RangeEnd};
use attr;
use source_map::Spanned;
use edition::{ALL_EDITIONS, Edition};
use syntax_pos::{Span, DUMMY_SP};
use errors::{DiagnosticBuilder, Handler};
use visit::{self, FnKind, Visitor};
use parse::ParseSess;
use symbol::{keywords, Symbol};

use std::{env, path};

macro_rules! set {
    ($field: ident) => {{
        fn f(features: &mut Features, _: Span) {
            features.$field = true;
        }
        f as fn(&mut Features, Span)
    }}
}

macro_rules! declare_features {
    ($((active, $feature: ident, $ver: expr, $issue: expr, $edition: expr),)+) => {
        /// Represents active features that are currently being implemented or
        /// currently being considered for addition/removal.
        const ACTIVE_FEATURES:
                &'static [(&'static str, &'static str, Option<u32>,
                           Option<Edition>, fn(&mut Features, Span))] =
            &[$((stringify!($feature), $ver, $issue, $edition, set!($feature))),+];

        /// A set of features to be used by later passes.
        #[derive(Clone)]
        pub struct Features {
            /// `#![feature]` attrs for language features, for error reporting
            pub declared_lang_features: Vec<(Symbol, Span, Option<Symbol>)>,
            /// `#![feature]` attrs for non-language (library) features
            pub declared_lib_features: Vec<(Symbol, Span)>,
            $(pub $feature: bool),+
        }

        impl Features {
            pub fn new() -> Features {
                Features {
                    declared_lang_features: Vec::new(),
                    declared_lib_features: Vec::new(),
                    $($feature: false),+
                }
            }

            pub fn walk_feature_fields<F>(&self, mut f: F)
                where F: FnMut(&str, bool)
            {
                $(f(stringify!($feature), self.$feature);)+
            }
        }
    };

    ($((removed, $feature: ident, $ver: expr, $issue: expr, None, $reason: expr),)+) => {
        /// Represents unstable features which have since been removed (it was once Active)
        const REMOVED_FEATURES: &[(&str, &str, Option<u32>, Option<&str>)] = &[
            $((stringify!($feature), $ver, $issue, $reason)),+
        ];
    };

    ($((stable_removed, $feature: ident, $ver: expr, $issue: expr, None),)+) => {
        /// Represents stable features which have since been removed (it was once Accepted)
        const STABLE_REMOVED_FEATURES: &[(&str, &str, Option<u32>, Option<&str>)] = &[
            $((stringify!($feature), $ver, $issue, None)),+
        ];
    };

    ($((accepted, $feature: ident, $ver: expr, $issue: expr, None),)+) => {
        /// Those language feature has since been Accepted (it was once Active)
        const ACCEPTED_FEATURES: &[(&str, &str, Option<u32>, Option<&str>)] = &[
            $((stringify!($feature), $ver, $issue, None)),+
        ];
    }
}

// If you change this, please modify src/doc/unstable-book as well.
//
// Don't ever remove anything from this list; set them to 'Removed'.
//
// The version numbers here correspond to the version in which the current status
// was set. This is most important for knowing when a particular feature became
// stable (active).
//
// NB: tools/tidy/src/features.rs parses this information directly out of the
// source, so take care when modifying it.

declare_features! (
    (active, asm, "1.0.0", Some(29722), None),
    (active, concat_idents, "1.0.0", Some(29599), None),
    (active, link_args, "1.0.0", Some(29596), None),
    (active, log_syntax, "1.0.0", Some(29598), None),
    (active, non_ascii_idents, "1.0.0", Some(28979), None),
    (active, plugin_registrar, "1.0.0", Some(29597), None),
    (active, thread_local, "1.0.0", Some(29594), None),
    (active, trace_macros, "1.0.0", Some(29598), None),

    // rustc internal, for now
    (active, intrinsics, "1.0.0", None, None),
    (active, lang_items, "1.0.0", None, None),
    (active, format_args_nl, "1.29.0", None, None),

    (active, link_llvm_intrinsics, "1.0.0", Some(29602), None),
    (active, linkage, "1.0.0", Some(29603), None),
    (active, quote, "1.0.0", Some(29601), None),

    // rustc internal
    (active, rustc_diagnostic_macros, "1.0.0", None, None),
    (active, rustc_const_unstable, "1.0.0", None, None),
    (active, box_syntax, "1.0.0", Some(49733), None),
    (active, unboxed_closures, "1.0.0", Some(29625), None),

    (active, fundamental, "1.0.0", Some(29635), None),
    (active, main, "1.0.0", Some(29634), None),
    (active, needs_allocator, "1.4.0", Some(27389), None),
    (active, on_unimplemented, "1.0.0", Some(29628), None),
    (active, plugin, "1.0.0", Some(29597), None),
    (active, simd_ffi, "1.0.0", Some(27731), None),
    (active, start, "1.0.0", Some(29633), None),
    (active, structural_match, "1.8.0", Some(31434), None),
    (active, panic_runtime, "1.10.0", Some(32837), None),
    (active, needs_panic_runtime, "1.10.0", Some(32837), None),

    // OIBIT specific features
    (active, optin_builtin_traits, "1.0.0", Some(13231), None),

    // Allows use of #[staged_api]
    //
    // rustc internal
    (active, staged_api, "1.0.0", None, None),

    // Allows using #![no_core]
    (active, no_core, "1.3.0", Some(29639), None),

    // Allows using `box` in patterns; RFC 469
    (active, box_patterns, "1.0.0", Some(29641), None),

    // Allows using the unsafe_destructor_blind_to_params attribute;
    // RFC 1238
    (active, dropck_parametricity, "1.3.0", Some(28498), None),

    // Allows using the may_dangle attribute; RFC 1327
    (active, dropck_eyepatch, "1.10.0", Some(34761), None),

    // Allows the use of custom attributes; RFC 572
    (active, custom_attribute, "1.0.0", Some(29642), None),

    // Allows the use of #[derive(Anything)] as sugar for
    // #[derive_Anything].
    (active, custom_derive, "1.0.0", Some(29644), None),

    // Allows the use of rustc_* attributes; RFC 572
    (active, rustc_attrs, "1.0.0", Some(29642), None),

    // Allows the use of non lexical lifetimes; RFC 2094
    (active, nll, "1.0.0", Some(43234), None),

    // Allows the use of #[allow_internal_unstable]. This is an
    // attribute on macro_rules! and can't use the attribute handling
    // below (it has to be checked before expansion possibly makes
    // macros disappear).
    //
    // rustc internal
    (active, allow_internal_unstable, "1.0.0", None, None),

    // Allows the use of #[allow_internal_unsafe]. This is an
    // attribute on macro_rules! and can't use the attribute handling
    // below (it has to be checked before expansion possibly makes
    // macros disappear).
    //
    // rustc internal
    (active, allow_internal_unsafe, "1.0.0", None, None),

    // #23121. Array patterns have some hazards yet.
    (active, slice_patterns, "1.0.0", Some(23121), None),

    // Allows the definition of `const fn` functions.
    (active, const_fn, "1.2.0", Some(24111), None),

    // Allows let bindings and destructuring in `const fn` functions and constants.
    (active, const_let, "1.22.1", Some(48821), None),

    // Allows accessing fields of unions inside const fn
    (active, const_fn_union, "1.27.0", Some(51909), None),

    // Allows casting raw pointers to `usize` during const eval
    (active, const_raw_ptr_to_usize_cast, "1.27.0", Some(51910), None),

    // Allows dereferencing raw pointers during const eval
    (active, const_raw_ptr_deref, "1.27.0", Some(51911), None),

    // Allows comparing raw pointers during const eval
    (active, const_compare_raw_pointers, "1.27.0", Some(53020), None),

    // Allows panicking during const eval (produces compile-time errors)
    (active, const_panic, "1.30.0", Some(51999), None),

    // Allows using #[prelude_import] on glob `use` items.
    //
    // rustc internal
    (active, prelude_import, "1.2.0", None, None),

    // Allows default type parameters to influence type inference.
    (active, default_type_parameter_fallback, "1.3.0", Some(27336), None),

    // Allows associated type defaults
    (active, associated_type_defaults, "1.2.0", Some(29661), None),

    // Allows `repr(simd)`, and importing the various simd intrinsics
    (active, repr_simd, "1.4.0", Some(27731), None),

    // Allows `extern "platform-intrinsic" { ... }`
    (active, platform_intrinsics, "1.4.0", Some(27731), None),

    // Allows `#[unwind(..)]`
    // rustc internal for rust runtime
    (active, unwind_attributes, "1.4.0", None, None),

    // Allows the use of `#[naked]` on functions.
    (active, naked_functions, "1.9.0", Some(32408), None),

    // Allows `#[no_debug]`
    (active, no_debug, "1.5.0", Some(29721), None),

    // Allows `#[omit_gdb_pretty_printer_section]`
    //
    // rustc internal
    (active, omit_gdb_pretty_printer_section, "1.5.0", None, None),

    // Allows cfg(target_vendor = "...").
    (active, cfg_target_vendor, "1.5.0", Some(29718), None),

    // Allow attributes on expressions and non-item statements
    (active, stmt_expr_attributes, "1.6.0", Some(15701), None),

    // allow using type ascription in expressions
    (active, type_ascription, "1.6.0", Some(23416), None),

    // Allows cfg(target_thread_local)
    (active, cfg_target_thread_local, "1.7.0", Some(29594), None),

    // rustc internal
    (active, abi_vectorcall, "1.7.0", None, None),

    // X..Y patterns
    (active, exclusive_range_pattern, "1.11.0", Some(37854), None),

    // impl specialization (RFC 1210)
    (active, specialization, "1.7.0", Some(31844), None),

    // Allows cfg(target_has_atomic = "...").
    (active, cfg_target_has_atomic, "1.9.0", Some(32976), None),

    // The `!` type. Does not imply exhaustive_patterns (below) any more.
    (active, never_type, "1.13.0", Some(35121), None),

    // Allows exhaustive pattern matching on types that contain uninhabited types
    (active, exhaustive_patterns, "1.13.0", Some(51085), None),

    // Allows all literals in attribute lists and values of key-value pairs
    (active, attr_literals, "1.13.0", Some(34981), None),

    // Allows untagged unions `union U { ... }`
    (active, untagged_unions, "1.13.0", Some(32836), None),

    // Used to identify the `compiler_builtins` crate
    // rustc internal
    (active, compiler_builtins, "1.13.0", None, None),

    // Allows #[link(..., cfg(..))]
    (active, link_cfg, "1.14.0", Some(37406), None),

    // `extern "ptx-*" fn()`
    (active, abi_ptx, "1.15.0", Some(38788), None),

    // The `repr(i128)` annotation for enums
    (active, repr128, "1.16.0", Some(35118), None),

    // The `unadjusted` ABI. Perma unstable.
    // rustc internal
    (active, abi_unadjusted, "1.16.0", None, None),

    // Declarative macros 2.0 (`macro`).
    (active, decl_macro, "1.17.0", Some(39412), None),

    // Allows #[link(kind="static-nobundle"...)]
    (active, static_nobundle, "1.16.0", Some(37403), None),

    // `extern "msp430-interrupt" fn()`
    (active, abi_msp430_interrupt, "1.16.0", Some(38487), None),

    // Used to identify crates that contain sanitizer runtimes
    // rustc internal
    (active, sanitizer_runtime, "1.17.0", None, None),

    // Used to identify crates that contain the profiler runtime
    //
    // rustc internal
    (active, profiler_runtime, "1.18.0", None, None),

    // `extern "x86-interrupt" fn()`
    (active, abi_x86_interrupt, "1.17.0", Some(40180), None),

    // Allows the `catch {...}` expression
    (active, catch_expr, "1.17.0", Some(31436), None),

    // Used to preserve symbols (see llvm.used)
    (active, used, "1.18.0", Some(40289), None),

    // Allows module-level inline assembly by way of global_asm!()
    (active, global_asm, "1.18.0", Some(35119), None),

    // Allows overlapping impls of marker traits
    (active, overlapping_marker_traits, "1.18.0", Some(29864), None),

    // rustc internal
    (active, abi_thiscall, "1.19.0", None, None),

    // Allows a test to fail without failing the whole suite
    (active, allow_fail, "1.19.0", Some(42219), None),

    // Allows unsized tuple coercion.
    (active, unsized_tuple_coercion, "1.20.0", Some(42877), None),

    // Generators
    (active, generators, "1.21.0", Some(43122), None),

    // Trait aliases
    (active, trait_alias, "1.24.0", Some(41517), None),

    // rustc internal
    (active, allocator_internals, "1.20.0", None, None),

    // #[doc(cfg(...))]
    (active, doc_cfg, "1.21.0", Some(43781), None),
    // #[doc(masked)]
    (active, doc_masked, "1.21.0", Some(44027), None),
    // #[doc(spotlight)]
    (active, doc_spotlight, "1.22.0", Some(45040), None),
    // #[doc(include="some-file")]
    (active, external_doc, "1.22.0", Some(44732), None),

    // Future-proofing enums/structs with #[non_exhaustive] attribute (RFC 2008)
    (active, non_exhaustive, "1.22.0", Some(44109), None),

    // `crate` as visibility modifier, synonymous to `pub(crate)`
    (active, crate_visibility_modifier, "1.23.0", Some(45388), Some(Edition::Edition2018)),

    // extern types
    (active, extern_types, "1.23.0", Some(43467), None),

    // Allows trait methods with arbitrary self types
    (active, arbitrary_self_types, "1.23.0", Some(44874), None),

    // `crate` in paths
    (active, crate_in_paths, "1.23.0", Some(45477), Some(Edition::Edition2018)),

    // In-band lifetime bindings (e.g. `fn foo(x: &'a u8) -> &'a u8`)
    (active, in_band_lifetimes, "1.23.0", Some(44524), None),

    // Generic associated types (RFC 1598)
    (active, generic_associated_types, "1.23.0", Some(44265), None),

    // Resolve absolute paths as paths from other crates
    (active, extern_absolute_paths, "1.24.0", Some(44660), Some(Edition::Edition2018)),

    // `foo.rs` as an alternative to `foo/mod.rs`
    (active, non_modrs_mods, "1.24.0", Some(44660), Some(Edition::Edition2018)),

    // `extern` in paths
    (active, extern_in_paths, "1.23.0", Some(44660), None),

    // Use `?` as the Kleene "at most one" operator
    (active, macro_at_most_once_rep, "1.25.0", Some(48075), None),

    // Infer outlives requirements; RFC 2093
    (active, infer_outlives_requirements, "1.26.0", Some(44493), None),

    // Infer static outlives requirements; RFC 2093
    (active, infer_static_outlives_requirements, "1.26.0", Some(44493), None),

    // Multiple patterns with `|` in `if let` and `while let`
    (active, if_while_or_patterns, "1.26.0", Some(48215), None),

    // Parentheses in patterns
    (active, pattern_parentheses, "1.26.0", Some(51087), None),

    // Allows `#[repr(packed)]` attribute on structs
    (active, repr_packed, "1.26.0", Some(33158), None),

    // `use path as _;` and `extern crate c as _;`
    (active, underscore_imports, "1.26.0", Some(48216), None),

    // Allows macro invocations in `extern {}` blocks
    (active, macros_in_extern, "1.27.0", Some(49476), None),

    // `existential type`
    (active, existential_type, "1.28.0", Some(34511), None),

    // unstable #[target_feature] directives
    (active, arm_target_feature, "1.27.0", Some(44839), None),
    (active, aarch64_target_feature, "1.27.0", Some(44839), None),
    (active, hexagon_target_feature, "1.27.0", Some(44839), None),
    (active, powerpc_target_feature, "1.27.0", Some(44839), None),
    (active, mips_target_feature, "1.27.0", Some(44839), None),
    (active, avx512_target_feature, "1.27.0", Some(44839), None),
    (active, mmx_target_feature, "1.27.0", Some(44839), None),
    (active, sse4a_target_feature, "1.27.0", Some(44839), None),
    (active, tbm_target_feature, "1.27.0", Some(44839), None),
    (active, wasm_target_feature, "1.30.0", Some(44839), None),

    // Allows macro invocations of the form `#[foo::bar]`
    (active, proc_macro_path_invoc, "1.27.0", Some(38356), None),

    // Allows macro invocations on modules expressions and statements and
    // procedural macros to expand to non-items.
    (active, proc_macro_mod, "1.27.0", Some(38356), None),
    (active, proc_macro_expr, "1.27.0", Some(38356), None),
    (active, proc_macro_non_items, "1.27.0", Some(38356), None),
    (active, proc_macro_gen, "1.27.0", Some(38356), None),

    // #[doc(alias = "...")]
    (active, doc_alias, "1.27.0", Some(50146), None),

    // Access to crate names passed via `--extern` through prelude
    (active, extern_prelude, "1.27.0", Some(44660), Some(Edition::Edition2018)),

    // Scoped attributes
    (active, tool_attributes, "1.25.0", Some(44690), None),
    // Scoped lints
    (active, tool_lints, "1.28.0", Some(44690), None),

    // Allows irrefutable patterns in if-let and while-let statements (RFC 2086)
    (active, irrefutable_let_patterns, "1.27.0", Some(44495), None),

    // Allows use of the :literal macro fragment specifier (RFC 1576)
    (active, macro_literal_matcher, "1.27.0", Some(35625), None),

    // inconsistent bounds in where clauses
    (active, trivial_bounds, "1.28.0", Some(48214), None),

    // 'a: { break 'a; }
    (active, label_break_value, "1.28.0", Some(48594), None),

    // Integer match exhaustiveness checking
    (active, exhaustive_integer_patterns, "1.30.0", Some(50907), None),

    // #[panic_implementation]
    (active, panic_implementation, "1.28.0", Some(44489), None),

    // #[doc(keyword = "...")]
    (active, doc_keyword, "1.28.0", Some(51315), None),

    // Allows async and await syntax
    (active, async_await, "1.28.0", Some(50547), None),

    // #[alloc_error_handler]
    (active, alloc_error_handler, "1.29.0", Some(51540), None),

    (active, abi_amdgpu_kernel, "1.29.0", Some(51575), None),

    // impl<I:Iterator> Iterator for &mut Iterator
    // impl Debug for Foo<'_>
    (active, impl_header_lifetime_elision, "1.30.0", Some(15872), Some(Edition::Edition2018)),

    // Support for arbitrary delimited token streams in non-macro attributes
    (active, unrestricted_attribute_tokens, "1.30.0", Some(44690), None),

    // Allows `use x::y;` to resolve through `self::x`, not just `::x`
    (active, uniform_paths, "1.30.0", Some(53130), None),

    // Allows `Self` in type definitions
    (active, self_in_typedefs, "1.30.0", Some(49303), None),

    // unsized rvalues at arguments and parameters
    (active, unsized_locals, "1.30.0", Some(48055), None),
);

declare_features! (
    (removed, import_shadowing, "1.0.0", None, None, None),
    (removed, managed_boxes, "1.0.0", None, None, None),
    // Allows use of unary negate on unsigned integers, e.g. -e for e: u8
    (removed, negate_unsigned, "1.0.0", Some(29645), None, None),
    (removed, reflect, "1.0.0", Some(27749), None, None),
    // A way to temporarily opt out of opt in copy. This will *never* be accepted.
    (removed, opt_out_copy, "1.0.0", None, None, None),
    (removed, quad_precision_float, "1.0.0", None, None, None),
    (removed, struct_inherit, "1.0.0", None, None, None),
    (removed, test_removed_feature, "1.0.0", None, None, None),
    (removed, visible_private_types, "1.0.0", None, None, None),
    (removed, unsafe_no_drop_flag, "1.0.0", None, None, None),
    // Allows using items which are missing stability attributes
    // rustc internal
    (removed, unmarked_api, "1.0.0", None, None, None),
    (removed, pushpop_unsafe, "1.2.0", None, None, None),
    (removed, allocator, "1.0.0", None, None, None),
    (removed, simd, "1.0.0", Some(27731), None,
     Some("removed in favor of `#[repr(simd)]`")),
    (removed, advanced_slice_patterns, "1.0.0", Some(23121), None,
     Some("merged into `#![feature(slice_patterns)]`")),
    (removed, macro_reexport, "1.0.0", Some(29638), None,
     Some("subsumed by `pub use`")),
);

declare_features! (
    (stable_removed, no_stack_check, "1.0.0", None, None),
);

declare_features! (
    (accepted, associated_types, "1.0.0", None, None),
    // allow overloading augmented assignment operations like `a += b`
    (accepted, augmented_assignments, "1.8.0", Some(28235), None),
    // allow empty structs and enum variants with braces
    (accepted, braced_empty_structs, "1.8.0", Some(29720), None),
    // Allows indexing into constant arrays.
    (accepted, const_indexing, "1.26.0", Some(29947), None),
    (accepted, default_type_params, "1.0.0", None, None),
    (accepted, globs, "1.0.0", None, None),
    (accepted, if_let, "1.0.0", None, None),
    // A temporary feature gate used to enable parser extensions needed
    // to bootstrap fix for #5723.
    (accepted, issue_5723_bootstrap, "1.0.0", None, None),
    (accepted, macro_rules, "1.0.0", None, None),
    // Allows using #![no_std]
    (accepted, no_std, "1.6.0", None, None),
    (accepted, slicing_syntax, "1.0.0", None, None),
    (accepted, struct_variant, "1.0.0", None, None),
    // These are used to test this portion of the compiler, they don't actually
    // mean anything
    (accepted, test_accepted_feature, "1.0.0", None, None),
    (accepted, tuple_indexing, "1.0.0", None, None),
    // Allows macros to appear in the type position.
    (accepted, type_macros, "1.13.0", Some(27245), None),
    (accepted, while_let, "1.0.0", None, None),
    // Allows `#[deprecated]` attribute
    (accepted, deprecated, "1.9.0", Some(29935), None),
    // `expr?`
    (accepted, question_mark, "1.13.0", Some(31436), None),
    // Allows `..` in tuple (struct) patterns
    (accepted, dotdot_in_tuple_patterns, "1.14.0", Some(33627), None),
    (accepted, item_like_imports, "1.15.0", Some(35120), None),
    // Allows using `Self` and associated types in struct expressions and patterns.
    (accepted, more_struct_aliases, "1.16.0", Some(37544), None),
    // elide `'static` lifetimes in `static`s and `const`s
    (accepted, static_in_const, "1.17.0", Some(35897), None),
    // Allows field shorthands (`x` meaning `x: x`) in struct literal expressions.
    (accepted, field_init_shorthand, "1.17.0", Some(37340), None),
    // Allows the definition recursive static items.
    (accepted, static_recursion, "1.17.0", Some(29719), None),
    // pub(restricted) visibilities (RFC 1422)
    (accepted, pub_restricted, "1.18.0", Some(32409), None),
    // The #![windows_subsystem] attribute
    (accepted, windows_subsystem, "1.18.0", Some(37499), None),
    // Allows `break {expr}` with a value inside `loop`s.
    (accepted, loop_break_value, "1.19.0", Some(37339), None),
    // Permits numeric fields in struct expressions and patterns.
    (accepted, relaxed_adts, "1.19.0", Some(35626), None),
    // Coerces non capturing closures to function pointers
    (accepted, closure_to_fn_coercion, "1.19.0", Some(39817), None),
    // Allows attributes on struct literal fields.
    (accepted, struct_field_attributes, "1.20.0", Some(38814), None),
    // Allows the definition of associated constants in `trait` or `impl`
    // blocks.
    (accepted, associated_consts, "1.20.0", Some(29646), None),
    // Usage of the `compile_error!` macro
    (accepted, compile_error, "1.20.0", Some(40872), None),
    // See rust-lang/rfcs#1414. Allows code like `let x: &'static u32 = &42` to work.
    (accepted, rvalue_static_promotion, "1.21.0", Some(38865), None),
    // Allow Drop types in constants (RFC 1440)
    (accepted, drop_types_in_const, "1.22.0", Some(33156), None),
    // Allows the sysV64 ABI to be specified on all platforms
    // instead of just the platforms on which it is the C ABI
    (accepted, abi_sysv64, "1.24.0", Some(36167), None),
    // Allows `repr(align(16))` struct attribute (RFC 1358)
    (accepted, repr_align, "1.25.0", Some(33626), None),
    // allow '|' at beginning of match arms (RFC 1925)
    (accepted, match_beginning_vert, "1.25.0", Some(44101), None),
    // Nested groups in `use` (RFC 2128)
    (accepted, use_nested_groups, "1.25.0", Some(44494), None),
    // a..=b and ..=b
    (accepted, inclusive_range_syntax, "1.26.0", Some(28237), None),
    // allow `..=` in patterns (RFC 1192)
    (accepted, dotdoteq_in_patterns, "1.26.0", Some(28237), None),
    // Termination trait in main (RFC 1937)
    (accepted, termination_trait, "1.26.0", Some(43301), None),
    // Copy/Clone closures (RFC 2132)
    (accepted, clone_closures, "1.26.0", Some(44490), None),
    (accepted, copy_closures, "1.26.0", Some(44490), None),
    // Allows `impl Trait` in function arguments.
    (accepted, universal_impl_trait, "1.26.0", Some(34511), None),
    // Allows `impl Trait` in function return types.
    (accepted, conservative_impl_trait, "1.26.0", Some(34511), None),
    // The `i128` type
    (accepted, i128_type, "1.26.0", Some(35118), None),
    // Default match binding modes (RFC 2005)
    (accepted, match_default_bindings, "1.26.0", Some(42640), None),
    // allow `'_` placeholder lifetimes
    (accepted, underscore_lifetimes, "1.26.0", Some(44524), None),
    // Allows attributes on lifetime/type formal parameters in generics (RFC 1327)
    (accepted, generic_param_attrs, "1.27.0", Some(48848), None),
    // Allows cfg(target_feature = "...").
    (accepted, cfg_target_feature, "1.27.0", Some(29717), None),
    // Allows #[target_feature(...)]
    (accepted, target_feature, "1.27.0", None, None),
    // Trait object syntax with `dyn` prefix
    (accepted, dyn_trait, "1.27.0", Some(44662), None),
    // allow `#[must_use]` on functions; and, must-use operators (RFC 1940)
    (accepted, fn_must_use, "1.27.0", Some(43302), None),
    // Allows use of the :lifetime macro fragment specifier
    (accepted, macro_lifetime_matcher, "1.27.0", Some(34303), None),
    // Termination trait in tests (RFC 1937)
    (accepted, termination_trait_test, "1.27.0", Some(48854), None),
    // The #[global_allocator] attribute
    (accepted, global_allocator, "1.28.0", Some(27389), None),
    // Allows `#[repr(transparent)]` attribute on newtype structs
    (accepted, repr_transparent, "1.28.0", Some(43036), None),
    // Defining procedural macros in `proc-macro` crates
    (accepted, proc_macro, "1.29.0", Some(38356), None),
    // Allows use of the :vis macro fragment specifier
    (accepted, macro_vis_matcher, "1.29.0", Some(41022), None),
    // Allows importing and reexporting macros with `use`,
    // enables macro modularization in general.
    (accepted, use_extern_macros, "1.30.0", Some(35896), None),
    // Allows keywords to be escaped for use as identifiers
    (accepted, raw_identifiers, "1.30.0", Some(48589), None),
);

// If you change this, please modify src/doc/unstable-book as well. You must
// move that documentation into the relevant place in the other docs, and
// remove the chapter on the flag.

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AttributeType {
    /// Normal, builtin attribute that is consumed
    /// by the compiler before the unused_attribute check
    Normal,

    /// Builtin attribute that may not be consumed by the compiler
    /// before the unused_attribute check. These attributes
    /// will be ignored by the unused_attribute lint
    Whitelisted,

    /// Builtin attribute that is only allowed at the crate level
    CrateLevel,
}

pub enum AttributeGate {
    /// Is gated by a given feature gate, reason
    /// and function to check if enabled
    Gated(Stability, &'static str, &'static str, fn(&Features) -> bool),

    /// Ungated attribute, can be used on all release channels
    Ungated,
}

impl AttributeGate {
    fn is_deprecated(&self) -> bool {
        match *self {
            Gated(Stability::Deprecated(_), ..) => true,
            _ => false,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Stability {
    Unstable,
    // Argument is tracking issue link.
    Deprecated(&'static str),
}

// fn() is not Debug
impl ::std::fmt::Debug for AttributeGate {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            Gated(ref stab, name, expl, _) =>
                write!(fmt, "Gated({:?}, {}, {})", stab, name, expl),
            Ungated => write!(fmt, "Ungated")
        }
    }
}

macro_rules! cfg_fn {
    ($field: ident) => {{
        fn f(features: &Features) -> bool {
            features.$field
        }
        f as fn(&Features) -> bool
    }}
}

pub fn deprecated_attributes() -> Vec<&'static (&'static str, AttributeType, AttributeGate)> {
    BUILTIN_ATTRIBUTES.iter().filter(|a| a.2.is_deprecated()).collect()
}

pub fn is_builtin_attr_name(name: ast::Name) -> bool {
    BUILTIN_ATTRIBUTES.iter().any(|&(builtin_name, _, _)| name == builtin_name)
}

pub fn is_builtin_attr(attr: &ast::Attribute) -> bool {
    BUILTIN_ATTRIBUTES.iter().any(|&(builtin_name, _, _)| attr.path == builtin_name)
}

// Attributes that have a special meaning to rustc or rustdoc
pub const BUILTIN_ATTRIBUTES: &'static [(&'static str, AttributeType, AttributeGate)] = &[
    // Normal attributes

    ("warn", Normal, Ungated),
    ("allow", Normal, Ungated),
    ("forbid", Normal, Ungated),
    ("deny", Normal, Ungated),

    ("macro_use", Normal, Ungated),
    ("macro_export", Normal, Ungated),
    ("plugin_registrar", Normal, Ungated),

    ("cfg", Normal, Ungated),
    ("cfg_attr", Normal, Ungated),
    ("main", Normal, Ungated),
    ("start", Normal, Ungated),
    ("test", Normal, Ungated),
    ("bench", Normal, Ungated),
    ("repr", Normal, Ungated),
    ("path", Normal, Ungated),
    ("abi", Normal, Ungated),
    ("automatically_derived", Normal, Ungated),
    ("no_mangle", Normal, Ungated),
    ("no_link", Normal, Ungated),
    ("derive", Normal, Ungated),
    ("should_panic", Normal, Ungated),
    ("ignore", Normal, Ungated),
    ("no_implicit_prelude", Normal, Ungated),
    ("reexport_test_harness_main", Normal, Ungated),
    ("link_args", Normal, Gated(Stability::Unstable,
                                "link_args",
                                "the `link_args` attribute is experimental and not \
                                 portable across platforms, it is recommended to \
                                 use `#[link(name = \"foo\")] instead",
                                cfg_fn!(link_args))),
    ("macro_escape", Normal, Ungated),

    // RFC #1445.
    ("structural_match", Whitelisted, Gated(Stability::Unstable,
                                            "structural_match",
                                            "the semantics of constant patterns is \
                                             not yet settled",
                                            cfg_fn!(structural_match))),

    // RFC #2008
    ("non_exhaustive", Whitelisted, Gated(Stability::Unstable,
                                          "non_exhaustive",
                                          "non exhaustive is an experimental feature",
                                          cfg_fn!(non_exhaustive))),

    ("plugin", CrateLevel, Gated(Stability::Unstable,
                                 "plugin",
                                 "compiler plugins are experimental \
                                  and possibly buggy",
                                 cfg_fn!(plugin))),

    ("no_std", CrateLevel, Ungated),
    ("no_core", CrateLevel, Gated(Stability::Unstable,
                                  "no_core",
                                  "no_core is experimental",
                                  cfg_fn!(no_core))),
    ("lang", Normal, Gated(Stability::Unstable,
                           "lang_items",
                           "language items are subject to change",
                           cfg_fn!(lang_items))),
    ("linkage", Whitelisted, Gated(Stability::Unstable,
                                   "linkage",
                                   "the `linkage` attribute is experimental \
                                    and not portable across platforms",
                                   cfg_fn!(linkage))),
    ("thread_local", Whitelisted, Gated(Stability::Unstable,
                                        "thread_local",
                                        "`#[thread_local]` is an experimental feature, and does \
                                         not currently handle destructors.",
                                        cfg_fn!(thread_local))),

    ("rustc_on_unimplemented", Normal, Gated(Stability::Unstable,
                                             "on_unimplemented",
                                             "the `#[rustc_on_unimplemented]` attribute \
                                              is an experimental feature",
                                             cfg_fn!(on_unimplemented))),
    ("rustc_const_unstable", Normal, Gated(Stability::Unstable,
                                             "rustc_const_unstable",
                                             "the `#[rustc_const_unstable]` attribute \
                                              is an internal feature",
                                             cfg_fn!(rustc_const_unstable))),
    ("global_allocator", Normal, Ungated),
    ("default_lib_allocator", Whitelisted, Gated(Stability::Unstable,
                                            "allocator_internals",
                                            "the `#[default_lib_allocator]` \
                                             attribute is an experimental feature",
                                            cfg_fn!(allocator_internals))),
    ("needs_allocator", Normal, Gated(Stability::Unstable,
                                      "allocator_internals",
                                      "the `#[needs_allocator]` \
                                       attribute is an experimental \
                                       feature",
                                      cfg_fn!(allocator_internals))),
    ("panic_runtime", Whitelisted, Gated(Stability::Unstable,
                                         "panic_runtime",
                                         "the `#[panic_runtime]` attribute is \
                                          an experimental feature",
                                         cfg_fn!(panic_runtime))),
    ("needs_panic_runtime", Whitelisted, Gated(Stability::Unstable,
                                               "needs_panic_runtime",
                                               "the `#[needs_panic_runtime]` \
                                                attribute is an experimental \
                                                feature",
                                               cfg_fn!(needs_panic_runtime))),
    ("rustc_outlives", Normal, Gated(Stability::Unstable,
                                     "rustc_attrs",
                                     "the `#[rustc_outlives]` attribute \
                                      is just used for rustc unit tests \
                                      and will never be stable",
                                     cfg_fn!(rustc_attrs))),
    ("rustc_variance", Normal, Gated(Stability::Unstable,
                                     "rustc_attrs",
                                     "the `#[rustc_variance]` attribute \
                                      is just used for rustc unit tests \
                                      and will never be stable",
                                     cfg_fn!(rustc_attrs))),
    ("rustc_regions", Normal, Gated(Stability::Unstable,
                                    "rustc_attrs",
                                    "the `#[rustc_regions]` attribute \
                                     is just used for rustc unit tests \
                                     and will never be stable",
                                    cfg_fn!(rustc_attrs))),
    ("rustc_error", Whitelisted, Gated(Stability::Unstable,
                                       "rustc_attrs",
                                       "the `#[rustc_error]` attribute \
                                        is just used for rustc unit tests \
                                        and will never be stable",
                                       cfg_fn!(rustc_attrs))),
    ("rustc_if_this_changed", Whitelisted, Gated(Stability::Unstable,
                                                 "rustc_attrs",
                                                 "the `#[rustc_if_this_changed]` attribute \
                                                  is just used for rustc unit tests \
                                                  and will never be stable",
                                                 cfg_fn!(rustc_attrs))),
    ("rustc_then_this_would_need", Whitelisted, Gated(Stability::Unstable,
                                                      "rustc_attrs",
                                                      "the `#[rustc_if_this_changed]` attribute \
                                                       is just used for rustc unit tests \
                                                       and will never be stable",
                                                      cfg_fn!(rustc_attrs))),
    ("rustc_dirty", Whitelisted, Gated(Stability::Unstable,
                                       "rustc_attrs",
                                       "the `#[rustc_dirty]` attribute \
                                        is just used for rustc unit tests \
                                        and will never be stable",
                                       cfg_fn!(rustc_attrs))),
    ("rustc_clean", Whitelisted, Gated(Stability::Unstable,
                                       "rustc_attrs",
                                       "the `#[rustc_clean]` attribute \
                                        is just used for rustc unit tests \
                                        and will never be stable",
                                       cfg_fn!(rustc_attrs))),
    ("rustc_partition_reused", Whitelisted, Gated(Stability::Unstable,
                                                  "rustc_attrs",
                                                  "this attribute \
                                                   is just used for rustc unit tests \
                                                   and will never be stable",
                                                  cfg_fn!(rustc_attrs))),
    ("rustc_partition_codegened", Whitelisted, Gated(Stability::Unstable,
                                                      "rustc_attrs",
                                                      "this attribute \
                                                       is just used for rustc unit tests \
                                                       and will never be stable",
                                                      cfg_fn!(rustc_attrs))),
    ("rustc_synthetic", Whitelisted, Gated(Stability::Unstable,
                                                      "rustc_attrs",
                                                      "this attribute \
                                                       is just used for rustc unit tests \
                                                       and will never be stable",
                                                      cfg_fn!(rustc_attrs))),
    ("rustc_symbol_name", Whitelisted, Gated(Stability::Unstable,
                                             "rustc_attrs",
                                             "internal rustc attributes will never be stable",
                                             cfg_fn!(rustc_attrs))),
    ("rustc_item_path", Whitelisted, Gated(Stability::Unstable,
                                           "rustc_attrs",
                                           "internal rustc attributes will never be stable",
                                           cfg_fn!(rustc_attrs))),
    ("rustc_mir", Whitelisted, Gated(Stability::Unstable,
                                     "rustc_attrs",
                                     "the `#[rustc_mir]` attribute \
                                      is just used for rustc unit tests \
                                      and will never be stable",
                                     cfg_fn!(rustc_attrs))),
    ("rustc_inherit_overflow_checks", Whitelisted, Gated(Stability::Unstable,
                                                         "rustc_attrs",
                                                         "the `#[rustc_inherit_overflow_checks]` \
                                                          attribute is just used to control \
                                                          overflow checking behavior of several \
                                                          libcore functions that are inlined \
                                                          across crates and will never be stable",
                                                          cfg_fn!(rustc_attrs))),

    ("rustc_dump_program_clauses", Whitelisted, Gated(Stability::Unstable,
                                                     "rustc_attrs",
                                                     "the `#[rustc_dump_program_clauses]` \
                                                      attribute is just used for rustc unit \
                                                      tests and will never be stable",
                                                     cfg_fn!(rustc_attrs))),

    // RFC #2094
    ("nll", Whitelisted, Gated(Stability::Unstable,
                               "nll",
                               "Non lexical lifetimes",
                               cfg_fn!(nll))),
    ("compiler_builtins", Whitelisted, Gated(Stability::Unstable,
                                             "compiler_builtins",
                                             "the `#[compiler_builtins]` attribute is used to \
                                              identify the `compiler_builtins` crate which \
                                              contains compiler-rt intrinsics and will never be \
                                              stable",
                                          cfg_fn!(compiler_builtins))),
    ("sanitizer_runtime", Whitelisted, Gated(Stability::Unstable,
                                             "sanitizer_runtime",
                                             "the `#[sanitizer_runtime]` attribute is used to \
                                              identify crates that contain the runtime of a \
                                              sanitizer and will never be stable",
                                             cfg_fn!(sanitizer_runtime))),
    ("profiler_runtime", Whitelisted, Gated(Stability::Unstable,
                                             "profiler_runtime",
                                             "the `#[profiler_runtime]` attribute is used to \
                                              identify the `profiler_builtins` crate which \
                                              contains the profiler runtime and will never be \
                                              stable",
                                             cfg_fn!(profiler_runtime))),

    ("allow_internal_unstable", Normal, Gated(Stability::Unstable,
                                              "allow_internal_unstable",
                                              EXPLAIN_ALLOW_INTERNAL_UNSTABLE,
                                              cfg_fn!(allow_internal_unstable))),

    ("allow_internal_unsafe", Normal, Gated(Stability::Unstable,
                                            "allow_internal_unsafe",
                                            EXPLAIN_ALLOW_INTERNAL_UNSAFE,
                                            cfg_fn!(allow_internal_unsafe))),

    ("fundamental", Whitelisted, Gated(Stability::Unstable,
                                       "fundamental",
                                       "the `#[fundamental]` attribute \
                                        is an experimental feature",
                                       cfg_fn!(fundamental))),

    ("proc_macro_derive", Normal, Ungated),

    ("rustc_copy_clone_marker", Whitelisted, Gated(Stability::Unstable,
                                                   "rustc_attrs",
                                                   "internal implementation detail",
                                                   cfg_fn!(rustc_attrs))),

    // FIXME: #14408 whitelist docs since rustdoc looks at them
    ("doc", Whitelisted, Ungated),

    // FIXME: #14406 these are processed in codegen, which happens after the
    // lint pass
    ("cold", Whitelisted, Ungated),
    ("naked", Whitelisted, Gated(Stability::Unstable,
                                 "naked_functions",
                                 "the `#[naked]` attribute \
                                  is an experimental feature",
                                 cfg_fn!(naked_functions))),
    ("target_feature", Whitelisted, Ungated),
    ("export_name", Whitelisted, Ungated),
    ("inline", Whitelisted, Ungated),
    ("link", Whitelisted, Ungated),
    ("link_name", Whitelisted, Ungated),
    ("link_section", Whitelisted, Ungated),
    ("no_builtins", Whitelisted, Ungated),
    ("no_mangle", Whitelisted, Ungated),
    ("no_debug", Whitelisted, Gated(
        Stability::Deprecated("https://github.com/rust-lang/rust/issues/29721"),
        "no_debug",
        "the `#[no_debug]` attribute was an experimental feature that has been \
         deprecated due to lack of demand",
        cfg_fn!(no_debug))),
    ("omit_gdb_pretty_printer_section", Whitelisted, Gated(Stability::Unstable,
                                                       "omit_gdb_pretty_printer_section",
                                                       "the `#[omit_gdb_pretty_printer_section]` \
                                                        attribute is just used for the Rust test \
                                                        suite",
                                                       cfg_fn!(omit_gdb_pretty_printer_section))),
    ("unsafe_destructor_blind_to_params",
     Normal,
     Gated(Stability::Deprecated("https://github.com/rust-lang/rust/issues/34761"),
           "dropck_parametricity",
           "unsafe_destructor_blind_to_params has been replaced by \
            may_dangle and will be removed in the future",
           cfg_fn!(dropck_parametricity))),
    ("may_dangle",
     Normal,
     Gated(Stability::Unstable,
           "dropck_eyepatch",
           "may_dangle has unstable semantics and may be removed in the future",
           cfg_fn!(dropck_eyepatch))),
    ("unwind", Whitelisted, Gated(Stability::Unstable,
                                  "unwind_attributes",
                                  "#[unwind] is experimental",
                                  cfg_fn!(unwind_attributes))),
    ("used", Whitelisted, Gated(
        Stability::Unstable, "used",
        "the `#[used]` attribute is an experimental feature",
        cfg_fn!(used))),

    // used in resolve
    ("prelude_import", Whitelisted, Gated(Stability::Unstable,
                                          "prelude_import",
                                          "`#[prelude_import]` is for use by rustc only",
                                          cfg_fn!(prelude_import))),

    // FIXME: #14407 these are only looked at on-demand so we can't
    // guarantee they'll have already been checked
    ("rustc_deprecated", Whitelisted, Ungated),
    ("must_use", Whitelisted, Ungated),
    ("stable", Whitelisted, Ungated),
    ("unstable", Whitelisted, Ungated),
    ("deprecated", Normal, Ungated),

    ("rustc_paren_sugar", Normal, Gated(Stability::Unstable,
                                        "unboxed_closures",
                                        "unboxed_closures are still evolving",
                                        cfg_fn!(unboxed_closures))),

    ("windows_subsystem", Whitelisted, Ungated),

    ("proc_macro_attribute", Normal, Ungated),
    ("proc_macro", Normal, Ungated),

    ("rustc_derive_registrar", Normal, Gated(Stability::Unstable,
                                             "rustc_derive_registrar",
                                             "used internally by rustc",
                                             cfg_fn!(rustc_attrs))),

    ("allow_fail", Normal, Gated(Stability::Unstable,
                                 "allow_fail",
                                 "allow_fail attribute is currently unstable",
                                 cfg_fn!(allow_fail))),

    ("rustc_std_internal_symbol", Whitelisted, Gated(Stability::Unstable,
                                     "rustc_attrs",
                                     "this is an internal attribute that will \
                                      never be stable",
                                     cfg_fn!(rustc_attrs))),

    // whitelists "identity-like" conversion methods to suggest on type mismatch
    ("rustc_conversion_suggestion", Whitelisted, Gated(Stability::Unstable,
                                                       "rustc_attrs",
                                                       "this is an internal attribute that will \
                                                        never be stable",
                                                       cfg_fn!(rustc_attrs))),

    ("rustc_args_required_const", Whitelisted, Gated(Stability::Unstable,
                                 "rustc_attrs",
                                 "never will be stable",
                                 cfg_fn!(rustc_attrs))),

    // RFC #2093
    ("infer_outlives_requirements", Normal, Gated(Stability::Unstable,
                                   "infer_outlives_requirements",
                                   "infer outlives requirements is an experimental feature",
                                   cfg_fn!(infer_outlives_requirements))),

    // RFC #2093
    ("infer_static_outlives_requirements", Normal, Gated(Stability::Unstable,
                                   "infer_static_outlives_requirements",
                                   "infer 'static lifetime requirements",
                                   cfg_fn!(infer_static_outlives_requirements))),

    // RFC 2070
    ("panic_implementation", Normal, Gated(Stability::Unstable,
                           "panic_implementation",
                           "#[panic_implementation] is an unstable feature",
                           cfg_fn!(panic_implementation))),

    ("alloc_error_handler", Normal, Gated(Stability::Unstable,
                           "alloc_error_handler",
                           "#[alloc_error_handler] is an unstable feature",
                           cfg_fn!(alloc_error_handler))),

    // Crate level attributes
    ("crate_name", CrateLevel, Ungated),
    ("crate_type", CrateLevel, Ungated),
    ("crate_id", CrateLevel, Ungated),
    ("feature", CrateLevel, Ungated),
    ("no_start", CrateLevel, Ungated),
    ("no_main", CrateLevel, Ungated),
    ("no_builtins", CrateLevel, Ungated),
    ("recursion_limit", CrateLevel, Ungated),
    ("type_length_limit", CrateLevel, Ungated),
];

// cfg(...)'s that are feature gated
const GATED_CFGS: &[(&str, &str, fn(&Features) -> bool)] = &[
    // (name in cfg, feature, function to check if the feature is enabled)
    ("target_vendor", "cfg_target_vendor", cfg_fn!(cfg_target_vendor)),
    ("target_thread_local", "cfg_target_thread_local", cfg_fn!(cfg_target_thread_local)),
    ("target_has_atomic", "cfg_target_has_atomic", cfg_fn!(cfg_target_has_atomic)),
];

#[derive(Debug)]
pub struct GatedCfg {
    span: Span,
    index: usize,
}

impl GatedCfg {
    pub fn gate(cfg: &ast::MetaItem) -> Option<GatedCfg> {
        let name = cfg.name().as_str();
        GATED_CFGS.iter()
                  .position(|info| info.0 == name)
                  .map(|idx| {
                      GatedCfg {
                          span: cfg.span,
                          index: idx
                      }
                  })
    }

    pub fn check_and_emit(&self, sess: &ParseSess, features: &Features) {
        let (cfg, feature, has_feature) = GATED_CFGS[self.index];
        if !has_feature(features) && !self.span.allows_unstable() {
            let explain = format!("`cfg({})` is experimental and subject to change", cfg);
            emit_feature_err(sess, feature, self.span, GateIssue::Language, &explain);
        }
    }
}

struct Context<'a> {
    features: &'a Features,
    parse_sess: &'a ParseSess,
    plugin_attributes: &'a [(String, AttributeType)],
}

macro_rules! gate_feature_fn {
    ($cx: expr, $has_feature: expr, $span: expr, $name: expr, $explain: expr, $level: expr) => {{
        let (cx, has_feature, span,
             name, explain, level) = ($cx, $has_feature, $span, $name, $explain, $level);
        let has_feature: bool = has_feature(&$cx.features);
        debug!("gate_feature(feature = {:?}, span = {:?}); has? {}", name, span, has_feature);
        if !has_feature && !span.allows_unstable() {
            leveled_feature_err(cx.parse_sess, name, span, GateIssue::Language, explain, level)
                .emit();
        }
    }}
}

macro_rules! gate_feature {
    ($cx: expr, $feature: ident, $span: expr, $explain: expr) => {
        gate_feature_fn!($cx, |x:&Features| x.$feature, $span,
                         stringify!($feature), $explain, GateStrength::Hard)
    };
    ($cx: expr, $feature: ident, $span: expr, $explain: expr, $level: expr) => {
        gate_feature_fn!($cx, |x:&Features| x.$feature, $span,
                         stringify!($feature), $explain, $level)
    };
}

impl<'a> Context<'a> {
    fn check_attribute(&self, attr: &ast::Attribute, is_macro: bool) {
        debug!("check_attribute(attr = {:?})", attr);
        let name = attr.name().as_str();
        for &(n, ty, ref gateage) in BUILTIN_ATTRIBUTES {
            if name == n {
                if let Gated(_, name, desc, ref has_feature) = *gateage {
                    gate_feature_fn!(self, has_feature, attr.span, name, desc, GateStrength::Hard);
                } else if name == "doc" {
                    if let Some(content) = attr.meta_item_list() {
                        if content.iter().any(|c| c.check_name("include")) {
                            gate_feature!(self, external_doc, attr.span,
                                "#[doc(include = \"...\")] is experimental"
                            );
                        }
                    }
                }
                debug!("check_attribute: {:?} is builtin, {:?}, {:?}", attr.path, ty, gateage);
                return;
            }
        }
        for &(ref n, ref ty) in self.plugin_attributes {
            if attr.path == &**n {
                // Plugins can't gate attributes, so we don't check for it
                // unlike the code above; we only use this loop to
                // short-circuit to avoid the checks below
                debug!("check_attribute: {:?} is registered by a plugin, {:?}", attr.path, ty);
                return;
            }
        }
        if name.starts_with("rustc_") {
            gate_feature!(self, rustc_attrs, attr.span,
                          "unless otherwise specified, attributes \
                           with the prefix `rustc_` \
                           are reserved for internal compiler diagnostics");
        } else if name.starts_with("derive_") {
            gate_feature!(self, custom_derive, attr.span, EXPLAIN_DERIVE_UNDERSCORE);
        } else if !attr::is_known(attr) {
            // Only run the custom attribute lint during regular
            // feature gate checking. Macro gating runs
            // before the plugin attributes are registered
            // so we skip this then
            if !is_macro {
                let msg = format!("The attribute `{}` is currently unknown to the compiler and \
                                   may have meaning added to it in the future", attr.path);
                gate_feature!(self, custom_attribute, attr.span, &msg);
            }
        }
    }
}

pub fn check_attribute(attr: &ast::Attribute, parse_sess: &ParseSess, features: &Features) {
    let cx = Context { features: features, parse_sess: parse_sess, plugin_attributes: &[] };
    cx.check_attribute(attr, true);
}

fn find_lang_feature_issue(feature: &str) -> Option<u32> {
    if let Some(info) = ACTIVE_FEATURES.iter().find(|t| t.0 == feature) {
        let issue = info.2;
        // FIXME (#28244): enforce that active features have issue numbers
        // assert!(issue.is_some())
        issue
    } else {
        // search in Accepted, Removed, or Stable Removed features
        let found = ACCEPTED_FEATURES.iter().chain(REMOVED_FEATURES).chain(STABLE_REMOVED_FEATURES)
            .find(|t| t.0 == feature);
        match found {
            Some(&(_, _, issue, _)) => issue,
            None => panic!("Feature `{}` is not declared anywhere", feature),
        }
    }
}

pub enum GateIssue {
    Language,
    Library(Option<u32>)
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum GateStrength {
    /// A hard error. (Most feature gates should use this.)
    Hard,
    /// Only a warning. (Use this only as backwards-compatibility demands.)
    Soft,
}

pub fn emit_feature_err(sess: &ParseSess, feature: &str, span: Span, issue: GateIssue,
                        explain: &str) {
    feature_err(sess, feature, span, issue, explain).emit();
}

pub fn feature_err<'a>(sess: &'a ParseSess, feature: &str, span: Span, issue: GateIssue,
                       explain: &str) -> DiagnosticBuilder<'a> {
    leveled_feature_err(sess, feature, span, issue, explain, GateStrength::Hard)
}

fn leveled_feature_err<'a>(sess: &'a ParseSess, feature: &str, span: Span, issue: GateIssue,
                           explain: &str, level: GateStrength) -> DiagnosticBuilder<'a> {
    let diag = &sess.span_diagnostic;

    let issue = match issue {
        GateIssue::Language => find_lang_feature_issue(feature),
        GateIssue::Library(lib) => lib,
    };

    let explanation = match issue {
        None | Some(0) => explain.to_owned(),
        Some(n) => format!("{} (see issue #{})", explain, n)
    };

    let mut err = match level {
        GateStrength::Hard => {
            diag.struct_span_err_with_code(span, &explanation, stringify_error_code!(E0658))
        }
        GateStrength::Soft => diag.struct_span_warn(span, &explanation),
    };

    // #23973: do not suggest `#![feature(...)]` if we are in beta/stable
    if sess.unstable_features.is_nightly_build() {
        err.help(&format!("add #![feature({})] to the \
                           crate attributes to enable",
                          feature));
    }

    // If we're on stable and only emitting a "soft" warning, add a note to
    // clarify that the feature isn't "on" (rather than being on but
    // warning-worthy).
    if !sess.unstable_features.is_nightly_build() && level == GateStrength::Soft {
        err.help("a nightly build of the compiler is required to enable this feature");
    }

    err

}

const EXPLAIN_BOX_SYNTAX: &'static str =
    "box expression syntax is experimental; you can call `Box::new` instead.";

pub const EXPLAIN_STMT_ATTR_SYNTAX: &'static str =
    "attributes on expressions are experimental.";

pub const EXPLAIN_ASM: &'static str =
    "inline assembly is not stable enough for use and is subject to change";

pub const EXPLAIN_GLOBAL_ASM: &'static str =
    "`global_asm!` is not stable enough for use and is subject to change";

pub const EXPLAIN_LOG_SYNTAX: &'static str =
    "`log_syntax!` is not stable enough for use and is subject to change";

pub const EXPLAIN_CONCAT_IDENTS: &'static str =
    "`concat_idents` is not stable enough for use and is subject to change";

pub const EXPLAIN_FORMAT_ARGS_NL: &'static str =
    "`format_args_nl` is only for internal language use and is subject to change";

pub const EXPLAIN_TRACE_MACROS: &'static str =
    "`trace_macros` is not stable enough for use and is subject to change";
pub const EXPLAIN_ALLOW_INTERNAL_UNSTABLE: &'static str =
    "allow_internal_unstable side-steps feature gating and stability checks";
pub const EXPLAIN_ALLOW_INTERNAL_UNSAFE: &'static str =
    "allow_internal_unsafe side-steps the unsafe_code lint";

pub const EXPLAIN_CUSTOM_DERIVE: &'static str =
    "`#[derive]` for custom traits is deprecated and will be removed in the future.";

pub const EXPLAIN_DEPR_CUSTOM_DERIVE: &'static str =
    "`#[derive]` for custom traits is deprecated and will be removed in the future. \
    Prefer using procedural macro custom derive.";

pub const EXPLAIN_DERIVE_UNDERSCORE: &'static str =
    "attributes of the form `#[derive_*]` are reserved for the compiler";

pub const EXPLAIN_LITERAL_MATCHER: &'static str =
    ":literal fragment specifier is experimental and subject to change";

pub const EXPLAIN_UNSIZED_TUPLE_COERCION: &'static str =
    "unsized tuple coercion is not stable enough for use and is subject to change";

pub const EXPLAIN_MACRO_AT_MOST_ONCE_REP: &'static str =
    "using the `?` macro Kleene operator for \"at most one\" repetition is unstable";

struct PostExpansionVisitor<'a> {
    context: &'a Context<'a>,
}

macro_rules! gate_feature_post {
    ($cx: expr, $feature: ident, $span: expr, $explain: expr) => {{
        let (cx, span) = ($cx, $span);
        if !span.allows_unstable() {
            gate_feature!(cx.context, $feature, span, $explain)
        }
    }};
    ($cx: expr, $feature: ident, $span: expr, $explain: expr, $level: expr) => {{
        let (cx, span) = ($cx, $span);
        if !span.allows_unstable() {
            gate_feature!(cx.context, $feature, span, $explain, $level)
        }
    }}
}

impl<'a> PostExpansionVisitor<'a> {
    fn check_abi(&self, abi: Abi, span: Span) {
        match abi {
            Abi::RustIntrinsic => {
                gate_feature_post!(&self, intrinsics, span,
                                   "intrinsics are subject to change");
            },
            Abi::PlatformIntrinsic => {
                gate_feature_post!(&self, platform_intrinsics, span,
                                   "platform intrinsics are experimental and possibly buggy");
            },
            Abi::Vectorcall => {
                gate_feature_post!(&self, abi_vectorcall, span,
                                   "vectorcall is experimental and subject to change");
            },
            Abi::Thiscall => {
                gate_feature_post!(&self, abi_thiscall, span,
                                   "thiscall is experimental and subject to change");
            },
            Abi::RustCall => {
                gate_feature_post!(&self, unboxed_closures, span,
                                   "rust-call ABI is subject to change");
            },
            Abi::PtxKernel => {
                gate_feature_post!(&self, abi_ptx, span,
                                   "PTX ABIs are experimental and subject to change");
            },
            Abi::Unadjusted => {
                gate_feature_post!(&self, abi_unadjusted, span,
                                   "unadjusted ABI is an implementation detail and perma-unstable");
            },
            Abi::Msp430Interrupt => {
                gate_feature_post!(&self, abi_msp430_interrupt, span,
                                   "msp430-interrupt ABI is experimental and subject to change");
            },
            Abi::X86Interrupt => {
                gate_feature_post!(&self, abi_x86_interrupt, span,
                                   "x86-interrupt ABI is experimental and subject to change");
            },
            Abi::AmdGpuKernel => {
                gate_feature_post!(&self, abi_amdgpu_kernel, span,
                                   "amdgpu-kernel ABI is experimental and subject to change");
            },
            // Stable
            Abi::Cdecl |
            Abi::Stdcall |
            Abi::Fastcall |
            Abi::Aapcs |
            Abi::Win64 |
            Abi::SysV64 |
            Abi::Rust |
            Abi::C |
            Abi::System => {}
        }
    }
}

fn contains_novel_literal(item: &ast::MetaItem) -> bool {
    use ast::MetaItemKind::*;
    use ast::NestedMetaItemKind::*;

    match item.node {
        Word => false,
        NameValue(ref lit) => !lit.node.is_str(),
        List(ref list) => list.iter().any(|li| {
            match li.node {
                MetaItem(ref mi) => contains_novel_literal(mi),
                Literal(_) => true,
            }
        }),
    }
}

impl<'a> PostExpansionVisitor<'a> {
    fn whole_crate_feature_gates(&mut self, _krate: &ast::Crate) {
        for &(ident, span) in &*self.context.parse_sess.non_modrs_mods.borrow() {
            if !span.allows_unstable() {
                let cx = &self.context;
                let level = GateStrength::Hard;
                let has_feature = cx.features.non_modrs_mods;
                let name = "non_modrs_mods";
                debug!("gate_feature(feature = {:?}, span = {:?}); has? {}",
                        name, span, has_feature);

                if !has_feature && !span.allows_unstable() {
                    leveled_feature_err(
                        cx.parse_sess, name, span, GateIssue::Language,
                        "mod statements in non-mod.rs files are unstable", level
                    )
                    .help(&format!("on stable builds, rename this file to {}{}mod.rs",
                                   ident, path::MAIN_SEPARATOR))
                    .emit();
                }
            }
        }
    }
}

impl<'a> Visitor<'a> for PostExpansionVisitor<'a> {
    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        if !attr.span.allows_unstable() {
            // check for gated attributes
            self.context.check_attribute(attr, false);
        }

        if attr.check_name("doc") {
            if let Some(content) = attr.meta_item_list() {
                if content.len() == 1 && content[0].check_name("cfg") {
                    gate_feature_post!(&self, doc_cfg, attr.span,
                        "#[doc(cfg(...))] is experimental"
                    );
                } else if content.iter().any(|c| c.check_name("masked")) {
                    gate_feature_post!(&self, doc_masked, attr.span,
                        "#[doc(masked)] is experimental"
                    );
                } else if content.iter().any(|c| c.check_name("spotlight")) {
                    gate_feature_post!(&self, doc_spotlight, attr.span,
                        "#[doc(spotlight)] is experimental"
                    );
                } else if content.iter().any(|c| c.check_name("alias")) {
                    gate_feature_post!(&self, doc_alias, attr.span,
                        "#[doc(alias = \"...\")] is experimental"
                    );
                } else if content.iter().any(|c| c.check_name("keyword")) {
                    gate_feature_post!(&self, doc_keyword, attr.span,
                        "#[doc(keyword = \"...\")] is experimental"
                    );
                }
            }
        }

        if !self.context.features.unrestricted_attribute_tokens {
            // Unfortunately, `parse_meta` cannot be called speculatively because it can report
            // errors by itself, so we have to call it only if the feature is disabled.
            match attr.parse_meta(self.context.parse_sess) {
                Ok(meta) => {
                    // allow attr_literals in #[repr(align(x))] and #[repr(packed(n))]
                    let mut allow_attr_literal = false;
                    if attr.path == "repr" {
                        if let Some(content) = meta.meta_item_list() {
                            allow_attr_literal = content.iter().any(
                                |c| c.check_name("align") || c.check_name("packed"));
                        }
                    }

                    if !allow_attr_literal && contains_novel_literal(&meta) {
                        gate_feature_post!(&self, attr_literals, attr.span,
                                        "non-string literals in attributes, or string \
                                        literals in top-level positions, are experimental");
                    }
                }
                Err(mut err) => {
                    err.help("try enabling `#![feature(unrestricted_attribute_tokens)]`").emit()
                }
            }
        }
    }

    fn visit_name(&mut self, sp: Span, name: ast::Name) {
        if !name.as_str().is_ascii() {
            gate_feature_post!(&self,
                               non_ascii_idents,
                               self.context.parse_sess.source_map().def_span(sp),
                               "non-ascii idents are not fully supported.");
        }
    }

    fn visit_use_tree(&mut self, use_tree: &'a ast::UseTree, id: NodeId, _nested: bool) {
        if let ast::UseTreeKind::Simple(Some(ident), ..) = use_tree.kind {
            if ident.name == "_" {
                gate_feature_post!(&self, underscore_imports, use_tree.span,
                                   "renaming imports with `_` is unstable");
            }
        }

        visit::walk_use_tree(self, use_tree, id);
    }

    fn visit_item(&mut self, i: &'a ast::Item) {
        match i.node {
            ast::ItemKind::ExternCrate(_) => {
                if i.ident.name == "_" {
                    gate_feature_post!(&self, underscore_imports, i.span,
                                       "renaming extern crates with `_` is unstable");
                }
            }

            ast::ItemKind::ForeignMod(ref foreign_module) => {
                self.check_abi(foreign_module.abi, i.span);
            }

            ast::ItemKind::Fn(..) => {
                if attr::contains_name(&i.attrs[..], "plugin_registrar") {
                    gate_feature_post!(&self, plugin_registrar, i.span,
                                       "compiler plugins are experimental and possibly buggy");
                }
                if attr::contains_name(&i.attrs[..], "start") {
                    gate_feature_post!(&self, start, i.span,
                                      "a #[start] function is an experimental \
                                       feature whose signature may change \
                                       over time");
                }
                if attr::contains_name(&i.attrs[..], "main") {
                    gate_feature_post!(&self, main, i.span,
                                       "declaration of a nonstandard #[main] \
                                        function may change over time, for now \
                                        a top-level `fn main()` is required");
                }
            }

            ast::ItemKind::Struct(..) => {
                if let Some(attr) = attr::find_by_name(&i.attrs[..], "repr") {
                    for item in attr.meta_item_list().unwrap_or_else(Vec::new) {
                        if item.check_name("simd") {
                            gate_feature_post!(&self, repr_simd, attr.span,
                                               "SIMD types are experimental and possibly buggy");
                        }
                        if let Some((name, _)) = item.name_value_literal() {
                            if name == "packed" {
                                gate_feature_post!(&self, repr_packed, attr.span,
                                                   "the `#[repr(packed(n))]` attribute \
                                                   is experimental");
                            }
                        }
                    }
                }
            }

            ast::ItemKind::TraitAlias(..) => {
                gate_feature_post!(&self, trait_alias,
                                   i.span,
                                   "trait aliases are not yet fully implemented");
            }

            ast::ItemKind::Impl(_, polarity, defaultness, _, _, _, _) => {
                if polarity == ast::ImplPolarity::Negative {
                    gate_feature_post!(&self, optin_builtin_traits,
                                       i.span,
                                       "negative trait bounds are not yet fully implemented; \
                                        use marker types for now");
                }

                if let ast::Defaultness::Default = defaultness {
                    gate_feature_post!(&self, specialization,
                                       i.span,
                                       "specialization is unstable");
                }
            }

            ast::ItemKind::Trait(ast::IsAuto::Yes, ..) => {
                gate_feature_post!(&self, optin_builtin_traits,
                                   i.span,
                                   "auto traits are experimental and possibly buggy");
            }

            ast::ItemKind::MacroDef(ast::MacroDef { legacy: false, .. }) => {
                let msg = "`macro` is experimental";
                gate_feature_post!(&self, decl_macro, i.span, msg);
            }

            ast::ItemKind::Existential(..) => {
                gate_feature_post!(
                    &self,
                    existential_type,
                    i.span,
                    "existential types are unstable"
                );
            }

            _ => {}
        }

        visit::walk_item(self, i);
    }

    fn visit_foreign_item(&mut self, i: &'a ast::ForeignItem) {
        match i.node {
            ast::ForeignItemKind::Fn(..) |
            ast::ForeignItemKind::Static(..) => {
                let link_name = attr::first_attr_value_str_by_name(&i.attrs, "link_name");
                let links_to_llvm = match link_name {
                    Some(val) => val.as_str().starts_with("llvm."),
                    _ => false
                };
                if links_to_llvm {
                    gate_feature_post!(&self, link_llvm_intrinsics, i.span,
                                       "linking to LLVM intrinsics is experimental");
                }
            }
            ast::ForeignItemKind::Ty => {
                    gate_feature_post!(&self, extern_types, i.span,
                                       "extern types are experimental");
            }
            ast::ForeignItemKind::Macro(..) => {}
        }

        visit::walk_foreign_item(self, i)
    }

    fn visit_ty(&mut self, ty: &'a ast::Ty) {
        match ty.node {
            ast::TyKind::BareFn(ref bare_fn_ty) => {
                self.check_abi(bare_fn_ty.abi, ty.span);
            }
            ast::TyKind::Never => {
                gate_feature_post!(&self, never_type, ty.span,
                                   "The `!` type is experimental");
            }
            _ => {}
        }
        visit::walk_ty(self, ty)
    }

    fn visit_fn_ret_ty(&mut self, ret_ty: &'a ast::FunctionRetTy) {
        if let ast::FunctionRetTy::Ty(ref output_ty) = *ret_ty {
            if let ast::TyKind::Never = output_ty.node {
                // Do nothing
            } else {
                self.visit_ty(output_ty)
            }
        }
    }

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        match e.node {
            ast::ExprKind::Box(_) => {
                gate_feature_post!(&self, box_syntax, e.span, EXPLAIN_BOX_SYNTAX);
            }
            ast::ExprKind::Type(..) => {
                gate_feature_post!(&self, type_ascription, e.span,
                                  "type ascription is experimental");
            }
            ast::ExprKind::ObsoleteInPlace(..) => {
                // these get a hard error in ast-validation
            }
            ast::ExprKind::Yield(..) => {
                gate_feature_post!(&self, generators,
                                  e.span,
                                  "yield syntax is experimental");
            }
            ast::ExprKind::Catch(_) => {
                gate_feature_post!(&self, catch_expr, e.span, "`catch` expression is experimental");
            }
            ast::ExprKind::IfLet(ref pats, ..) | ast::ExprKind::WhileLet(ref pats, ..) => {
                if pats.len() > 1 {
                    gate_feature_post!(&self, if_while_or_patterns, e.span,
                                    "multiple patterns in `if let` and `while let` are unstable");
                }
            }
            ast::ExprKind::Block(_, opt_label) => {
                if let Some(label) = opt_label {
                    gate_feature_post!(&self, label_break_value, label.ident.span,
                                    "labels on blocks are unstable");
                }
            }
            ast::ExprKind::Closure(_, ast::IsAsync::Async { .. }, ..) => {
                gate_feature_post!(&self, async_await, e.span, "async closures are unstable");
            }
            ast::ExprKind::Async(..) => {
                gate_feature_post!(&self, async_await, e.span, "async blocks are unstable");
            }
            _ => {}
        }
        visit::walk_expr(self, e);
    }

    fn visit_arm(&mut self, arm: &'a ast::Arm) {
        visit::walk_arm(self, arm)
    }

    fn visit_pat(&mut self, pattern: &'a ast::Pat) {
        match pattern.node {
            PatKind::Slice(_, Some(ref subslice), _) => {
                gate_feature_post!(&self, slice_patterns,
                                   subslice.span,
                                   "syntax for subslices in slice patterns is not yet stabilized");
            }
            PatKind::Box(..) => {
                gate_feature_post!(&self, box_patterns,
                                  pattern.span,
                                  "box pattern syntax is experimental");
            }
            PatKind::Range(_, _, Spanned { node: RangeEnd::Excluded, .. }) => {
                gate_feature_post!(&self, exclusive_range_pattern, pattern.span,
                                   "exclusive range pattern syntax is experimental");
            }
            PatKind::Paren(..) => {
                gate_feature_post!(&self, pattern_parentheses, pattern.span,
                                   "parentheses in patterns are unstable");
            }
            _ => {}
        }
        visit::walk_pat(self, pattern)
    }

    fn visit_fn(&mut self,
                fn_kind: FnKind<'a>,
                fn_decl: &'a ast::FnDecl,
                span: Span,
                _node_id: NodeId) {
        match fn_kind {
            FnKind::ItemFn(_, header, _, _) => {
                // check for const fn and async fn declarations
                if header.asyncness.is_async() {
                    gate_feature_post!(&self, async_await, span, "async fn is unstable");
                }
                if header.constness.node == ast::Constness::Const {
                    gate_feature_post!(&self, const_fn, span, "const fn is unstable");
                }
                // stability of const fn methods are covered in
                // visit_trait_item and visit_impl_item below; this is
                // because default methods don't pass through this
                // point.

                self.check_abi(header.abi, span);
            }
            FnKind::Method(_, sig, _, _) => {
                self.check_abi(sig.header.abi, span);
            }
            _ => {}
        }
        visit::walk_fn(self, fn_kind, fn_decl, span);
    }

    fn visit_trait_item(&mut self, ti: &'a ast::TraitItem) {
        match ti.node {
            ast::TraitItemKind::Method(ref sig, ref block) => {
                if block.is_none() {
                    self.check_abi(sig.header.abi, ti.span);
                }
                if sig.header.constness.node == ast::Constness::Const {
                    gate_feature_post!(&self, const_fn, ti.span, "const fn is unstable");
                }
            }
            ast::TraitItemKind::Type(_, ref default) => {
                // We use three if statements instead of something like match guards so that all
                // of these errors can be emitted if all cases apply.
                if default.is_some() {
                    gate_feature_post!(&self, associated_type_defaults, ti.span,
                                       "associated type defaults are unstable");
                }
                if !ti.generics.params.is_empty() {
                    gate_feature_post!(&self, generic_associated_types, ti.span,
                                       "generic associated types are unstable");
                }
                if !ti.generics.where_clause.predicates.is_empty() {
                    gate_feature_post!(&self, generic_associated_types, ti.span,
                                       "where clauses on associated types are unstable");
                }
            }
            _ => {}
        }
        visit::walk_trait_item(self, ti);
    }

    fn visit_impl_item(&mut self, ii: &'a ast::ImplItem) {
        if ii.defaultness == ast::Defaultness::Default {
            gate_feature_post!(&self, specialization,
                              ii.span,
                              "specialization is unstable");
        }

        match ii.node {
            ast::ImplItemKind::Method(ref sig, _) => {
                if sig.header.constness.node == ast::Constness::Const {
                    gate_feature_post!(&self, const_fn, ii.span, "const fn is unstable");
                }
            }
            ast::ImplItemKind::Existential(..) => {
                gate_feature_post!(
                    &self,
                    existential_type,
                    ii.span,
                    "existential types are unstable"
                );
            }

            ast::ImplItemKind::Type(_) if !ii.generics.params.is_empty() => {
                gate_feature_post!(&self, generic_associated_types, ii.span,
                                   "generic associated types are unstable");
            }
            _ => {}
        }
        visit::walk_impl_item(self, ii);
    }

    fn visit_path(&mut self, path: &'a ast::Path, _id: NodeId) {
        for segment in &path.segments {
            // Identifiers we are going to check could come from a legacy macro (e.g. `#[test]`).
            // For such macros identifiers must have empty context, because this context is
            // used during name resolution and produced names must be unhygienic for compatibility.
            // On the other hand, we need the actual non-empty context for feature gate checking
            // because it's hygienic even for legacy macros. As previously stated, such context
            // cannot be kept in identifiers, so it's kept in paths instead and we take it from
            // there while keeping location info from the ident span.
            let span = segment.ident.span.with_ctxt(path.span.ctxt());
            if segment.ident.name == keywords::Crate.name() {
                gate_feature_post!(&self, crate_in_paths, span,
                                   "`crate` in paths is experimental");
            } else if segment.ident.name == keywords::Extern.name() {
                gate_feature_post!(&self, extern_in_paths, span,
                                   "`extern` in paths is experimental");
            }
        }

        visit::walk_path(self, path);
    }

    fn visit_vis(&mut self, vis: &'a ast::Visibility) {
        if let ast::VisibilityKind::Crate(ast::CrateSugar::JustCrate) = vis.node {
            gate_feature_post!(&self, crate_visibility_modifier, vis.span,
                               "`crate` visibility modifier is experimental");
        }
        visit::walk_vis(self, vis);
    }
}

pub fn get_features(span_handler: &Handler, krate_attrs: &[ast::Attribute],
                    crate_edition: Edition) -> Features {
    fn feature_removed(span_handler: &Handler, span: Span, reason: Option<&str>) {
        let mut err = struct_span_err!(span_handler, span, E0557, "feature has been removed");
        if let Some(reason) = reason {
            err.span_note(span, reason);
        }
        err.emit();
    }

    // Some features are known to be incomplete and using them is likely to have
    // unanticipated results, such as compiler crashes. We warn the user about these
    // to alert them.
    let incomplete_features = ["generic_associated_types"];

    let mut features = Features::new();
    let mut edition_enabled_features = FxHashMap();

    for &edition in ALL_EDITIONS {
        if edition <= crate_edition {
            // The `crate_edition` implies its respective umbrella feature-gate
            // (i.e. `#![feature(rust_20XX_preview)]` isn't needed on edition 20XX).
            edition_enabled_features.insert(Symbol::intern(edition.feature_name()), edition);
        }
    }

    for &(name, .., f_edition, set) in ACTIVE_FEATURES {
        if let Some(f_edition) = f_edition {
            if f_edition <= crate_edition {
                set(&mut features, DUMMY_SP);
                edition_enabled_features.insert(Symbol::intern(name), crate_edition);
            }
        }
    }

    // Process the edition umbrella feature-gates first, to ensure
    // `edition_enabled_features` is completed before it's queried.
    for attr in krate_attrs {
        if !attr.check_name("feature") {
            continue
        }

        let list = match attr.meta_item_list() {
            Some(list) => list,
            None => continue,
        };

        for mi in list {
            let name = if let Some(word) = mi.word() {
                word.name()
            } else {
                continue
            };

            if incomplete_features.iter().any(|f| *f == name.as_str()) {
                span_handler.struct_span_warn(
                    mi.span,
                    &format!(
                        "the feature `{}` is incomplete and may cause the compiler to crash",
                        name
                    )
                ).emit();
            }

            if let Some(edition) = ALL_EDITIONS.iter().find(|e| name == e.feature_name()) {
                if *edition <= crate_edition {
                    continue;
                }

                for &(name, .., f_edition, set) in ACTIVE_FEATURES {
                    if let Some(f_edition) = f_edition {
                        if f_edition <= *edition {
                            // FIXME(Manishearth) there is currently no way to set
                            // lib features by edition
                            set(&mut features, DUMMY_SP);
                            edition_enabled_features.insert(Symbol::intern(name), *edition);
                        }
                    }
                }
            }
        }
    }

    for attr in krate_attrs {
        if !attr.check_name("feature") {
            continue
        }

        let list = match attr.meta_item_list() {
            Some(list) => list,
            None => {
                span_err!(span_handler, attr.span, E0555,
                          "malformed feature attribute, expected #![feature(...)]");
                continue
            }
        };

        for mi in list {
            let name = if let Some(word) = mi.word() {
                word.name()
            } else {
                span_err!(span_handler, mi.span, E0556,
                          "malformed feature, expected just one word");
                continue
            };

            if let Some(edition) = edition_enabled_features.get(&name) {
                struct_span_warn!(
                    span_handler,
                    mi.span,
                    E0705,
                    "the feature `{}` is included in the Rust {} edition",
                    name,
                    edition,
                ).emit();
                continue;
            }

            if ALL_EDITIONS.iter().any(|e| name == e.feature_name()) {
                // Handled in the separate loop above.
                continue;
            }

            if let Some((.., set)) = ACTIVE_FEATURES.iter().find(|f| name == f.0) {
                set(&mut features, mi.span);
                features.declared_lang_features.push((name, mi.span, None));
                continue
            }

            let removed = REMOVED_FEATURES.iter().find(|f| name == f.0);
            let stable_removed = STABLE_REMOVED_FEATURES.iter().find(|f| name == f.0);
            if let Some((.., reason)) = removed.or(stable_removed) {
                feature_removed(span_handler, mi.span, *reason);
                continue
            }

            if let Some((_, since, ..)) = ACCEPTED_FEATURES.iter().find(|f| name == f.0) {
                let since = Some(Symbol::intern(since));
                features.declared_lang_features.push((name, mi.span, since));
                continue
            }

            features.declared_lib_features.push((name, mi.span));
        }
    }

    features
}

pub fn check_crate(krate: &ast::Crate,
                   sess: &ParseSess,
                   features: &Features,
                   plugin_attributes: &[(String, AttributeType)],
                   unstable: UnstableFeatures) {
    maybe_stage_features(&sess.span_diagnostic, krate, unstable);
    let ctx = Context {
        features,
        parse_sess: sess,
        plugin_attributes,
    };

    let visitor = &mut PostExpansionVisitor { context: &ctx };
    visitor.whole_crate_feature_gates(krate);
    visit::walk_crate(visitor, krate);
}

#[derive(Clone, Copy, Hash)]
pub enum UnstableFeatures {
    /// Hard errors for unstable features are active, as on
    /// beta/stable channels.
    Disallow,
    /// Allow features to be activated, as on nightly.
    Allow,
    /// Errors are bypassed for bootstrapping. This is required any time
    /// during the build that feature-related lints are set to warn or above
    /// because the build turns on warnings-as-errors and uses lots of unstable
    /// features. As a result, this is always required for building Rust itself.
    Cheat
}

impl UnstableFeatures {
    pub fn from_environment() -> UnstableFeatures {
        // Whether this is a feature-staged build, i.e. on the beta or stable channel
        let disable_unstable_features = option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_some();
        // Whether we should enable unstable features for bootstrapping
        let bootstrap = env::var("RUSTC_BOOTSTRAP").is_ok();
        match (disable_unstable_features, bootstrap) {
            (_, true) => UnstableFeatures::Cheat,
            (true, _) => UnstableFeatures::Disallow,
            (false, _) => UnstableFeatures::Allow
        }
    }

    pub fn is_nightly_build(&self) -> bool {
        match *self {
            UnstableFeatures::Allow | UnstableFeatures::Cheat => true,
            _ => false,
        }
    }
}

fn maybe_stage_features(span_handler: &Handler, krate: &ast::Crate,
                        unstable: UnstableFeatures) {
    let allow_features = match unstable {
        UnstableFeatures::Allow => true,
        UnstableFeatures::Disallow => false,
        UnstableFeatures::Cheat => true
    };
    if !allow_features {
        for attr in &krate.attrs {
            if attr.check_name("feature") {
                let release_channel = option_env!("CFG_RELEASE_CHANNEL").unwrap_or("(unknown)");
                span_err!(span_handler, attr.span, E0554,
                          "#![feature] may not be used on the {} release channel",
                          release_channel);
            }
        }
    }
}
