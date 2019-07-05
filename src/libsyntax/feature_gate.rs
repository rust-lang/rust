//! # Feature gating
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

use AttributeType::*;
use AttributeGate::*;

use crate::ast::{
    self, AssocTyConstraint, AssocTyConstraintKind, NodeId, GenericParam, GenericParamKind,
    PatKind, RangeEnd,
};
use crate::attr;
use crate::early_buffered_lints::BufferedEarlyLintId;
use crate::source_map::Spanned;
use crate::edition::{ALL_EDITIONS, Edition};
use crate::visit::{self, FnKind, Visitor};
use crate::parse::{token, ParseSess};
use crate::parse::parser::Parser;
use crate::symbol::{Symbol, sym};
use crate::tokenstream::TokenTree;

use errors::{Applicability, DiagnosticBuilder, Handler};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lock;
use rustc_target::spec::abi::Abi;
use syntax_pos::{Span, DUMMY_SP, MultiSpan};
use log::debug;
use lazy_static::lazy_static;

use std::env;

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
            &[(Symbol, &str, Option<u32>, Option<Edition>, fn(&mut Features, Span))] =
            &[$((sym::$feature, $ver, $issue, $edition, set!($feature))),+];

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
        const REMOVED_FEATURES: &[(Symbol, &str, Option<u32>, Option<&str>)] = &[
            $((sym::$feature, $ver, $issue, $reason)),+
        ];
    };

    ($((stable_removed, $feature: ident, $ver: expr, $issue: expr, None),)+) => {
        /// Represents stable features which have since been removed (it was once Accepted)
        const STABLE_REMOVED_FEATURES: &[(Symbol, &str, Option<u32>, Option<&str>)] = &[
            $((sym::$feature, $ver, $issue, None)),+
        ];
    };

    ($((accepted, $feature: ident, $ver: expr, $issue: expr, None),)+) => {
        /// Those language feature has since been Accepted (it was once Active)
        const ACCEPTED_FEATURES: &[(Symbol, &str, Option<u32>, Option<&str>)] = &[
            $((sym::$feature, $ver, $issue, None)),+
        ];
    }
}

// If you change this, please modify `src/doc/unstable-book` as well.
//
// Don't ever remove anything from this list; set them to 'Removed'.
//
// The version numbers here correspond to the version in which the current status
// was set. This is most important for knowing when a particular feature became
// stable (active).
//
// Note that the features are grouped into internal/user-facing and then
// sorted by version inside those groups. This is inforced with tidy.
//
// N.B., `tools/tidy/src/features.rs` parses this information directly out of the
// source, so take care when modifying it.

declare_features! (
    // -------------------------------------------------------------------------
    // feature-group-start: internal feature gates
    // -------------------------------------------------------------------------

    // no-tracking-issue-start

    // Allows using the `rust-intrinsic`'s "ABI".
    (active, intrinsics, "1.0.0", None, None),

    // Allows using `#[lang = ".."]` attribute for linking items to special compiler logic.
    (active, lang_items, "1.0.0", None, None),

    // Allows using the `#[stable]` and `#[unstable]` attributes.
    (active, staged_api, "1.0.0", None, None),

    // Allows using `#[allow_internal_unstable]`. This is an
    // attribute on `macro_rules!` and can't use the attribute handling
    // below (it has to be checked before expansion possibly makes
    // macros disappear).
    (active, allow_internal_unstable, "1.0.0", None, None),

    // Allows using `#[allow_internal_unsafe]`. This is an
    // attribute on `macro_rules!` and can't use the attribute handling
    // below (it has to be checked before expansion possibly makes
    // macros disappear).
    (active, allow_internal_unsafe, "1.0.0", None, None),

    // Allows using the macros:
    // + `__diagnostic_used`
    // + `__register_diagnostic`
    // +`__build_diagnostic_array`
    (active, rustc_diagnostic_macros, "1.0.0", None, None),

    // Allows using `#[rustc_const_unstable(feature = "foo", ..)]` which
    // lets a function to be `const` when opted into with `#![feature(foo)]`.
    (active, rustc_const_unstable, "1.0.0", None, None),

    // no-tracking-issue-end

    // Allows using `#[link_name="llvm.*"]`.
    (active, link_llvm_intrinsics, "1.0.0", Some(29602), None),

    // Allows using `rustc_*` attributes (RFC 572).
    (active, rustc_attrs, "1.0.0", Some(29642), None),

    // Allows using `#[on_unimplemented(..)]` on traits.
    (active, on_unimplemented, "1.0.0", Some(29628), None),

    // Allows using the `box $expr` syntax.
    (active, box_syntax, "1.0.0", Some(49733), None),

    // Allows using `#[main]` to replace the entrypoint `#[lang = "start"]` calls.
    (active, main, "1.0.0", Some(29634), None),

    // Allows using `#[start]` on a function indicating that it is the program entrypoint.
    (active, start, "1.0.0", Some(29633), None),

    // Allows using the `#[fundamental]` attribute.
    (active, fundamental, "1.0.0", Some(29635), None),

    // Allows using the `rust-call` ABI.
    (active, unboxed_closures, "1.0.0", Some(29625), None),

    // Allows using the `#[linkage = ".."]` attribute.
    (active, linkage, "1.0.0", Some(29603), None),

    // Allows features specific to OIBIT (auto traits).
    (active, optin_builtin_traits, "1.0.0", Some(13231), None),

    // Allows using `box` in patterns (RFC 469).
    (active, box_patterns, "1.0.0", Some(29641), None),

    // no-tracking-issue-start

    // Allows using `#[prelude_import]` on glob `use` items.
    (active, prelude_import, "1.2.0", None, None),

    // no-tracking-issue-end

    // Allows using `#[unsafe_destructor_blind_to_params]` (RFC 1238).
    (active, dropck_parametricity, "1.3.0", Some(28498), None),

    // no-tracking-issue-start

    // Allows using `#[omit_gdb_pretty_printer_section]`.
    (active, omit_gdb_pretty_printer_section, "1.5.0", None, None),

    // Allows using the `vectorcall` ABI.
    (active, abi_vectorcall, "1.7.0", None, None),

    // no-tracking-issue-end

    // Allows using `#[structural_match]` which indicates that a type is structurally matchable.
    (active, structural_match, "1.8.0", Some(31434), None),

    // Allows using the `may_dangle` attribute (RFC 1327).
    (active, dropck_eyepatch, "1.10.0", Some(34761), None),

    // Allows using the `#![panic_runtime]` attribute.
    (active, panic_runtime, "1.10.0", Some(32837), None),

    // Allows declaring with `#![needs_panic_runtime]` that a panic runtime is needed.
    (active, needs_panic_runtime, "1.10.0", Some(32837), None),

    // no-tracking-issue-start

    // Allows identifying the `compiler_builtins` crate.
    (active, compiler_builtins, "1.13.0", None, None),

    // Allows using the `unadjusted` ABI; perma-unstable.
    (active, abi_unadjusted, "1.16.0", None, None),

    // Allows identifying crates that contain sanitizer runtimes.
    (active, sanitizer_runtime, "1.17.0", None, None),

    // Used to identify crates that contain the profiler runtime.
    (active, profiler_runtime, "1.18.0", None, None),

    // Allows using the `thiscall` ABI.
    (active, abi_thiscall, "1.19.0", None, None),

    // Allows using `#![needs_allocator]`, an implementation detail of `#[global_allocator]`.
    (active, allocator_internals, "1.20.0", None, None),

    // Allows using the `format_args_nl` macro.
    (active, format_args_nl, "1.29.0", None, None),

    // no-tracking-issue-end

    // Added for testing E0705; perma-unstable.
    (active, test_2018_feature, "1.31.0", Some(0), Some(Edition::Edition2018)),

    // -------------------------------------------------------------------------
    // feature-group-end: internal feature gates
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // feature-group-start: actual feature gates (target features)
    // -------------------------------------------------------------------------

    // FIXME: Document these and merge with the list below.

    // Unstable `#[target_feature]` directives.
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
    (active, adx_target_feature, "1.32.0", Some(44839), None),
    (active, cmpxchg16b_target_feature, "1.32.0", Some(44839), None),
    (active, movbe_target_feature, "1.34.0", Some(44839), None),
    (active, rtm_target_feature, "1.35.0", Some(44839), None),
    (active, f16c_target_feature, "1.36.0", Some(44839), None),

    // -------------------------------------------------------------------------
    // feature-group-end: actual feature gates (target features)
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // feature-group-start: actual feature gates
    // -------------------------------------------------------------------------

    // Allows using `asm!` macro with which inline assembly can be embedded.
    (active, asm, "1.0.0", Some(29722), None),

    // Allows using the `concat_idents!` macro with which identifiers can be concatenated.
    (active, concat_idents, "1.0.0", Some(29599), None),

    // Allows using the `#[link_args]` attribute.
    (active, link_args, "1.0.0", Some(29596), None),

    // Allows defining identifiers beyond ASCII.
    (active, non_ascii_idents, "1.0.0", Some(55467), None),

    // Allows using `#[plugin_registrar]` on functions.
    (active, plugin_registrar, "1.0.0", Some(29597), None),

    // Allows using `#![plugin(myplugin)]`.
    (active, plugin, "1.0.0", Some(29597), None),

    // Allows using `#[thread_local]` on `static` items.
    (active, thread_local, "1.0.0", Some(29594), None),

    // Allows using the `log_syntax!` macro.
    (active, log_syntax, "1.0.0", Some(29598), None),

    // Allows using the `trace_macros!` macro.
    (active, trace_macros, "1.0.0", Some(29598), None),

    // Allows the use of SIMD types in functions declared in `extern` blocks.
    (active, simd_ffi, "1.0.0", Some(27731), None),

    // Allows using custom attributes (RFC 572).
    (active, custom_attribute, "1.0.0", Some(29642), None),

    // Allows using non lexical lifetimes (RFC 2094).
    (active, nll, "1.0.0", Some(43234), None),

    // Allows using slice patterns.
    (active, slice_patterns, "1.0.0", Some(62254), None),

    // Allows the definition of `const` functions with some advanced features.
    (active, const_fn, "1.2.0", Some(57563), None),

    // Allows associated type defaults.
    (active, associated_type_defaults, "1.2.0", Some(29661), None),

    // Allows `#![no_core]`.
    (active, no_core, "1.3.0", Some(29639), None),

    // Allows default type parameters to influence type inference.
    (active, default_type_parameter_fallback, "1.3.0", Some(27336), None),

    // Allows `repr(simd)` and importing the various simd intrinsics.
    (active, repr_simd, "1.4.0", Some(27731), None),

    // Allows `extern "platform-intrinsic" { ... }`.
    (active, platform_intrinsics, "1.4.0", Some(27731), None),

    // Allows `#[unwind(..)]`.
    //
    // Permits specifying whether a function should permit unwinding or abort on unwind.
    (active, unwind_attributes, "1.4.0", Some(58760), None),

    // Allows `#[no_debug]`.
    (active, no_debug, "1.5.0", Some(29721), None),

    // Allows attributes on expressions and non-item statements.
    (active, stmt_expr_attributes, "1.6.0", Some(15701), None),

    // Allows the use of type ascription in expressions.
    (active, type_ascription, "1.6.0", Some(23416), None),

    // Allows `cfg(target_thread_local)`.
    (active, cfg_target_thread_local, "1.7.0", Some(29594), None),

    // Allows specialization of implementations (RFC 1210).
    (active, specialization, "1.7.0", Some(31844), None),

    // Allows using `#[naked]` on functions.
    (active, naked_functions, "1.9.0", Some(32408), None),

    // Allows `cfg(target_has_atomic = "...")`.
    (active, cfg_target_has_atomic, "1.9.0", Some(32976), None),

    // Allows `X..Y` patterns.
    (active, exclusive_range_pattern, "1.11.0", Some(37854), None),

    // Allows the `!` type. Does not imply 'exhaustive_patterns' (below) any more.
    (active, never_type, "1.13.0", Some(35121), None),

    // Allows exhaustive pattern matching on types that contain uninhabited types.
    (active, exhaustive_patterns, "1.13.0", Some(51085), None),

    // Allows untagged unions `union U { ... }`.
    (active, untagged_unions, "1.13.0", Some(32836), None),

    // Allows `#[link(..., cfg(..))]`.
    (active, link_cfg, "1.14.0", Some(37406), None),

    // Allows `extern "ptx-*" fn()`.
    (active, abi_ptx, "1.15.0", Some(38788), None),

    // Allows the `#[repr(i128)]` attribute for enums.
    (active, repr128, "1.16.0", Some(35118), None),

    // Allows `#[link(kind="static-nobundle"...)]`.
    (active, static_nobundle, "1.16.0", Some(37403), None),

    // Allows `extern "msp430-interrupt" fn()`.
    (active, abi_msp430_interrupt, "1.16.0", Some(38487), None),

    // Allows declarative macros 2.0 (`macro`).
    (active, decl_macro, "1.17.0", Some(39412), None),

    // Allows `extern "x86-interrupt" fn()`.
    (active, abi_x86_interrupt, "1.17.0", Some(40180), None),

    // Allows module-level inline assembly by way of `global_asm!()`.
    (active, global_asm, "1.18.0", Some(35119), None),

    // Allows overlapping impls of marker traits.
    (active, overlapping_marker_traits, "1.18.0", Some(29864), None),

    // Allows a test to fail without failing the whole suite.
    (active, allow_fail, "1.19.0", Some(46488), None),

    // Allows unsized tuple coercion.
    (active, unsized_tuple_coercion, "1.20.0", Some(42877), None),

    // Allows defining generators.
    (active, generators, "1.21.0", Some(43122), None),

    // Allows `#[doc(cfg(...))]`.
    (active, doc_cfg, "1.21.0", Some(43781), None),

    // Allows `#[doc(masked)]`.
    (active, doc_masked, "1.21.0", Some(44027), None),

    // Allows `#[doc(spotlight)]`.
    (active, doc_spotlight, "1.22.0", Some(45040), None),

    // Allows `#[doc(include = "some-file")]`.
    (active, external_doc, "1.22.0", Some(44732), None),

    // Allows future-proofing enums/structs with the `#[non_exhaustive]` attribute (RFC 2008).
    (active, non_exhaustive, "1.22.0", Some(44109), None),

    // Allows using `crate` as visibility modifier, synonymous with `pub(crate)`.
    (active, crate_visibility_modifier, "1.23.0", Some(53120), None),

    // Allows defining `extern type`s.
    (active, extern_types, "1.23.0", Some(43467), None),

    // Allows trait methods with arbitrary self types.
    (active, arbitrary_self_types, "1.23.0", Some(44874), None),

    // Allows in-band quantification of lifetime bindings (e.g., `fn foo(x: &'a u8) -> &'a u8`).
    (active, in_band_lifetimes, "1.23.0", Some(44524), None),

    // Allows associated types to be generic, e.g., `type Foo<T>;` (RFC 1598).
    (active, generic_associated_types, "1.23.0", Some(44265), None),

    // Allows defining `trait X = A + B;` alias items.
    (active, trait_alias, "1.24.0", Some(41517), None),

    // Allows infering `'static` outlives requirements (RFC 2093).
    (active, infer_static_outlives_requirements, "1.26.0", Some(54185), None),

    // Allows macro invocations in `extern {}` blocks.
    (active, macros_in_extern, "1.27.0", Some(49476), None),

    // Allows accessing fields of unions inside `const` functions.
    (active, const_fn_union, "1.27.0", Some(51909), None),

    // Allows casting raw pointers to `usize` during const eval.
    (active, const_raw_ptr_to_usize_cast, "1.27.0", Some(51910), None),

    // Allows dereferencing raw pointers during const eval.
    (active, const_raw_ptr_deref, "1.27.0", Some(51911), None),

    // Allows comparing raw pointers during const eval.
    (active, const_compare_raw_pointers, "1.27.0", Some(53020), None),

    // Allows `#[doc(alias = "...")]`.
    (active, doc_alias, "1.27.0", Some(50146), None),

    // Allows defining `existential type`s.
    (active, existential_type, "1.28.0", Some(34511), None),

    // Allows inconsistent bounds in where clauses.
    (active, trivial_bounds, "1.28.0", Some(48214), None),

    // Allows `'a: { break 'a; }`.
    (active, label_break_value, "1.28.0", Some(48594), None),

    // Allows using `#[doc(keyword = "...")]`.
    (active, doc_keyword, "1.28.0", Some(51315), None),

    // Allows async and await syntax.
    (active, async_await, "1.28.0", Some(50547), None),

    // Allows await! macro-like syntax.
    // This will likely be removed prior to stabilization of async/await.
    (active, await_macro, "1.28.0", Some(50547), None),

    // Allows reinterpretation of the bits of a value of one type as another type during const eval.
    (active, const_transmute, "1.29.0", Some(53605), None),

    // Allows using `try {...}` expressions.
    (active, try_blocks, "1.29.0", Some(31436), None),

    // Allows defining an `#[alloc_error_handler]`.
    (active, alloc_error_handler, "1.29.0", Some(51540), None),

    // Allows using the `amdgpu-kernel` ABI.
    (active, abi_amdgpu_kernel, "1.29.0", Some(51575), None),

    // Allows panicking during const eval (producing compile-time errors).
    (active, const_panic, "1.30.0", Some(51999), None),

    // Allows `#[marker]` on certain traits allowing overlapping implementations.
    (active, marker_trait_attr, "1.30.0", Some(29864), None),

    // Allows macro invocations on modules expressions and statements and
    // procedural macros to expand to non-items.
    (active, proc_macro_hygiene, "1.30.0", Some(54727), None),

    // Allows unsized rvalues at arguments and parameters.
    (active, unsized_locals, "1.30.0", Some(48055), None),

    // Allows custom test frameworks with `#![test_runner]` and `#[test_case]`.
    (active, custom_test_frameworks, "1.30.0", Some(50297), None),

    // Allows non-builtin attributes in inner attribute position.
    (active, custom_inner_attributes, "1.30.0", Some(54726), None),

    // Allows mixing bind-by-move in patterns and references to those identifiers in guards.
    (active, bind_by_move_pattern_guards, "1.30.0", Some(15287), None),

    // Allows `impl Trait` in bindings (`let`, `const`, `static`).
    (active, impl_trait_in_bindings, "1.30.0", Some(34511), None),

    // Allows using `reason` in lint attributes and the `#[expect(lint)]` lint check.
    (active, lint_reasons, "1.31.0", Some(54503), None),

    // Allows exhaustive integer pattern matching on `usize` and `isize`.
    (active, precise_pointer_size_matching, "1.32.0", Some(56354), None),

    // Allows relaxing the coherence rules such that
    // `impl<T> ForeignTrait<LocalType> for ForeignType<T> is permitted.
    (active, re_rebalance_coherence, "1.32.0", Some(55437), None),

    // Allows using `#[ffi_returns_twice]` on foreign functions.
    (active, ffi_returns_twice, "1.34.0", Some(58314), None),

    // Allows const generic types (e.g. `struct Foo<const N: usize>(...);`).
    (active, const_generics, "1.34.0", Some(44580), None),

    // Allows using `#[optimize(X)]`.
    (active, optimize_attribute, "1.34.0", Some(54882), None),

    // Allows using C-variadics.
    (active, c_variadic, "1.34.0", Some(44930), None),

    // Allows the user of associated type bounds.
    (active, associated_type_bounds, "1.34.0", Some(52662), None),

    // Attributes on formal function params.
    (active, param_attrs, "1.36.0", Some(60406), None),

    // Allows calling constructor functions in `const fn`.
    (active, const_constructor, "1.37.0", Some(61456), None),

    // Allows `if/while p && let q = r && ...` chains.
    (active, let_chains, "1.37.0", Some(53667), None),

    // #[repr(transparent)] on enums.
    (active, transparent_enums, "1.37.0", Some(60405), None),

    // #[repr(transparent)] on unions.
    (active, transparent_unions, "1.37.0", Some(60405), None),

    // Allows explicit discriminants on non-unit enum variants.
    (active, arbitrary_enum_discriminant, "1.37.0", Some(60553), None),

    // Allows `impl Trait` with multiple unrelated lifetimes.
    (active, member_constraints, "1.37.0", Some(61977), None),

    // Allows `async || body` closures.
    (active, async_closure, "1.37.0", Some(62290), None),

    // -------------------------------------------------------------------------
    // feature-group-end: actual feature gates
    // -------------------------------------------------------------------------
);

// Some features are known to be incomplete and using them is likely to have
// unanticipated results, such as compiler crashes. We warn the user about these
// to alert them.
const INCOMPLETE_FEATURES: &[Symbol] = &[
    sym::impl_trait_in_bindings,
    sym::generic_associated_types,
    sym::const_generics,
    sym::let_chains,
];

declare_features! (
    // -------------------------------------------------------------------------
    // feature-group-start: removed features
    // -------------------------------------------------------------------------

    (removed, import_shadowing, "1.0.0", None, None, None),
    (removed, managed_boxes, "1.0.0", None, None, None),
    // Allows use of unary negate on unsigned integers, e.g., -e for e: u8
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
    (removed, unmarked_api, "1.0.0", None, None, None),
    (removed, allocator, "1.0.0", None, None, None),
    (removed, simd, "1.0.0", Some(27731), None,
     Some("removed in favor of `#[repr(simd)]`")),
    (removed, advanced_slice_patterns, "1.0.0", Some(62254), None,
     Some("merged into `#![feature(slice_patterns)]`")),
    (removed, macro_reexport, "1.0.0", Some(29638), None,
     Some("subsumed by `pub use`")),
    (removed, pushpop_unsafe, "1.2.0", None, None, None),
    (removed, needs_allocator, "1.4.0", Some(27389), None,
     Some("subsumed by `#![feature(allocator_internals)]`")),
    (removed, proc_macro_mod, "1.27.0", Some(54727), None,
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, proc_macro_expr, "1.27.0", Some(54727), None,
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, proc_macro_non_items, "1.27.0", Some(54727), None,
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, proc_macro_gen, "1.27.0", Some(54727), None,
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, panic_implementation, "1.28.0", Some(44489), None,
     Some("subsumed by `#[panic_handler]`")),
    // Allows the use of `#[derive(Anything)]` as sugar for `#[derive_Anything]`.
    (removed, custom_derive, "1.32.0", Some(29644), None,
     Some("subsumed by `#[proc_macro_derive]`")),
    // Paths of the form: `extern::foo::bar`
    (removed, extern_in_paths, "1.33.0", Some(55600), None,
     Some("subsumed by `::foo::bar` paths")),
    (removed, quote, "1.33.0", Some(29601), None, None),

    // -------------------------------------------------------------------------
    // feature-group-end: removed features
    // -------------------------------------------------------------------------
);

declare_features! (
    (stable_removed, no_stack_check, "1.0.0", None, None),
);

declare_features! (
    // -------------------------------------------------------------------------
    // feature-group-start: for testing purposes
    // -------------------------------------------------------------------------

    // A temporary feature gate used to enable parser extensions needed
    // to bootstrap fix for #5723.
    (accepted, issue_5723_bootstrap, "1.0.0", None, None),
    // These are used to test this portion of the compiler,
    // they don't actually mean anything.
    (accepted, test_accepted_feature, "1.0.0", None, None),

    // -------------------------------------------------------------------------
    // feature-group-end: for testing purposes
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // feature-group-start: accepted features
    // -------------------------------------------------------------------------

    // Allows using associated `type`s in `trait`s.
    (accepted, associated_types, "1.0.0", None, None),
    // Allows using assigning a default type to type parameters in algebraic data type definitions.
    (accepted, default_type_params, "1.0.0", None, None),
    // FIXME: explain `globs`.
    (accepted, globs, "1.0.0", None, None),
    // Allows `macro_rules!` items.
    (accepted, macro_rules, "1.0.0", None, None),
    // Allows use of `&foo[a..b]` as a slicing syntax.
    (accepted, slicing_syntax, "1.0.0", None, None),
    // Allows struct variants `Foo { baz: u8, .. }` in enums (RFC 418).
    (accepted, struct_variant, "1.0.0", None, None),
    // Allows indexing tuples.
    (accepted, tuple_indexing, "1.0.0", None, None),
    // Allows the use of `if let` expressions.
    (accepted, if_let, "1.0.0", None, None),
    // Allows the use of `while let` expressions.
    (accepted, while_let, "1.0.0", None, None),
    // Allows using `#![no_std]`.
    (accepted, no_std, "1.6.0", None, None),
    // Allows overloading augmented assignment operations like `a += b`.
    (accepted, augmented_assignments, "1.8.0", Some(28235), None),
    // Allows empty structs and enum variants with braces.
    (accepted, braced_empty_structs, "1.8.0", Some(29720), None),
    // Allows `#[deprecated]` attribute.
    (accepted, deprecated, "1.9.0", Some(29935), None),
    // Allows macros to appear in the type position.
    (accepted, type_macros, "1.13.0", Some(27245), None),
    // Allows use of the postfix `?` operator in expressions.
    (accepted, question_mark, "1.13.0", Some(31436), None),
    // Allows `..` in tuple (struct) patterns.
    (accepted, dotdot_in_tuple_patterns, "1.14.0", Some(33627), None),
    // Allows some increased flexibility in the name resolution rules,
    // especially around globs and shadowing (RFC 1560).
    (accepted, item_like_imports, "1.15.0", Some(35120), None),
    // Allows using `Self` and associated types in struct expressions and patterns.
    (accepted, more_struct_aliases, "1.16.0", Some(37544), None),
    // Allows elision of `'static` lifetimes in `static`s and `const`s.
    (accepted, static_in_const, "1.17.0", Some(35897), None),
    // Allows field shorthands (`x` meaning `x: x`) in struct literal expressions.
    (accepted, field_init_shorthand, "1.17.0", Some(37340), None),
    // Allows the definition recursive static items.
    (accepted, static_recursion, "1.17.0", Some(29719), None),
    // Allows `pub(restricted)` visibilities (RFC 1422).
    (accepted, pub_restricted, "1.18.0", Some(32409), None),
    // Allows `#![windows_subsystem]`.
    (accepted, windows_subsystem, "1.18.0", Some(37499), None),
    // Allows `break {expr}` with a value inside `loop`s.
    (accepted, loop_break_value, "1.19.0", Some(37339), None),
    // Allows numeric fields in struct expressions and patterns.
    (accepted, relaxed_adts, "1.19.0", Some(35626), None),
    // Allows coercing non capturing closures to function pointers.
    (accepted, closure_to_fn_coercion, "1.19.0", Some(39817), None),
    // Allows attributes on struct literal fields.
    (accepted, struct_field_attributes, "1.20.0", Some(38814), None),
    // Allows the definition of associated constants in `trait` or `impl` blocks.
    (accepted, associated_consts, "1.20.0", Some(29646), None),
    // Allows usage of the `compile_error!` macro.
    (accepted, compile_error, "1.20.0", Some(40872), None),
    // Allows code like `let x: &'static u32 = &42` to work (RFC 1414).
    (accepted, rvalue_static_promotion, "1.21.0", Some(38865), None),
    // Allows `Drop` types in constants (RFC 1440).
    (accepted, drop_types_in_const, "1.22.0", Some(33156), None),
    // Allows the sysV64 ABI to be specified on all platforms
    // instead of just the platforms on which it is the C ABI.
    (accepted, abi_sysv64, "1.24.0", Some(36167), None),
    // Allows `repr(align(16))` struct attribute (RFC 1358).
    (accepted, repr_align, "1.25.0", Some(33626), None),
    // Allows '|' at beginning of match arms (RFC 1925).
    (accepted, match_beginning_vert, "1.25.0", Some(44101), None),
    // Allows nested groups in `use` items (RFC 2128).
    (accepted, use_nested_groups, "1.25.0", Some(44494), None),
    // Allows indexing into constant arrays.
    (accepted, const_indexing, "1.26.0", Some(29947), None),
    // Allows using `a..=b` and `..=b` as inclusive range syntaxes.
    (accepted, inclusive_range_syntax, "1.26.0", Some(28237), None),
    // Allows `..=` in patterns (RFC 1192).
    (accepted, dotdoteq_in_patterns, "1.26.0", Some(28237), None),
    // Allows `fn main()` with return types which implements `Termination` (RFC 1937).
    (accepted, termination_trait, "1.26.0", Some(43301), None),
    // Allows implementing `Clone` for closures where possible (RFC 2132).
    (accepted, clone_closures, "1.26.0", Some(44490), None),
    // Allows implementing `Copy` for closures where possible (RFC 2132).
    (accepted, copy_closures, "1.26.0", Some(44490), None),
    // Allows `impl Trait` in function arguments.
    (accepted, universal_impl_trait, "1.26.0", Some(34511), None),
    // Allows `impl Trait` in function return types.
    (accepted, conservative_impl_trait, "1.26.0", Some(34511), None),
    // Allows using the `u128` and `i128` types.
    (accepted, i128_type, "1.26.0", Some(35118), None),
    // Allows default match binding modes (RFC 2005).
    (accepted, match_default_bindings, "1.26.0", Some(42640), None),
    // Allows `'_` placeholder lifetimes.
    (accepted, underscore_lifetimes, "1.26.0", Some(44524), None),
    // Allows attributes on lifetime/type formal parameters in generics (RFC 1327).
    (accepted, generic_param_attrs, "1.27.0", Some(48848), None),
    // Allows `cfg(target_feature = "...")`.
    (accepted, cfg_target_feature, "1.27.0", Some(29717), None),
    // Allows `#[target_feature(...)]`.
    (accepted, target_feature, "1.27.0", None, None),
    // Allows using `dyn Trait` as a syntax for trait objects.
    (accepted, dyn_trait, "1.27.0", Some(44662), None),
    // Allows `#[must_use]` on functions, and introduces must-use operators (RFC 1940).
    (accepted, fn_must_use, "1.27.0", Some(43302), None),
    // Allows use of the `:lifetime` macro fragment specifier.
    (accepted, macro_lifetime_matcher, "1.27.0", Some(34303), None),
    // Allows `#[test]` functions where the return type implements `Termination` (RFC 1937).
    (accepted, termination_trait_test, "1.27.0", Some(48854), None),
    // Allows the `#[global_allocator]` attribute.
    (accepted, global_allocator, "1.28.0", Some(27389), None),
    // Allows `#[repr(transparent)]` attribute on newtype structs.
    (accepted, repr_transparent, "1.28.0", Some(43036), None),
    // Allows procedural macros in `proc-macro` crates.
    (accepted, proc_macro, "1.29.0", Some(38356), None),
    // Allows `foo.rs` as an alternative to `foo/mod.rs`.
    (accepted, non_modrs_mods, "1.30.0", Some(44660), None),
    // Allows use of the `:vis` macro fragment specifier
    (accepted, macro_vis_matcher, "1.30.0", Some(41022), None),
    // Allows importing and reexporting macros with `use`,
    // enables macro modularization in general.
    (accepted, use_extern_macros, "1.30.0", Some(35896), None),
    // Allows keywords to be escaped for use as identifiers.
    (accepted, raw_identifiers, "1.30.0", Some(48589), None),
    // Allows attributes scoped to tools.
    (accepted, tool_attributes, "1.30.0", Some(44690), None),
    // Allows multi-segment paths in attributes and derives.
    (accepted, proc_macro_path_invoc, "1.30.0", Some(38356), None),
    // Allows all literals in attribute lists and values of key-value pairs.
    (accepted, attr_literals, "1.30.0", Some(34981), None),
    // Allows inferring outlives requirements (RFC 2093).
    (accepted, infer_outlives_requirements, "1.30.0", Some(44493), None),
    // Allows annotating functions conforming to `fn(&PanicInfo) -> !` with `#[panic_handler]`.
    // This defines the behavior of panics.
    (accepted, panic_handler, "1.30.0", Some(44489), None),
    // Allows `#[used]` to preserve symbols (see llvm.used).
    (accepted, used, "1.30.0", Some(40289), None),
    // Allows `crate` in paths.
    (accepted, crate_in_paths, "1.30.0", Some(45477), None),
    // Allows resolving absolute paths as paths from other crates.
    (accepted, extern_absolute_paths, "1.30.0", Some(44660), None),
    // Allows access to crate names passed via `--extern` through prelude.
    (accepted, extern_prelude, "1.30.0", Some(44660), None),
    // Allows parentheses in patterns.
    (accepted, pattern_parentheses, "1.31.0", Some(51087), None),
    // Allows the definition of `const fn` functions.
    (accepted, min_const_fn, "1.31.0", Some(53555), None),
    // Allows scoped lints.
    (accepted, tool_lints, "1.31.0", Some(44690), None),
    // Allows lifetime elision in `impl` headers. For example:
    // + `impl<I:Iterator> Iterator for &mut Iterator`
    // + `impl Debug for Foo<'_>`
    (accepted, impl_header_lifetime_elision, "1.31.0", Some(15872), None),
    // Allows `extern crate foo as bar;`. This puts `bar` into extern prelude.
    (accepted, extern_crate_item_prelude, "1.31.0", Some(55599), None),
    // Allows use of the `:literal` macro fragment specifier (RFC 1576).
    (accepted, macro_literal_matcher, "1.32.0", Some(35625), None),
    // Allows use of `?` as the Kleene "at most one" operator in macros.
    (accepted, macro_at_most_once_rep, "1.32.0", Some(48075), None),
    // Allows `Self` struct constructor (RFC 2302).
    (accepted, self_struct_ctor, "1.32.0", Some(51994), None),
    // Allows `Self` in type definitions (RFC 2300).
    (accepted, self_in_typedefs, "1.32.0", Some(49303), None),
    // Allows `use x::y;` to search `x` in the current scope.
    (accepted, uniform_paths, "1.32.0", Some(53130), None),
    // Allows integer match exhaustiveness checking (RFC 2591).
    (accepted, exhaustive_integer_patterns, "1.33.0", Some(50907), None),
    // Allows `use path as _;` and `extern crate c as _;`.
    (accepted, underscore_imports, "1.33.0", Some(48216), None),
    // Allows `#[repr(packed(N))]` attribute on structs.
    (accepted, repr_packed, "1.33.0", Some(33158), None),
    // Allows irrefutable patterns in `if let` and `while let` statements (RFC 2086).
    (accepted, irrefutable_let_patterns, "1.33.0", Some(44495), None),
    // Allows calling `const unsafe fn` inside `unsafe` blocks in `const fn` functions.
    (accepted, min_const_unsafe_fn, "1.33.0", Some(55607), None),
    // Allows let bindings, assignments and destructuring in `const` functions and constants.
    // As long as control flow is not implemented in const eval, `&&` and `||` may not be used
    // at the same time as let bindings.
    (accepted, const_let, "1.33.0", Some(48821), None),
    // Allows `#[cfg_attr(predicate, multiple, attributes, here)]`.
    (accepted, cfg_attr_multi, "1.33.0", Some(54881), None),
    // Allows top level or-patterns (`p | q`) in `if let` and `while let`.
    (accepted, if_while_or_patterns, "1.33.0", Some(48215), None),
    // Allows `cfg(target_vendor = "...")`.
    (accepted, cfg_target_vendor, "1.33.0", Some(29718), None),
    // Allows `extern crate self as foo;`.
    // This puts local crate root into extern prelude under name `foo`.
    (accepted, extern_crate_self, "1.34.0", Some(56409), None),
    // Allows arbitrary delimited token streams in non-macro attributes.
    (accepted, unrestricted_attribute_tokens, "1.34.0", Some(55208), None),
    // Allows paths to enum variants on type aliases including `Self`.
    (accepted, type_alias_enum_variants, "1.37.0", Some(49683), None),
    // Allows using `#[repr(align(X))]` on enums with equivalent semantics
    // to wrapping an enum in a wrapper struct with `#[repr(align(X))]`.
    (accepted, repr_align_enum, "1.37.0", Some(57996), None),
    // Allows `const _: TYPE = VALUE`.
    (accepted, underscore_const_names, "1.37.0", Some(54912), None),

    // -------------------------------------------------------------------------
    // feature-group-end: accepted features
    // -------------------------------------------------------------------------
);

// If you change this, please modify `src/doc/unstable-book` as well. You must
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
    Gated(Stability, Symbol, &'static str, fn(&Features) -> bool),

    /// Ungated attribute, can be used on all release channels
    Ungated,
}

/// A template that the attribute input must match.
/// Only top-level shape (`#[attr]` vs `#[attr(...)]` vs `#[attr = ...]`) is considered now.
#[derive(Clone, Copy)]
pub struct AttributeTemplate {
    word: bool,
    list: Option<&'static str>,
    name_value_str: Option<&'static str>,
}

impl AttributeTemplate {
    /// Checks that the given meta-item is compatible with this template.
    fn compatible(&self, meta_item_kind: &ast::MetaItemKind) -> bool {
        match meta_item_kind {
            ast::MetaItemKind::Word => self.word,
            ast::MetaItemKind::List(..) => self.list.is_some(),
            ast::MetaItemKind::NameValue(lit) if lit.node.is_str() => self.name_value_str.is_some(),
            ast::MetaItemKind::NameValue(..) => false,
        }
    }
}

/// A convenience macro for constructing attribute templates.
/// E.g., `template!(Word, List: "description")` means that the attribute
/// supports forms `#[attr]` and `#[attr(description)]`.
macro_rules! template {
    (Word) => { template!(@ true, None, None) };
    (List: $descr: expr) => { template!(@ false, Some($descr), None) };
    (NameValueStr: $descr: expr) => { template!(@ false, None, Some($descr)) };
    (Word, List: $descr: expr) => { template!(@ true, Some($descr), None) };
    (Word, NameValueStr: $descr: expr) => { template!(@ true, None, Some($descr)) };
    (List: $descr1: expr, NameValueStr: $descr2: expr) => {
        template!(@ false, Some($descr1), Some($descr2))
    };
    (Word, List: $descr1: expr, NameValueStr: $descr2: expr) => {
        template!(@ true, Some($descr1), Some($descr2))
    };
    (@ $word: expr, $list: expr, $name_value_str: expr) => { AttributeTemplate {
        word: $word, list: $list, name_value_str: $name_value_str
    } };
}

impl AttributeGate {
    fn is_deprecated(&self) -> bool {
        match *self {
            Gated(Stability::Deprecated(_, _), ..) => true,
            _ => false,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Stability {
    Unstable,
    // First argument is tracking issue link; second argument is an optional
    // help message, which defaults to "remove this attribute"
    Deprecated(&'static str, Option<&'static str>),
}

// fn() is not Debug
impl std::fmt::Debug for AttributeGate {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

pub fn deprecated_attributes() -> Vec<&'static (Symbol, AttributeType,
                                                AttributeTemplate, AttributeGate)> {
    BUILTIN_ATTRIBUTES.iter().filter(|(.., gate)| gate.is_deprecated()).collect()
}

pub fn is_builtin_attr_name(name: ast::Name) -> bool {
    BUILTIN_ATTRIBUTE_MAP.get(&name).is_some()
}

pub fn is_builtin_attr(attr: &ast::Attribute) -> bool {
    attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name)).is_some()
}

/// Attributes that have a special meaning to rustc or rustdoc
pub const BUILTIN_ATTRIBUTES: &[BuiltinAttribute] = &[
    // Normal attributes

    (
        sym::warn,
        Normal,
        template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#),
        Ungated
    ),
    (
        sym::allow,
        Normal,
        template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#),
        Ungated
    ),
    (
        sym::forbid,
        Normal,
        template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#),
        Ungated
    ),
    (
        sym::deny,
        Normal,
        template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#),
        Ungated
    ),

    (sym::macro_use, Normal, template!(Word, List: "name1, name2, ..."), Ungated),
    (sym::macro_export, Normal, template!(Word, List: "local_inner_macros"), Ungated),
    (sym::plugin_registrar, Normal, template!(Word), Ungated),

    (sym::cfg, Normal, template!(List: "predicate"), Ungated),
    (sym::cfg_attr, Normal, template!(List: "predicate, attr1, attr2, ..."), Ungated),
    (sym::main, Normal, template!(Word), Ungated),
    (sym::start, Normal, template!(Word), Ungated),
    (sym::repr, Normal, template!(List: "C, packed, ..."), Ungated),
    (sym::path, Normal, template!(NameValueStr: "file"), Ungated),
    (sym::automatically_derived, Normal, template!(Word), Ungated),
    (sym::no_mangle, Whitelisted, template!(Word), Ungated),
    (sym::no_link, Normal, template!(Word), Ungated),
    (sym::derive, Normal, template!(List: "Trait1, Trait2, ..."), Ungated),
    (
        sym::should_panic,
        Normal,
        template!(Word, List: r#"expected = "reason"#, NameValueStr: "reason"),
        Ungated
    ),
    (sym::ignore, Normal, template!(Word, NameValueStr: "reason"), Ungated),
    (sym::no_implicit_prelude, Normal, template!(Word), Ungated),
    (sym::reexport_test_harness_main, Normal, template!(NameValueStr: "name"), Ungated),
    (sym::link_args, Normal, template!(NameValueStr: "args"), Gated(Stability::Unstable,
                                sym::link_args,
                                "the `link_args` attribute is experimental and not \
                                portable across platforms, it is recommended to \
                                use `#[link(name = \"foo\")] instead",
                                cfg_fn!(link_args))),
    (sym::macro_escape, Normal, template!(Word), Ungated),

    // RFC #1445.
    (sym::structural_match, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                            sym::structural_match,
                                            "the semantics of constant patterns is \
                                            not yet settled",
                                            cfg_fn!(structural_match))),

    // RFC #2008
    (sym::non_exhaustive, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                        sym::non_exhaustive,
                                        "non exhaustive is an experimental feature",
                                        cfg_fn!(non_exhaustive))),

    // RFC #1268
    (sym::marker, Normal, template!(Word), Gated(Stability::Unstable,
                            sym::marker_trait_attr,
                            "marker traits is an experimental feature",
                            cfg_fn!(marker_trait_attr))),

    (sym::plugin, CrateLevel, template!(List: "name|name(args)"), Gated(Stability::Unstable,
                                sym::plugin,
                                "compiler plugins are experimental \
                                and possibly buggy",
                                cfg_fn!(plugin))),

    (sym::no_std, CrateLevel, template!(Word), Ungated),
    (sym::no_core, CrateLevel, template!(Word), Gated(Stability::Unstable,
                                sym::no_core,
                                "no_core is experimental",
                                cfg_fn!(no_core))),
    (sym::lang, Normal, template!(NameValueStr: "name"), Gated(Stability::Unstable,
                        sym::lang_items,
                        "language items are subject to change",
                        cfg_fn!(lang_items))),
    (sym::linkage, Whitelisted, template!(NameValueStr: "external|internal|..."),
                                Gated(Stability::Unstable,
                                sym::linkage,
                                "the `linkage` attribute is experimental \
                                    and not portable across platforms",
                                cfg_fn!(linkage))),
    (sym::thread_local, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                        sym::thread_local,
                                        "`#[thread_local]` is an experimental feature, and does \
                                         not currently handle destructors",
                                        cfg_fn!(thread_local))),

    (sym::rustc_on_unimplemented, Whitelisted, template!(List:
                        r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#,
                        NameValueStr: "message"),
                                            Gated(Stability::Unstable,
                                            sym::on_unimplemented,
                                            "the `#[rustc_on_unimplemented]` attribute \
                                            is an experimental feature",
                                            cfg_fn!(on_unimplemented))),
    (sym::rustc_const_unstable, Normal, template!(List: r#"feature = "name""#),
                                            Gated(Stability::Unstable,
                                            sym::rustc_const_unstable,
                                            "the `#[rustc_const_unstable]` attribute \
                                            is an internal feature",
                                            cfg_fn!(rustc_const_unstable))),
    (sym::global_allocator, Normal, template!(Word), Ungated),
    (sym::default_lib_allocator, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                            sym::allocator_internals,
                                            "the `#[default_lib_allocator]` \
                                            attribute is an experimental feature",
                                            cfg_fn!(allocator_internals))),
    (sym::needs_allocator, Normal, template!(Word), Gated(Stability::Unstable,
                                    sym::allocator_internals,
                                    "the `#[needs_allocator]` \
                                    attribute is an experimental \
                                    feature",
                                    cfg_fn!(allocator_internals))),
    (sym::panic_runtime, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                        sym::panic_runtime,
                                        "the `#[panic_runtime]` attribute is \
                                        an experimental feature",
                                        cfg_fn!(panic_runtime))),
    (sym::needs_panic_runtime, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                            sym::needs_panic_runtime,
                                            "the `#[needs_panic_runtime]` \
                                                attribute is an experimental \
                                                feature",
                                            cfg_fn!(needs_panic_runtime))),
    (sym::rustc_outlives, Normal, template!(Word), Gated(Stability::Unstable,
                                    sym::rustc_attrs,
                                    "the `#[rustc_outlives]` attribute \
                                    is just used for rustc unit tests \
                                    and will never be stable",
                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_variance, Normal, template!(Word), Gated(Stability::Unstable,
                                    sym::rustc_attrs,
                                    "the `#[rustc_variance]` attribute \
                                    is just used for rustc unit tests \
                                    and will never be stable",
                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_layout, Normal, template!(List: "field1, field2, ..."),
    Gated(Stability::Unstable,
        sym::rustc_attrs,
        "the `#[rustc_layout]` attribute \
            is just used for rustc unit tests \
            and will never be stable",
        cfg_fn!(rustc_attrs))),
    (sym::rustc_layout_scalar_valid_range_start, Whitelisted, template!(List: "value"),
    Gated(Stability::Unstable,
        sym::rustc_attrs,
        "the `#[rustc_layout_scalar_valid_range_start]` attribute \
            is just used to enable niche optimizations in libcore \
            and will never be stable",
        cfg_fn!(rustc_attrs))),
    (sym::rustc_layout_scalar_valid_range_end, Whitelisted, template!(List: "value"),
    Gated(Stability::Unstable,
        sym::rustc_attrs,
        "the `#[rustc_layout_scalar_valid_range_end]` attribute \
            is just used to enable niche optimizations in libcore \
            and will never be stable",
        cfg_fn!(rustc_attrs))),
    (sym::rustc_nonnull_optimization_guaranteed, Whitelisted, template!(Word),
    Gated(Stability::Unstable,
        sym::rustc_attrs,
        "the `#[rustc_nonnull_optimization_guaranteed]` attribute \
            is just used to enable niche optimizations in libcore \
            and will never be stable",
        cfg_fn!(rustc_attrs))),
    (sym::rustc_regions, Normal, template!(Word), Gated(Stability::Unstable,
                                    sym::rustc_attrs,
                                    "the `#[rustc_regions]` attribute \
                                    is just used for rustc unit tests \
                                    and will never be stable",
                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_error, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                    sym::rustc_attrs,
                                    "the `#[rustc_error]` attribute \
                                        is just used for rustc unit tests \
                                        and will never be stable",
                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_dump_user_substs, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                    sym::rustc_attrs,
                                    "this attribute \
                                        is just used for rustc unit tests \
                                        and will never be stable",
                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_if_this_changed, Whitelisted, template!(Word, List: "DepNode"),
                                                Gated(Stability::Unstable,
                                                sym::rustc_attrs,
                                                "the `#[rustc_if_this_changed]` attribute \
                                                is just used for rustc unit tests \
                                                and will never be stable",
                                                cfg_fn!(rustc_attrs))),
    (sym::rustc_then_this_would_need, Whitelisted, template!(List: "DepNode"),
                                                    Gated(Stability::Unstable,
                                                    sym::rustc_attrs,
                                                    "the `#[rustc_if_this_changed]` attribute \
                                                    is just used for rustc unit tests \
                                                    and will never be stable",
                                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_dirty, Whitelisted, template!(List: r#"cfg = "...", /*opt*/ label = "...",
                                                    /*opt*/ except = "...""#),
                                    Gated(Stability::Unstable,
                                    sym::rustc_attrs,
                                    "the `#[rustc_dirty]` attribute \
                                        is just used for rustc unit tests \
                                        and will never be stable",
                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_clean, Whitelisted, template!(List: r#"cfg = "...", /*opt*/ label = "...",
                                                    /*opt*/ except = "...""#),
                                    Gated(Stability::Unstable,
                                    sym::rustc_attrs,
                                    "the `#[rustc_clean]` attribute \
                                        is just used for rustc unit tests \
                                        and will never be stable",
                                    cfg_fn!(rustc_attrs))),
    (
        sym::rustc_partition_reused,
        Whitelisted,
        template!(List: r#"cfg = "...", module = "...""#),
        Gated(
            Stability::Unstable,
            sym::rustc_attrs,
            "this attribute \
            is just used for rustc unit tests \
            and will never be stable",
            cfg_fn!(rustc_attrs)
        )
    ),
    (
        sym::rustc_partition_codegened,
        Whitelisted,
        template!(List: r#"cfg = "...", module = "...""#),
        Gated(
            Stability::Unstable,
            sym::rustc_attrs,
            "this attribute \
            is just used for rustc unit tests \
            and will never be stable",
            cfg_fn!(rustc_attrs),
        )
    ),
    (sym::rustc_expected_cgu_reuse, Whitelisted, template!(List: r#"cfg = "...", module = "...",
                                                            kind = "...""#),
                                                    Gated(Stability::Unstable,
                                                    sym::rustc_attrs,
                                                    "this attribute \
                                                    is just used for rustc unit tests \
                                                    and will never be stable",
                                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_synthetic, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                                    sym::rustc_attrs,
                                                    "this attribute \
                                                    is just used for rustc unit tests \
                                                    and will never be stable",
                                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_symbol_name, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                            sym::rustc_attrs,
                                            "internal rustc attributes will never be stable",
                                            cfg_fn!(rustc_attrs))),
    (sym::rustc_def_path, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                        sym::rustc_attrs,
                                        "internal rustc attributes will never be stable",
                                        cfg_fn!(rustc_attrs))),
    (sym::rustc_mir, Whitelisted, template!(List: "arg1, arg2, ..."), Gated(Stability::Unstable,
                                    sym::rustc_attrs,
                                    "the `#[rustc_mir]` attribute \
                                    is just used for rustc unit tests \
                                    and will never be stable",
                                    cfg_fn!(rustc_attrs))),
    (
        sym::rustc_inherit_overflow_checks,
        Whitelisted,
        template!(Word),
        Gated(
            Stability::Unstable,
            sym::rustc_attrs,
            "the `#[rustc_inherit_overflow_checks]` \
            attribute is just used to control \
            overflow checking behavior of several \
            libcore functions that are inlined \
            across crates and will never be stable",
            cfg_fn!(rustc_attrs),
        )
    ),

    (sym::rustc_dump_program_clauses, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                                    sym::rustc_attrs,
                                                    "the `#[rustc_dump_program_clauses]` \
                                                    attribute is just used for rustc unit \
                                                    tests and will never be stable",
                                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_test_marker, Normal, template!(Word), Gated(Stability::Unstable,
                                    sym::rustc_attrs,
                                    "the `#[rustc_test_marker]` attribute \
                                    is used internally to track tests",
                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_transparent_macro, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                                sym::rustc_attrs,
                                                "used internally for testing macro hygiene",
                                                    cfg_fn!(rustc_attrs))),
    (sym::compiler_builtins, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                            sym::compiler_builtins,
                                            "the `#[compiler_builtins]` attribute is used to \
                                            identify the `compiler_builtins` crate which \
                                            contains compiler-rt intrinsics and will never be \
                                            stable",
                                        cfg_fn!(compiler_builtins))),
    (sym::sanitizer_runtime, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                            sym::sanitizer_runtime,
                                            "the `#[sanitizer_runtime]` attribute is used to \
                                            identify crates that contain the runtime of a \
                                            sanitizer and will never be stable",
                                            cfg_fn!(sanitizer_runtime))),
    (sym::profiler_runtime, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                            sym::profiler_runtime,
                                            "the `#[profiler_runtime]` attribute is used to \
                                            identify the `profiler_builtins` crate which \
                                            contains the profiler runtime and will never be \
                                            stable",
                                            cfg_fn!(profiler_runtime))),

    (sym::allow_internal_unstable, Normal, template!(Word, List: "feat1, feat2, ..."),
                                            Gated(Stability::Unstable,
                                            sym::allow_internal_unstable,
                                            EXPLAIN_ALLOW_INTERNAL_UNSTABLE,
                                            cfg_fn!(allow_internal_unstable))),

    (sym::allow_internal_unsafe, Normal, template!(Word), Gated(Stability::Unstable,
                                            sym::allow_internal_unsafe,
                                            EXPLAIN_ALLOW_INTERNAL_UNSAFE,
                                            cfg_fn!(allow_internal_unsafe))),

    (sym::fundamental, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                    sym::fundamental,
                                    "the `#[fundamental]` attribute \
                                        is an experimental feature",
                                    cfg_fn!(fundamental))),

    (sym::proc_macro_derive, Normal, template!(List: "TraitName, \
                                                /*opt*/ attributes(name1, name2, ...)"),
                                    Ungated),

    (sym::rustc_copy_clone_marker, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                                sym::rustc_attrs,
                                                "internal implementation detail",
                                                cfg_fn!(rustc_attrs))),

    (sym::rustc_allocator, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                                sym::rustc_attrs,
                                                "internal implementation detail",
                                                cfg_fn!(rustc_attrs))),

    (sym::rustc_dummy, Normal, template!(Word /* doesn't matter*/), Gated(Stability::Unstable,
                                         sym::rustc_attrs,
                                         "used by the test suite",
                                         cfg_fn!(rustc_attrs))),

    // FIXME: #14408 whitelist docs since rustdoc looks at them
    (
        sym::doc,
        Whitelisted,
        template!(List: "hidden|inline|...", NameValueStr: "string"),
        Ungated
    ),

    // FIXME: #14406 these are processed in codegen, which happens after the
    // lint pass
    (sym::cold, Whitelisted, template!(Word), Ungated),
    (sym::naked, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                sym::naked_functions,
                                "the `#[naked]` attribute \
                                is an experimental feature",
                                cfg_fn!(naked_functions))),
    (sym::ffi_returns_twice, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                sym::ffi_returns_twice,
                                "the `#[ffi_returns_twice]` attribute \
                                is an experimental feature",
                                cfg_fn!(ffi_returns_twice))),
    (sym::target_feature, Whitelisted, template!(List: r#"enable = "name""#), Ungated),
    (sym::export_name, Whitelisted, template!(NameValueStr: "name"), Ungated),
    (sym::inline, Whitelisted, template!(Word, List: "always|never"), Ungated),
    (sym::link, Whitelisted, template!(List: r#"name = "...", /*opt*/ kind = "dylib|static|...",
                                               /*opt*/ cfg = "...""#), Ungated),
    (sym::link_name, Whitelisted, template!(NameValueStr: "name"), Ungated),
    (sym::link_section, Whitelisted, template!(NameValueStr: "name"), Ungated),
    (sym::no_builtins, Whitelisted, template!(Word), Ungated),
    (sym::no_debug, Whitelisted, template!(Word), Gated(
        Stability::Deprecated("https://github.com/rust-lang/rust/issues/29721", None),
        sym::no_debug,
        "the `#[no_debug]` attribute was an experimental feature that has been \
        deprecated due to lack of demand",
        cfg_fn!(no_debug))),
    (
        sym::omit_gdb_pretty_printer_section,
        Whitelisted,
        template!(Word),
        Gated(
            Stability::Unstable,
            sym::omit_gdb_pretty_printer_section,
            "the `#[omit_gdb_pretty_printer_section]` \
                attribute is just used for the Rust test \
                suite",
            cfg_fn!(omit_gdb_pretty_printer_section)
        )
    ),
    (sym::unsafe_destructor_blind_to_params,
    Normal,
    template!(Word),
    Gated(Stability::Deprecated("https://github.com/rust-lang/rust/issues/34761",
                                Some("replace this attribute with `#[may_dangle]`")),
        sym::dropck_parametricity,
        "unsafe_destructor_blind_to_params has been replaced by \
            may_dangle and will be removed in the future",
        cfg_fn!(dropck_parametricity))),
    (sym::may_dangle,
    Normal,
    template!(Word),
    Gated(Stability::Unstable,
        sym::dropck_eyepatch,
        "may_dangle has unstable semantics and may be removed in the future",
        cfg_fn!(dropck_eyepatch))),
    (sym::unwind, Whitelisted, template!(List: "allowed|aborts"), Gated(Stability::Unstable,
                                sym::unwind_attributes,
                                "#[unwind] is experimental",
                                cfg_fn!(unwind_attributes))),
    (sym::used, Whitelisted, template!(Word), Ungated),

    // used in resolve
    (sym::prelude_import, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                        sym::prelude_import,
                                        "`#[prelude_import]` is for use by rustc only",
                                        cfg_fn!(prelude_import))),

    // FIXME: #14407 these are only looked at on-demand so we can't
    // guarantee they'll have already been checked
    (
        sym::rustc_deprecated,
        Whitelisted,
        template!(List: r#"since = "version", reason = "...""#),
        Ungated
    ),
    (sym::must_use, Whitelisted, template!(Word, NameValueStr: "reason"), Ungated),
    (
        sym::stable,
        Whitelisted,
        template!(List: r#"feature = "name", since = "version""#),
        Ungated
    ),
    (
        sym::unstable,
        Whitelisted,
        template!(List: r#"feature = "name", reason = "...", issue = "N""#),
        Ungated
    ),
    (sym::deprecated,
        Normal,
        template!(
            Word,
            List: r#"/*opt*/ since = "version", /*opt*/ note = "reason""#,
            NameValueStr: "reason"
        ),
        Ungated
    ),

    (sym::rustc_paren_sugar, Normal, template!(Word), Gated(Stability::Unstable,
                                        sym::unboxed_closures,
                                        "unboxed_closures are still evolving",
                                        cfg_fn!(unboxed_closures))),

    (sym::windows_subsystem, Whitelisted, template!(NameValueStr: "windows|console"), Ungated),

    (sym::proc_macro_attribute, Normal, template!(Word), Ungated),
    (sym::proc_macro, Normal, template!(Word), Ungated),

    (sym::rustc_proc_macro_decls, Normal, template!(Word), Gated(Stability::Unstable,
                                            sym::rustc_attrs,
                                            "used internally by rustc",
                                            cfg_fn!(rustc_attrs))),

    (sym::allow_fail, Normal, template!(Word), Gated(Stability::Unstable,
                                sym::allow_fail,
                                "allow_fail attribute is currently unstable",
                                cfg_fn!(allow_fail))),

    (sym::rustc_std_internal_symbol, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                    sym::rustc_attrs,
                                    "this is an internal attribute that will \
                                    never be stable",
                                    cfg_fn!(rustc_attrs))),

    // whitelists "identity-like" conversion methods to suggest on type mismatch
    (sym::rustc_conversion_suggestion, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                                    sym::rustc_attrs,
                                                    "this is an internal attribute that will \
                                                        never be stable",
                                                    cfg_fn!(rustc_attrs))),

    (
        sym::rustc_args_required_const,
        Whitelisted,
        template!(List: "N"),
        Gated(Stability::Unstable, sym::rustc_attrs, "never will be stable",
           cfg_fn!(rustc_attrs))
    ),
    // RFC 2070
    (sym::panic_handler, Normal, template!(Word), Ungated),

    (sym::alloc_error_handler, Normal, template!(Word), Gated(Stability::Unstable,
                        sym::alloc_error_handler,
                        "#[alloc_error_handler] is an unstable feature",
                        cfg_fn!(alloc_error_handler))),

    // RFC 2412
    (sym::optimize, Whitelisted, template!(List: "size|speed"), Gated(Stability::Unstable,
                            sym::optimize_attribute,
                            "#[optimize] attribute is an unstable feature",
                            cfg_fn!(optimize_attribute))),

    // Crate level attributes
    (sym::crate_name, CrateLevel, template!(NameValueStr: "name"), Ungated),
    (sym::crate_type, CrateLevel, template!(NameValueStr: "bin|lib|..."), Ungated),
    (sym::crate_id, CrateLevel, template!(NameValueStr: "ignored"), Ungated),
    (sym::feature, CrateLevel, template!(List: "name1, name1, ..."), Ungated),
    (sym::no_start, CrateLevel, template!(Word), Ungated),
    (sym::no_main, CrateLevel, template!(Word), Ungated),
    (sym::recursion_limit, CrateLevel, template!(NameValueStr: "N"), Ungated),
    (sym::type_length_limit, CrateLevel, template!(NameValueStr: "N"), Ungated),
    (sym::test_runner, CrateLevel, template!(List: "path"), Gated(Stability::Unstable,
                    sym::custom_test_frameworks,
                    EXPLAIN_CUSTOM_TEST_FRAMEWORKS,
                    cfg_fn!(custom_test_frameworks))),
];

pub type BuiltinAttribute = (Symbol, AttributeType, AttributeTemplate, AttributeGate);

lazy_static! {
    pub static ref BUILTIN_ATTRIBUTE_MAP: FxHashMap<Symbol, &'static BuiltinAttribute> = {
        let mut map = FxHashMap::default();
        for attr in BUILTIN_ATTRIBUTES.iter() {
            if map.insert(attr.0, attr).is_some() {
                panic!("duplicate builtin attribute `{}`", attr.0);
            }
        }
        map
    };
}

// cfg(...)'s that are feature gated
const GATED_CFGS: &[(Symbol, Symbol, fn(&Features) -> bool)] = &[
    // (name in cfg, feature, function to check if the feature is enabled)
    (sym::target_thread_local, sym::cfg_target_thread_local, cfg_fn!(cfg_target_thread_local)),
    (sym::target_has_atomic, sym::cfg_target_has_atomic, cfg_fn!(cfg_target_has_atomic)),
    (sym::rustdoc, sym::doc_cfg, cfg_fn!(doc_cfg)),
];

#[derive(Debug)]
pub struct GatedCfg {
    span: Span,
    index: usize,
}

impl GatedCfg {
    pub fn gate(cfg: &ast::MetaItem) -> Option<GatedCfg> {
        GATED_CFGS.iter()
                  .position(|info| cfg.check_name(info.0))
                  .map(|idx| {
                      GatedCfg {
                          span: cfg.span,
                          index: idx
                      }
                  })
    }

    pub fn check_and_emit(&self, sess: &ParseSess, features: &Features) {
        let (cfg, feature, has_feature) = GATED_CFGS[self.index];
        if !has_feature(features) && !self.span.allows_unstable(feature) {
            let explain = format!("`cfg({})` is experimental and subject to change", cfg);
            emit_feature_err(sess, feature, self.span, GateIssue::Language, &explain);
        }
    }
}

struct Context<'a> {
    features: &'a Features,
    parse_sess: &'a ParseSess,
    plugin_attributes: &'a [(Symbol, AttributeType)],
}

macro_rules! gate_feature_fn {
    ($cx: expr, $has_feature: expr, $span: expr, $name: expr, $explain: expr, $level: expr) => {{
        let (cx, has_feature, span,
             name, explain, level) = ($cx, $has_feature, $span, $name, $explain, $level);
        let has_feature: bool = has_feature(&$cx.features);
        debug!("gate_feature(feature = {:?}, span = {:?}); has? {}", name, span, has_feature);
        if !has_feature && !span.allows_unstable($name) {
            leveled_feature_err(cx.parse_sess, name, span, GateIssue::Language, explain, level)
                .emit();
        }
    }}
}

macro_rules! gate_feature {
    ($cx: expr, $feature: ident, $span: expr, $explain: expr) => {
        gate_feature_fn!($cx, |x:&Features| x.$feature, $span,
                         sym::$feature, $explain, GateStrength::Hard)
    };
    ($cx: expr, $feature: ident, $span: expr, $explain: expr, $level: expr) => {
        gate_feature_fn!($cx, |x:&Features| x.$feature, $span,
                         sym::$feature, $explain, $level)
    };
}

impl<'a> Context<'a> {
    fn check_attribute(
        &self,
        attr: &ast::Attribute,
        attr_info: Option<&BuiltinAttribute>,
        is_macro: bool
    ) {
        debug!("check_attribute(attr = {:?})", attr);
        if let Some(&(name, ty, _template, ref gateage)) = attr_info {
            if let Gated(_, name, desc, ref has_feature) = *gateage {
                if !attr.span.allows_unstable(name) {
                    gate_feature_fn!(
                        self, has_feature, attr.span, name, desc, GateStrength::Hard
                    );
                }
            } else if name == sym::doc {
                if let Some(content) = attr.meta_item_list() {
                    if content.iter().any(|c| c.check_name(sym::include)) {
                        gate_feature!(self, external_doc, attr.span,
                            "#[doc(include = \"...\")] is experimental"
                        );
                    }
                }
            }
            debug!("check_attribute: {:?} is builtin, {:?}, {:?}", attr.path, ty, gateage);
            return;
        }
        for &(n, ty) in self.plugin_attributes {
            if attr.path == n {
                // Plugins can't gate attributes, so we don't check for it
                // unlike the code above; we only use this loop to
                // short-circuit to avoid the checks below.
                debug!("check_attribute: {:?} is registered by a plugin, {:?}", attr.path, ty);
                return;
            }
        }
        if !attr::is_known(attr) {
            if attr.name_or_empty().as_str().starts_with("rustc_") {
                let msg = "unless otherwise specified, attributes with the prefix `rustc_` \
                           are reserved for internal compiler diagnostics";
                gate_feature!(self, rustc_attrs, attr.span, msg);
            } else if !is_macro {
                // Only run the custom attribute lint during regular feature gate
                // checking. Macro gating runs before the plugin attributes are
                // registered, so we skip this in that case.
                let msg = format!("The attribute `{}` is currently unknown to the compiler and \
                                   may have meaning added to it in the future", attr.path);
                gate_feature!(self, custom_attribute, attr.span, &msg);
            }
        }
    }
}

pub fn check_attribute(attr: &ast::Attribute, parse_sess: &ParseSess, features: &Features) {
    let cx = Context { features, parse_sess, plugin_attributes: &[] };
    cx.check_attribute(
        attr,
        attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name).map(|a| *a)),
        true
    );
}

fn find_lang_feature_issue(feature: Symbol) -> Option<u32> {
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

pub fn emit_feature_err(
    sess: &ParseSess,
    feature: Symbol,
    span: Span,
    issue: GateIssue,
    explain: &str,
) {
    feature_err(sess, feature, span, issue, explain).emit();
}

pub fn feature_err<'a, S: Into<MultiSpan>>(
    sess: &'a ParseSess,
    feature: Symbol,
    span: S,
    issue: GateIssue,
    explain: &str,
) -> DiagnosticBuilder<'a> {
    leveled_feature_err(sess, feature, span, issue, explain, GateStrength::Hard)
}

fn leveled_feature_err<'a, S: Into<MultiSpan>>(
    sess: &'a ParseSess,
    feature: Symbol,
    span: S,
    issue: GateIssue,
    explain: &str,
    level: GateStrength,
) -> DiagnosticBuilder<'a> {
    let diag = &sess.span_diagnostic;

    let issue = match issue {
        GateIssue::Language => find_lang_feature_issue(feature),
        GateIssue::Library(lib) => lib,
    };

    let mut err = match level {
        GateStrength::Hard => {
            diag.struct_span_err_with_code(span, explain, stringify_error_code!(E0658))
        }
        GateStrength::Soft => diag.struct_span_warn(span, explain),
    };

    match issue {
        None | Some(0) => {}  // We still accept `0` as a stand-in for backwards compatibility
        Some(n) => {
            err.note(&format!(
                "for more information, see https://github.com/rust-lang/rust/issues/{}",
                n,
            ));
        }
    }

    // #23973: do not suggest `#![feature(...)]` if we are in beta/stable
    if sess.unstable_features.is_nightly_build() {
        err.help(&format!("add #![feature({})] to the crate attributes to enable", feature));
    }

    // If we're on stable and only emitting a "soft" warning, add a note to
    // clarify that the feature isn't "on" (rather than being on but
    // warning-worthy).
    if !sess.unstable_features.is_nightly_build() && level == GateStrength::Soft {
        err.help("a nightly build of the compiler is required to enable this feature");
    }

    err

}

const EXPLAIN_BOX_SYNTAX: &str =
    "box expression syntax is experimental; you can call `Box::new` instead";

pub const EXPLAIN_STMT_ATTR_SYNTAX: &str =
    "attributes on expressions are experimental";

pub const EXPLAIN_ASM: &str =
    "inline assembly is not stable enough for use and is subject to change";

pub const EXPLAIN_GLOBAL_ASM: &str =
    "`global_asm!` is not stable enough for use and is subject to change";

pub const EXPLAIN_CUSTOM_TEST_FRAMEWORKS: &str =
    "custom test frameworks are an unstable feature";

pub const EXPLAIN_LOG_SYNTAX: &str =
    "`log_syntax!` is not stable enough for use and is subject to change";

pub const EXPLAIN_CONCAT_IDENTS: &str =
    "`concat_idents` is not stable enough for use and is subject to change";

pub const EXPLAIN_FORMAT_ARGS_NL: &str =
    "`format_args_nl` is only for internal language use and is subject to change";

pub const EXPLAIN_TRACE_MACROS: &str =
    "`trace_macros` is not stable enough for use and is subject to change";
pub const EXPLAIN_ALLOW_INTERNAL_UNSTABLE: &str =
    "allow_internal_unstable side-steps feature gating and stability checks";
pub const EXPLAIN_ALLOW_INTERNAL_UNSAFE: &str =
    "allow_internal_unsafe side-steps the unsafe_code lint";

pub const EXPLAIN_UNSIZED_TUPLE_COERCION: &str =
    "unsized tuple coercion is not stable enough for use and is subject to change";

struct PostExpansionVisitor<'a> {
    context: &'a Context<'a>,
    builtin_attributes: &'static FxHashMap<Symbol, &'static BuiltinAttribute>,
}

macro_rules! gate_feature_post {
    ($cx: expr, $feature: ident, $span: expr, $explain: expr) => {{
        let (cx, span) = ($cx, $span);
        if !span.allows_unstable(sym::$feature) {
            gate_feature!(cx.context, $feature, span, $explain)
        }
    }};
    ($cx: expr, $feature: ident, $span: expr, $explain: expr, $level: expr) => {{
        let (cx, span) = ($cx, $span);
        if !span.allows_unstable(sym::$feature) {
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

    fn check_builtin_attribute(&mut self, attr: &ast::Attribute, name: Symbol,
                               template: AttributeTemplate) {
        // Some special attributes like `cfg` must be checked
        // before the generic check, so we skip them here.
        let should_skip = |name| name == sym::cfg;
        // Some of previously accepted forms were used in practice,
        // report them as warnings for now.
        let should_warn = |name| name == sym::doc || name == sym::ignore ||
                                 name == sym::inline || name == sym::link;

        match attr.parse_meta(self.context.parse_sess) {
            Ok(meta) => if !should_skip(name) && !template.compatible(&meta.node) {
                let error_msg = format!("malformed `{}` attribute input", name);
                let mut msg = "attribute must be of the form ".to_owned();
                let mut suggestions = vec![];
                let mut first = true;
                if template.word {
                    first = false;
                    let code = format!("#[{}]", name);
                    msg.push_str(&format!("`{}`", &code));
                    suggestions.push(code);
                }
                if let Some(descr) = template.list {
                    if !first {
                        msg.push_str(" or ");
                    }
                    first = false;
                    let code = format!("#[{}({})]", name, descr);
                    msg.push_str(&format!("`{}`", &code));
                    suggestions.push(code);
                }
                if let Some(descr) = template.name_value_str {
                    if !first {
                        msg.push_str(" or ");
                    }
                    let code = format!("#[{} = \"{}\"]", name, descr);
                    msg.push_str(&format!("`{}`", &code));
                    suggestions.push(code);
                }
                if should_warn(name) {
                    self.context.parse_sess.buffer_lint(
                        BufferedEarlyLintId::IllFormedAttributeInput,
                        meta.span,
                        ast::CRATE_NODE_ID,
                        &msg,
                    );
                } else {
                    self.context.parse_sess.span_diagnostic.struct_span_err(meta.span, &error_msg)
                        .span_suggestions(
                            meta.span,
                            if suggestions.len() == 1 {
                                "must be of the form"
                            } else {
                                "the following are the possible correct uses"
                            },
                            suggestions.into_iter(),
                            Applicability::HasPlaceholders,
                        ).emit();
                }
            }
            Err(mut err) => err.emit(),
        }
    }
}

impl<'a> Visitor<'a> for PostExpansionVisitor<'a> {
    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        let attr_info = attr.ident().and_then(|ident| {
            self.builtin_attributes.get(&ident.name).map(|a| *a)
        });

        // Check for gated attributes.
        self.context.check_attribute(attr, attr_info, false);

        if attr.check_name(sym::doc) {
            if let Some(content) = attr.meta_item_list() {
                if content.len() == 1 && content[0].check_name(sym::cfg) {
                    gate_feature_post!(&self, doc_cfg, attr.span,
                        "#[doc(cfg(...))] is experimental"
                    );
                } else if content.iter().any(|c| c.check_name(sym::masked)) {
                    gate_feature_post!(&self, doc_masked, attr.span,
                        "#[doc(masked)] is experimental"
                    );
                } else if content.iter().any(|c| c.check_name(sym::spotlight)) {
                    gate_feature_post!(&self, doc_spotlight, attr.span,
                        "#[doc(spotlight)] is experimental"
                    );
                } else if content.iter().any(|c| c.check_name(sym::alias)) {
                    gate_feature_post!(&self, doc_alias, attr.span,
                        "#[doc(alias = \"...\")] is experimental"
                    );
                } else if content.iter().any(|c| c.check_name(sym::keyword)) {
                    gate_feature_post!(&self, doc_keyword, attr.span,
                        "#[doc(keyword = \"...\")] is experimental"
                    );
                }
            }
        }

        match attr_info {
            // `rustc_dummy` doesn't have any restrictions specific to built-in attributes.
            Some(&(name, _, template, _)) if name != sym::rustc_dummy =>
                self.check_builtin_attribute(attr, name, template),
            _ => if let Some(TokenTree::Token(token)) = attr.tokens.trees().next() {
                if token == token::Eq {
                    // All key-value attributes are restricted to meta-item syntax.
                    attr.parse_meta(self.context.parse_sess).map_err(|mut err| err.emit()).ok();
                }
            }
        }
    }

    fn visit_name(&mut self, sp: Span, name: ast::Name) {
        if !name.as_str().is_ascii() {
            gate_feature_post!(
                &self,
                non_ascii_idents,
                self.context.parse_sess.source_map().def_span(sp),
                "non-ascii idents are not fully supported"
            );
        }
    }

    fn visit_item(&mut self, i: &'a ast::Item) {
        match i.node {
            ast::ItemKind::ForeignMod(ref foreign_module) => {
                self.check_abi(foreign_module.abi, i.span);
            }

            ast::ItemKind::Fn(..) => {
                if attr::contains_name(&i.attrs[..], sym::plugin_registrar) {
                    gate_feature_post!(&self, plugin_registrar, i.span,
                                       "compiler plugins are experimental and possibly buggy");
                }
                if attr::contains_name(&i.attrs[..], sym::start) {
                    gate_feature_post!(&self, start, i.span,
                                      "a #[start] function is an experimental \
                                       feature whose signature may change \
                                       over time");
                }
                if attr::contains_name(&i.attrs[..], sym::main) {
                    gate_feature_post!(&self, main, i.span,
                                       "declaration of a nonstandard #[main] \
                                        function may change over time, for now \
                                        a top-level `fn main()` is required");
                }
            }

            ast::ItemKind::Struct(..) => {
                for attr in attr::filter_by_name(&i.attrs[..], sym::repr) {
                    for item in attr.meta_item_list().unwrap_or_else(Vec::new) {
                        if item.check_name(sym::simd) {
                            gate_feature_post!(&self, repr_simd, attr.span,
                                               "SIMD types are experimental and possibly buggy");
                        }
                    }
                }
            }

            ast::ItemKind::Enum(ast::EnumDef{ref variants, ..}, ..) => {
                for variant in variants {
                    match (&variant.node.data, &variant.node.disr_expr) {
                        (ast::VariantData::Unit(..), _) => {},
                        (_, Some(disr_expr)) =>
                            gate_feature_post!(
                                &self,
                                arbitrary_enum_discriminant,
                                disr_expr.value.span,
                                "discriminants on non-unit variants are experimental"),
                        _ => {},
                    }
                }

                let has_feature = self.context.features.arbitrary_enum_discriminant;
                if !has_feature && !i.span.allows_unstable(sym::arbitrary_enum_discriminant) {
                    Parser::maybe_report_invalid_custom_discriminants(
                        self.context.parse_sess,
                        &variants,
                    );
                }
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

            ast::ItemKind::TraitAlias(..) => {
                gate_feature_post!(
                    &self,
                    trait_alias,
                    i.span,
                    "trait aliases are experimental"
                );
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
                let link_name = attr::first_attr_value_str_by_name(&i.attrs, sym::link_name);
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
                // Do nothing.
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
                // To avoid noise about type ascription in common syntax errors, only emit if it
                // is the *only* error.
                if self.context.parse_sess.span_diagnostic.err_count() == 0 {
                    gate_feature_post!(&self, type_ascription, e.span,
                                       "type ascription is experimental");
                }
            }
            ast::ExprKind::Yield(..) => {
                gate_feature_post!(&self, generators,
                                  e.span,
                                  "yield syntax is experimental");
            }
            ast::ExprKind::TryBlock(_) => {
                gate_feature_post!(&self, try_blocks, e.span, "`try` expression is experimental");
            }
            ast::ExprKind::Block(_, opt_label) => {
                if let Some(label) = opt_label {
                    gate_feature_post!(&self, label_break_value, label.ident.span,
                                    "labels on blocks are unstable");
                }
            }
            ast::ExprKind::Async(..) => {
                gate_feature_post!(&self, async_await, e.span, "async blocks are unstable");
            }
            ast::ExprKind::Await(origin, _) => {
                match origin {
                    ast::AwaitOrigin::FieldLike =>
                        gate_feature_post!(&self, async_await, e.span, "async/await is unstable"),
                    ast::AwaitOrigin::MacroLike =>
                        gate_feature_post!(
                            &self,
                            await_macro,
                            e.span,
                            "`await!(<expr>)` macro syntax is unstable, and will soon be removed \
                            in favor of `<expr>.await` syntax."
                        ),
                }
            }
            _ => {}
        }
        visit::walk_expr(self, e)
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
            _ => {}
        }
        visit::walk_pat(self, pattern)
    }

    fn visit_fn(&mut self,
                fn_kind: FnKind<'a>,
                fn_decl: &'a ast::FnDecl,
                span: Span,
                _node_id: NodeId) {
        if let Some(header) = fn_kind.header() {
            // Check for const fn and async fn declarations.
            if header.asyncness.node.is_async() {
                gate_feature_post!(&self, async_await, span, "async fn is unstable");
            }

            // Stability of const fn methods are covered in
            // `visit_trait_item` and `visit_impl_item` below; this is
            // because default methods don't pass through this point.
            self.check_abi(header.abi, span);
        }

        if fn_decl.c_variadic {
            gate_feature_post!(&self, c_variadic, span, "C-variadic functions are unstable");
        }

        visit::walk_fn(self, fn_kind, fn_decl, span)
    }

    fn visit_generic_param(&mut self, param: &'a GenericParam) {
        match param.kind {
            GenericParamKind::Const { .. } =>
                gate_feature_post!(&self, const_generics, param.ident.span,
                    "const generics are unstable"),
            _ => {}
        }
        visit::walk_generic_param(self, param)
    }

    fn visit_assoc_ty_constraint(&mut self, constraint: &'a AssocTyConstraint) {
        match constraint.kind {
            AssocTyConstraintKind::Bound { .. } =>
                gate_feature_post!(&self, associated_type_bounds, constraint.span,
                    "associated type bounds are unstable"),
            _ => {}
        }
        visit::walk_assoc_ty_constraint(self, constraint)
    }

    fn visit_trait_item(&mut self, ti: &'a ast::TraitItem) {
        match ti.node {
            ast::TraitItemKind::Method(ref sig, ref block) => {
                if block.is_none() {
                    self.check_abi(sig.header.abi, ti.span);
                }
                if sig.header.asyncness.node.is_async() {
                    gate_feature_post!(&self, async_await, ti.span, "async fn is unstable");
                }
                if sig.decl.c_variadic {
                    gate_feature_post!(&self, c_variadic, ti.span,
                                       "C-variadic functions are unstable");
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
        visit::walk_trait_item(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'a ast::ImplItem) {
        if ii.defaultness == ast::Defaultness::Default {
            gate_feature_post!(&self, specialization,
                              ii.span,
                              "specialization is unstable");
        }

        match ii.node {
            ast::ImplItemKind::Method(..) => {}
            ast::ImplItemKind::Existential(..) => {
                gate_feature_post!(
                    &self,
                    existential_type,
                    ii.span,
                    "existential types are unstable"
                );
            }
            ast::ImplItemKind::Type(_) => {
                if !ii.generics.params.is_empty() {
                    gate_feature_post!(&self, generic_associated_types, ii.span,
                                       "generic associated types are unstable");
                }
                if !ii.generics.where_clause.predicates.is_empty() {
                    gate_feature_post!(&self, generic_associated_types, ii.span,
                                       "where clauses on associated types are unstable");
                }
            }
            _ => {}
        }
        visit::walk_impl_item(self, ii)
    }

    fn visit_vis(&mut self, vis: &'a ast::Visibility) {
        if let ast::VisibilityKind::Crate(ast::CrateSugar::JustCrate) = vis.node {
            gate_feature_post!(&self, crate_visibility_modifier, vis.span,
                               "`crate` visibility modifier is experimental");
        }
        visit::walk_vis(self, vis)
    }
}

pub fn get_features(span_handler: &Handler, krate_attrs: &[ast::Attribute],
                    crate_edition: Edition, allow_features: &Option<Vec<String>>) -> Features {
    fn feature_removed(span_handler: &Handler, span: Span, reason: Option<&str>) {
        let mut err = struct_span_err!(span_handler, span, E0557, "feature has been removed");
        if let Some(reason) = reason {
            err.span_note(span, reason);
        } else {
            err.span_label(span, "feature has been removed");
        }
        err.emit();
    }

    let mut features = Features::new();
    let mut edition_enabled_features = FxHashMap::default();

    for &edition in ALL_EDITIONS {
        if edition <= crate_edition {
            // The `crate_edition` implies its respective umbrella feature-gate
            // (i.e., `#![feature(rust_20XX_preview)]` isn't needed on edition 20XX).
            edition_enabled_features.insert(edition.feature_name(), edition);
        }
    }

    for &(name, .., f_edition, set) in ACTIVE_FEATURES {
        if let Some(f_edition) = f_edition {
            if f_edition <= crate_edition {
                set(&mut features, DUMMY_SP);
                edition_enabled_features.insert(name, crate_edition);
            }
        }
    }

    // Process the edition umbrella feature-gates first, to ensure
    // `edition_enabled_features` is completed before it's queried.
    for attr in krate_attrs {
        if !attr.check_name(sym::feature) {
            continue
        }

        let list = match attr.meta_item_list() {
            Some(list) => list,
            None => continue,
        };

        for mi in list {
            if !mi.is_word() {
                continue;
            }

            let name = mi.name_or_empty();
            if INCOMPLETE_FEATURES.iter().any(|f| name == *f) {
                span_handler.struct_span_warn(
                    mi.span(),
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
                            edition_enabled_features.insert(name, *edition);
                        }
                    }
                }
            }
        }
    }

    for attr in krate_attrs {
        if !attr.check_name(sym::feature) {
            continue
        }

        let list = match attr.meta_item_list() {
            Some(list) => list,
            None => continue,
        };

        let bad_input = |span| {
            struct_span_err!(span_handler, span, E0556, "malformed `feature` attribute input")
        };

        for mi in list {
            let name = match mi.ident() {
                Some(ident) if mi.is_word() => ident.name,
                Some(ident) => {
                    bad_input(mi.span()).span_suggestion(
                        mi.span(),
                        "expected just one word",
                        format!("{}", ident.name),
                        Applicability::MaybeIncorrect,
                    ).emit();
                    continue
                }
                None => {
                    bad_input(mi.span()).span_label(mi.span(), "expected just one word").emit();
                    continue
                }
            };

            if let Some(edition) = edition_enabled_features.get(&name) {
                struct_span_warn!(
                    span_handler,
                    mi.span(),
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

            let removed = REMOVED_FEATURES.iter().find(|f| name == f.0);
            let stable_removed = STABLE_REMOVED_FEATURES.iter().find(|f| name == f.0);
            if let Some((.., reason)) = removed.or(stable_removed) {
                feature_removed(span_handler, mi.span(), *reason);
                continue;
            }

            if let Some((_, since, ..)) = ACCEPTED_FEATURES.iter().find(|f| name == f.0) {
                let since = Some(Symbol::intern(since));
                features.declared_lang_features.push((name, mi.span(), since));
                continue;
            }

            if let Some(allowed) = allow_features.as_ref() {
                if allowed.iter().find(|f| *f == name.as_str()).is_none() {
                    span_err!(span_handler, mi.span(), E0725,
                              "the feature `{}` is not in the list of allowed features",
                              name);
                    continue;
                }
            }

            if let Some((.., set)) = ACTIVE_FEATURES.iter().find(|f| name == f.0) {
                set(&mut features, mi.span());
                features.declared_lang_features.push((name, mi.span(), None));
                continue;
            }

            features.declared_lib_features.push((name, mi.span()));
        }
    }

    features
}

fn for_each_in_lock<T>(vec: &Lock<Vec<T>>, f: impl Fn(&T)) {
    vec.borrow().iter().for_each(f);
}

pub fn check_crate(krate: &ast::Crate,
                   sess: &ParseSess,
                   features: &Features,
                   plugin_attributes: &[(Symbol, AttributeType)],
                   unstable: UnstableFeatures) {
    maybe_stage_features(&sess.span_diagnostic, krate, unstable);
    let ctx = Context {
        features,
        parse_sess: sess,
        plugin_attributes,
    };

    for_each_in_lock(&sess.param_attr_spans, |span| gate_feature!(
        &ctx,
        param_attrs,
        *span,
        "attributes on function parameters are unstable"
    ));

    for_each_in_lock(&sess.let_chains_spans, |span| gate_feature!(
        &ctx,
        let_chains,
        *span,
        "`let` expressions in this position are experimental"
    ));

    for_each_in_lock(&sess.async_closure_spans, |span| gate_feature!(
        &ctx,
        async_closure,
        *span,
        "async closures are unstable"
    ));

    let visitor = &mut PostExpansionVisitor {
        context: &ctx,
        builtin_attributes: &*BUILTIN_ATTRIBUTE_MAP,
    };
    visit::walk_crate(visitor, krate);
}

#[derive(Clone, Copy, Hash)]
pub enum UnstableFeatures {
    /// Hard errors for unstable features are active, as on beta/stable channels.
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
        // Whether this is a feature-staged build, i.e., on the beta or stable channel
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
            if attr.check_name(sym::feature) {
                let release_channel = option_env!("CFG_RELEASE_CHANNEL").unwrap_or("(unknown)");
                span_err!(span_handler, attr.span, E0554,
                          "#![feature] may not be used on the {} release channel",
                          release_channel);
            }
        }
    }
}
