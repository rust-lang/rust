//! List of the active feature gates.

use super::{to_nonzero, Feature, State};

use rustc_data_structures::fx::FxHashSet;
use rustc_span::edition::Edition;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;

macro_rules! set {
    ($field: ident) => {{
        fn f(features: &mut Features, _: Span) {
            features.$field = true;
        }
        f as fn(&mut Features, Span)
    }};
}

macro_rules! declare_features {
    (__status_to_bool active) => {
        false
    };
    (__status_to_bool incomplete) => {
        true
    };
    ($(
        $(#[doc = $doc:tt])* ($status:ident, $feature:ident, $ver:expr, $issue:expr, $edition:expr),
    )+) => {
        /// Represents active features that are currently being implemented or
        /// currently being considered for addition/removal.
        pub const ACTIVE_FEATURES:
            &[Feature] =
            &[$(
                // (sym::$feature, $ver, $issue, $edition, set!($feature))
                Feature {
                    state: State::Active { set: set!($feature) },
                    name: sym::$feature,
                    since: $ver,
                    issue: to_nonzero($issue),
                    edition: $edition,
                }
            ),+];

        /// A set of features to be used by later passes.
        #[derive(Clone, Default, Debug)]
        pub struct Features {
            /// `#![feature]` attrs for language features, for error reporting.
            pub declared_lang_features: Vec<(Symbol, Span, Option<Symbol>)>,
            /// `#![feature]` attrs for non-language (library) features.
            pub declared_lib_features: Vec<(Symbol, Span)>,
            /// Features enabled for this crate.
            pub active_features: FxHashSet<Symbol>,
            $(
                $(#[doc = $doc])*
                pub $feature: bool
            ),+
        }

        impl Features {
            pub fn walk_feature_fields(&self, mut f: impl FnMut(&str, bool)) {
                $(f(stringify!($feature), self.$feature);)+
            }

            /// Is the given feature active?
            pub fn active(&self, feature: Symbol) -> bool {
                self.active_features.contains(&feature)
            }

            /// Is the given feature enabled?
            ///
            /// Panics if the symbol doesn't correspond to a declared feature.
            pub fn enabled(&self, feature: Symbol) -> bool {
                match feature {
                    $( sym::$feature => self.$feature, )*

                    _ => panic!("`{}` was not listed in `declare_features`", feature),
                }
            }

            /// Some features are known to be incomplete and using them is likely to have
            /// unanticipated results, such as compiler crashes. We warn the user about these
            /// to alert them.
            pub fn incomplete(&self, feature: Symbol) -> bool {
                match feature {
                    $(
                        sym::$feature => declare_features!(__status_to_bool $status),
                    )*
                    // accepted and removed features aren't in this file but are never incomplete
                    _ if self.declared_lang_features.iter().any(|f| f.0 == feature) => false,
                    _ if self.declared_lib_features.iter().any(|f| f.0 == feature) => false,
                    _ => panic!("`{}` was not listed in `declare_features`", feature),
                }
            }
        }
    };
}

impl Feature {
    /// Sets this feature in `Features`. Panics if called on a non-active feature.
    pub fn set(&self, features: &mut Features, span: Span) {
        match self.state {
            State::Active { set } => set(features, span),
            _ => panic!("called `set` on feature `{}` which is not `active`", self.name),
        }
    }
}

// See https://rustc-dev-guide.rust-lang.org/feature-gates.html#feature-gates for more
// documentation about handling feature gates.
//
// If you change this, please modify `src/doc/unstable-book` as well.
//
// Don't ever remove anything from this list; move them to `accepted.rs` if
// accepted or `removed.rs` if removed.
//
// The version numbers here correspond to the version in which the current status
// was set. This is most important for knowing when a particular feature became
// stable (active).
//
// Note that the features are grouped into internal/user-facing and then
// sorted by version inside those groups. This is enforced with tidy.
//
// N.B., `tools/tidy/src/features.rs` parses this information directly out of the
// source, so take care when modifying it.

#[rustfmt::skip]
declare_features! (
    // -------------------------------------------------------------------------
    // feature-group-start: internal feature gates (no tracking issue)
    // -------------------------------------------------------------------------
    // no-tracking-issue-start

    /// Allows using the `thiscall` ABI.
    (active, abi_thiscall, "1.19.0", None, None),
    /// Allows using the `unadjusted` ABI; perma-unstable.
    (active, abi_unadjusted, "1.16.0", None, None),
    /// Allows using the `vectorcall` ABI.
    (active, abi_vectorcall, "1.7.0", None, None),
    /// Allows using `#![needs_allocator]`, an implementation detail of `#[global_allocator]`.
    (active, allocator_internals, "1.20.0", None, None),
    /// Allows using `#[allow_internal_unsafe]`. This is an
    /// attribute on `macro_rules!` and can't use the attribute handling
    /// below (it has to be checked before expansion possibly makes
    /// macros disappear).
    (active, allow_internal_unsafe, "1.0.0", None, None),
    /// Allows using `#[allow_internal_unstable]`. This is an
    /// attribute on `macro_rules!` and can't use the attribute handling
    /// below (it has to be checked before expansion possibly makes
    /// macros disappear).
    (active, allow_internal_unstable, "1.0.0", None, None),
    /// Allows using anonymous lifetimes in argument-position impl-trait.
    (active, anonymous_lifetime_in_impl_trait, "1.63.0", None, None),
    /// Allows identifying the `compiler_builtins` crate.
    (active, compiler_builtins, "1.13.0", None, None),
    /// Outputs useful `assert!` messages
    (active, generic_assert, "1.63.0", None, None),
    /// Allows using the `rust-intrinsic`'s "ABI".
    (active, intrinsics, "1.0.0", None, None),
    /// Allows using `#[lang = ".."]` attribute for linking items to special compiler logic.
    (active, lang_items, "1.0.0", None, None),
    /// Allows using `#[omit_gdb_pretty_printer_section]`.
    (active, omit_gdb_pretty_printer_section, "1.5.0", None, None),
    /// Allows using `#[prelude_import]` on glob `use` items.
    (active, prelude_import, "1.2.0", None, None),
    /// Used to identify crates that contain the profiler runtime.
    (active, profiler_runtime, "1.18.0", None, None),
    /// Allows using `rustc_*` attributes (RFC 572).
    (active, rustc_attrs, "1.0.0", None, None),
    /// Allows using the `#[stable]` and `#[unstable]` attributes.
    (active, staged_api, "1.0.0", None, None),
    /// Added for testing E0705; perma-unstable.
    (active, test_2018_feature, "1.31.0", None, Some(Edition::Edition2018)),
    /// Added for testing unstable lints; perma-unstable.
    (active, test_unstable_lint, "1.60.0", None, None),
    /// Allows non-`unsafe` —and thus, unsound— access to `Pin` constructions.
    /// Marked `incomplete` since perma-unstable and unsound.
    (incomplete, unsafe_pin_internals, "1.60.0", None, None),
    /// Use for stable + negative coherence and strict coherence depending on trait's
    /// rustc_strict_coherence value.
    (active, with_negative_coherence, "1.60.0", None, None),
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!
    // Features are listed in alphabetical order. Tidy will fail if you don't keep it this way.
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!

    // no-tracking-issue-end
    // -------------------------------------------------------------------------
    // feature-group-end: internal feature gates (no tracking issue)
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // feature-group-start: internal feature gates
    // -------------------------------------------------------------------------

    /// Allows features specific to auto traits.
    /// Renamed from `optin_builtin_traits`.
    (active, auto_traits, "1.50.0", Some(13231), None),
    /// Allows using `box` in patterns (RFC 469).
    (active, box_patterns, "1.0.0", Some(29641), None),
    /// Allows using the `box $expr` syntax.
    (active, box_syntax, "1.0.0", Some(49733), None),
    /// Allows `#[doc(notable_trait)]`.
    /// Renamed from `doc_spotlight`.
    (active, doc_notable_trait, "1.52.0", Some(45040), None),
    /// Allows using the `may_dangle` attribute (RFC 1327).
    (active, dropck_eyepatch, "1.10.0", Some(34761), None),
    /// Allows using the `#[fundamental]` attribute.
    (active, fundamental, "1.0.0", Some(29635), None),
    /// Allows using `#[link_name="llvm.*"]`.
    (active, link_llvm_intrinsics, "1.0.0", Some(29602), None),
    /// Allows using the `#[linkage = ".."]` attribute.
    (active, linkage, "1.0.0", Some(29603), None),
    /// Allows declaring with `#![needs_panic_runtime]` that a panic runtime is needed.
    (active, needs_panic_runtime, "1.10.0", Some(32837), None),
    /// Allows using the `#![panic_runtime]` attribute.
    (active, panic_runtime, "1.10.0", Some(32837), None),
    /// Allows using `#[rustc_allow_const_fn_unstable]`.
    /// This is an attribute on `const fn` for the same
    /// purpose as `#[allow_internal_unstable]`.
    (active, rustc_allow_const_fn_unstable, "1.49.0", Some(69399), None),
    /// Allows using compiler's own crates.
    (active, rustc_private, "1.0.0", Some(27812), None),
    /// Allows using internal rustdoc features like `doc(primitive)` or `doc(keyword)`.
    (active, rustdoc_internals, "1.58.0", Some(90418), None),
    /// Allows using `#[start]` on a function indicating that it is the program entrypoint.
    (active, start, "1.0.0", Some(29633), None),
    /// Allows using `#[structural_match]` which indicates that a type is structurally matchable.
    /// FIXME: Subsumed by trait `StructuralPartialEq`, cannot move to removed until a library
    /// feature with the same name exists.
    (active, structural_match, "1.8.0", Some(31434), None),
    /// Allows using the `rust-call` ABI.
    (active, unboxed_closures, "1.0.0", Some(29625), None),
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!
    // Features are listed in alphabetical order. Tidy will fail if you don't keep it this way.
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!

    // -------------------------------------------------------------------------
    // feature-group-end: internal feature gates
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // feature-group-start: actual feature gates (target features)
    // -------------------------------------------------------------------------

    // FIXME: Document these and merge with the list below.

    // Unstable `#[target_feature]` directives.
    (active, aarch64_ver_target_feature, "1.27.0", Some(44839), None),
    (active, arm_target_feature, "1.27.0", Some(44839), None),
    (active, avx512_target_feature, "1.27.0", Some(44839), None),
    (active, bpf_target_feature, "1.54.0", Some(44839), None),
    (active, cmpxchg16b_target_feature, "1.32.0", Some(44839), None),
    (active, ermsb_target_feature, "1.49.0", Some(44839), None),
    (active, f16c_target_feature, "1.36.0", Some(44839), None),
    (active, hexagon_target_feature, "1.27.0", Some(44839), None),
    (active, mips_target_feature, "1.27.0", Some(44839), None),
    (active, movbe_target_feature, "1.34.0", Some(44839), None),
    (active, powerpc_target_feature, "1.27.0", Some(44839), None),
    (active, riscv_target_feature, "1.45.0", Some(44839), None),
    (active, rtm_target_feature, "1.35.0", Some(44839), None),
    (active, sse4a_target_feature, "1.27.0", Some(44839), None),
    (active, tbm_target_feature, "1.27.0", Some(44839), None),
    (active, wasm_target_feature, "1.30.0", Some(44839), None),
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!
    // Features are listed in alphabetical order. Tidy will fail if you don't keep it this way.
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!

    // -------------------------------------------------------------------------
    // feature-group-end: actual feature gates (target features)
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // feature-group-start: actual feature gates
    // -------------------------------------------------------------------------

    /// Allows using the `amdgpu-kernel` ABI.
    (active, abi_amdgpu_kernel, "1.29.0", Some(51575), None),
    /// Allows `extern "avr-interrupt" fn()` and `extern "avr-non-blocking-interrupt" fn()`.
    (active, abi_avr_interrupt, "1.45.0", Some(69664), None),
    /// Allows `extern "C-cmse-nonsecure-call" fn()`.
    (active, abi_c_cmse_nonsecure_call, "1.51.0", Some(81391), None),
    /// Allows using the `efiapi` ABI.
    (active, abi_efiapi, "1.40.0", Some(65815), None),
    /// Allows `extern "msp430-interrupt" fn()`.
    (active, abi_msp430_interrupt, "1.16.0", Some(38487), None),
    /// Allows `extern "ptx-*" fn()`.
    (active, abi_ptx, "1.15.0", Some(38788), None),
    /// Allows `extern "x86-interrupt" fn()`.
    (active, abi_x86_interrupt, "1.17.0", Some(40180), None),
    /// Allows additional const parameter types, such as `&'static str` or user defined types
    (incomplete, adt_const_params, "1.56.0", Some(95174), None),
    /// Allows defining an `#[alloc_error_handler]`.
    (active, alloc_error_handler, "1.29.0", Some(51540), None),
    /// Allows explicit discriminants on non-unit enum variants.
    (active, arbitrary_enum_discriminant, "1.37.0", Some(60553), None),
    /// Allows trait methods with arbitrary self types.
    (active, arbitrary_self_types, "1.23.0", Some(44874), None),
    /// Allows using `const` operands in inline assembly.
    (active, asm_const, "1.58.0", Some(93332), None),
    /// Enables experimental inline assembly support for additional architectures.
    (active, asm_experimental_arch, "1.58.0", Some(93335), None),
    /// Allows using `sym` operands in inline assembly.
    (active, asm_sym, "1.58.0", Some(93333), None),
    /// Allows the `may_unwind` option in inline assembly.
    (active, asm_unwind, "1.58.0", Some(93334), None),
    /// Allows users to enforce equality of associated constants `TraitImpl<AssocConst=3>`.
    (active, associated_const_equality, "1.58.0", Some(92827), None),
    /// Allows the user of associated type bounds.
    (active, associated_type_bounds, "1.34.0", Some(52662), None),
    /// Allows associated type defaults.
    (active, associated_type_defaults, "1.2.0", Some(29661), None),
    /// Allows `async || body` closures.
    (active, async_closure, "1.37.0", Some(62290), None),
    /// Allows `extern "C-unwind" fn` to enable unwinding across ABI boundaries.
    (active, c_unwind, "1.52.0", Some(74990), None),
    /// Allows using C-variadics.
    (active, c_variadic, "1.34.0", Some(44930), None),
    /// Allows capturing disjoint fields in a closure/generator (RFC 2229).
    (incomplete, capture_disjoint_fields, "1.49.0", Some(53488), None),
    /// Allows the use of `#[cfg(sanitize = "option")]`; set when -Zsanitizer is used.
    (active, cfg_sanitize, "1.41.0", Some(39699), None),
    /// Allows `cfg(target_abi = "...")`.
    (active, cfg_target_abi, "1.55.0", Some(80970), None),
    /// Allows `cfg(target(abi = "..."))`.
    (active, cfg_target_compact, "1.63.0", Some(96901), None),
    /// Allows `cfg(target_has_atomic_load_store = "...")`.
    (active, cfg_target_has_atomic, "1.60.0", Some(94039), None),
    /// Allows `cfg(target_has_atomic_equal_alignment = "...")`.
    (active, cfg_target_has_atomic_equal_alignment, "1.60.0", Some(93822), None),
    /// Allows `cfg(target_thread_local)`.
    (active, cfg_target_thread_local, "1.7.0", Some(29594), None),
    /// Allow conditional compilation depending on rust version
    (active, cfg_version, "1.45.0", Some(64796), None),
    /// Allows `for<...>` on closures and generators.
    (active, closure_lifetime_binder, "1.64.0", Some(97362), None),
    /// Allows `#[track_caller]` on closures and generators.
    (active, closure_track_caller, "1.57.0", Some(87417), None),
    /// Allows to use the `#[cmse_nonsecure_entry]` attribute.
    (active, cmse_nonsecure_entry, "1.48.0", Some(75835), None),
    /// Allows `async {}` expressions in const contexts.
    (active, const_async_blocks, "1.53.0", Some(85368), None),
    // Allows limiting the evaluation steps of const expressions
    (active, const_eval_limit, "1.43.0", Some(67217), None),
    /// Allows the definition of `const extern fn` and `const unsafe extern fn`.
    (active, const_extern_fn, "1.40.0", Some(64926), None),
    /// Allows basic arithmetic on floating point types in a `const fn`.
    (active, const_fn_floating_point_arithmetic, "1.48.0", Some(57241), None),
    /// Allows `for _ in _` loops in const contexts.
    (active, const_for, "1.56.0", Some(87575), None),
    /// Allows using `&mut` in constant functions.
    (active, const_mut_refs, "1.41.0", Some(57349), None),
    /// Be more precise when looking for live drops in a const context.
    (active, const_precise_live_drops, "1.46.0", Some(73255), None),
    /// Allows references to types with interior mutability within constants
    (active, const_refs_to_cell, "1.51.0", Some(80384), None),
    /// Allows `impl const Trait for T` syntax.
    (active, const_trait_impl, "1.42.0", Some(67792), None),
    /// Allows the `?` operator in const contexts.
    (active, const_try, "1.56.0", Some(74935), None),
    /// Allows non-builtin attributes in inner attribute position.
    (active, custom_inner_attributes, "1.30.0", Some(54726), None),
    /// Allows custom test frameworks with `#![test_runner]` and `#[test_case]`.
    (active, custom_test_frameworks, "1.30.0", Some(50297), None),
    /// Allows using `#[debugger_visualizer]`.
    (active, debugger_visualizer, "1.62.0", Some(95939), None),
    /// Allows declarative macros 2.0 (`macro`).
    (active, decl_macro, "1.17.0", Some(39412), None),
    /// Allows rustc to inject a default alloc_error_handler
    (active, default_alloc_error_handler, "1.48.0", Some(66741), None),
    /// Allows default type parameters to influence type inference.
    (active, default_type_parameter_fallback, "1.3.0", Some(27336), None),
    /// Allows using `#[deprecated_safe]` to deprecate the safeness of a function or trait
    (active, deprecated_safe, "1.61.0", Some(94978), None),
    /// Allows having using `suggestion` in the `#[deprecated]` attribute.
    (active, deprecated_suggestion, "1.61.0", Some(94785), None),
    /// Tells rustdoc to automatically generate `#[doc(cfg(...))]`.
    (active, doc_auto_cfg, "1.58.0", Some(43781), None),
    /// Allows `#[doc(cfg(...))]`.
    (active, doc_cfg, "1.21.0", Some(43781), None),
    /// Allows `#[doc(cfg_hide(...))]`.
    (active, doc_cfg_hide, "1.57.0", Some(43781), None),
    /// Allows `#[doc(masked)]`.
    (active, doc_masked, "1.21.0", Some(44027), None),
    /// Allows `X..Y` patterns.
    (active, exclusive_range_pattern, "1.11.0", Some(37854), None),
    /// Allows exhaustive pattern matching on types that contain uninhabited types.
    (active, exhaustive_patterns, "1.13.0", Some(51085), None),
    /// Allows defining `extern type`s.
    (active, extern_types, "1.23.0", Some(43467), None),
    /// Allows the use of `#[ffi_const]` on foreign functions.
    (active, ffi_const, "1.45.0", Some(58328), None),
    /// Allows the use of `#[ffi_pure]` on foreign functions.
    (active, ffi_pure, "1.45.0", Some(58329), None),
    /// Allows using `#[ffi_returns_twice]` on foreign functions.
    (active, ffi_returns_twice, "1.34.0", Some(58314), None),
    /// Allows using `#[repr(align(...))]` on function items
    (active, fn_align, "1.53.0", Some(82232), None),
    /// Allows defining generators.
    (active, generators, "1.21.0", Some(43122), None),
    /// Infer generic args for both consts and types.
    (active, generic_arg_infer, "1.55.0", Some(85077), None),
    /// Allows associated types to be generic, e.g., `type Foo<T>;` (RFC 1598).
    (active, generic_associated_types, "1.23.0", Some(44265), None),
    /// An extension to the `generic_associated_types` feature, allowing incomplete features.
    (incomplete, generic_associated_types_extended, "1.61.0", Some(95451), None),
    /// Allows non-trivial generic constants which have to have wfness manually propagated to callers
    (incomplete, generic_const_exprs, "1.56.0", Some(76560), None),
    /// Allows using `..X`, `..=X`, `...X`, and `X..` as a pattern.
    (active, half_open_range_patterns, "1.41.0", Some(67264), None),
    /// Allows `if let` guard in match arms.
    (active, if_let_guard, "1.47.0", Some(51114), None),
    /// Allows using imported `main` function
    (active, imported_main, "1.53.0", Some(28937), None),
    /// Allows associated types in inherent impls.
    (incomplete, inherent_associated_types, "1.52.0", Some(8995), None),
    /// Allow anonymous constants from an inline `const` block
    (active, inline_const, "1.49.0", Some(76001), None),
    /// Allow anonymous constants from an inline `const` block in pattern position
    (incomplete, inline_const_pat, "1.58.0", Some(76001), None),
    /// Allows using `pointer` and `reference` in intra-doc links
    (active, intra_doc_pointers, "1.51.0", Some(80896), None),
    /// Allows `#[instruction_set(_)]` attribute
    (active, isa_attribute, "1.48.0", Some(74727), None),
    // Allows setting the threshold for the `large_assignments` lint.
    (active, large_assignments, "1.52.0", Some(83518), None),
    /// Allows `if/while p && let q = r && ...` chains.
    (active, let_chains, "1.37.0", Some(53667), None),
    /// Allows `let...else` statements.
    (active, let_else, "1.56.0", Some(87335), None),
    /// Allows `#[link(..., cfg(..))]`.
    (active, link_cfg, "1.14.0", Some(37406), None),
    /// Allows using `reason` in lint attributes and the `#[expect(lint)]` lint check.
    (active, lint_reasons, "1.31.0", Some(54503), None),
    /// Give access to additional metadata about declarative macro meta-variables.
    (active, macro_metavar_expr, "1.61.0", Some(83527), None),
    /// Allows `#[marker]` on certain traits allowing overlapping implementations.
    (active, marker_trait_attr, "1.30.0", Some(29864), None),
    /// A minimal, sound subset of specialization intended to be used by the
    /// standard library until the soundness issues with specialization
    /// are fixed.
    (active, min_specialization, "1.7.0", Some(31844), None),
    /// Allows qualified paths in struct expressions, struct patterns and tuple struct patterns.
    (active, more_qualified_paths, "1.54.0", Some(86935), None),
    /// Allows the `#[must_not_suspend]` attribute.
    (active, must_not_suspend, "1.57.0", Some(83310), None),
    /// Allows using `#[naked]` on functions.
    (active, naked_functions, "1.9.0", Some(32408), None),
    /// Allows specifying the as-needed link modifier
    (active, native_link_modifiers_as_needed, "1.53.0", Some(81490), None),
    /// Allows specifying the verbatim link modifier
    (active, native_link_modifiers_verbatim, "1.53.0", Some(81490), None),
    /// Allow negative trait implementations.
    (active, negative_impls, "1.44.0", Some(68318), None),
    /// Allows the `!` type. Does not imply 'exhaustive_patterns' (below) any more.
    (active, never_type, "1.13.0", Some(35121), None),
    /// Allows diverging expressions to fall back to `!` rather than `()`.
    (active, never_type_fallback, "1.41.0", Some(65992), None),
    /// Allows `#![no_core]`.
    (active, no_core, "1.3.0", Some(29639), None),
    /// Allows function attribute `#[no_coverage]`, to bypass coverage
    /// instrumentation of that function.
    (active, no_coverage, "1.53.0", Some(84605), None),
    /// Allows the use of `no_sanitize` attribute.
    (active, no_sanitize, "1.42.0", Some(39699), None),
    /// Allows using the `non_exhaustive_omitted_patterns` lint.
    (active, non_exhaustive_omitted_patterns_lint, "1.57.0", Some(89554), None),
    /// Allows making `dyn Trait` well-formed even if `Trait` is not object safe.
    /// In that case, `dyn Trait: Trait` does not hold. Moreover, coercions and
    /// casts in safe Rust to `dyn Trait` for such a `Trait` is also forbidden.
    (active, object_safe_for_dispatch, "1.40.0", Some(43561), None),
    /// Allows using `#[optimize(X)]`.
    (active, optimize_attribute, "1.34.0", Some(54882), None),
    /// Allows `extern "platform-intrinsic" { ... }`.
    (active, platform_intrinsics, "1.4.0", Some(27731), None),
    /// Allows using `#![plugin(myplugin)]`.
    (active, plugin, "1.0.0", Some(29597), None),
    /// Allows exhaustive integer pattern matching on `usize` and `isize`.
    (active, precise_pointer_size_matching, "1.32.0", Some(56354), None),
    /// Allows macro attributes on expressions, statements and non-inline modules.
    (active, proc_macro_hygiene, "1.30.0", Some(54727), None),
    /// Allows the use of raw-dylibs (RFC 2627).
    (active, raw_dylib, "CURRENT_RUSTC_VERSION", Some(58713), None),
    /// Allows `&raw const $place_expr` and `&raw mut $place_expr` expressions.
    (active, raw_ref_op, "1.41.0", Some(64490), None),
    /// Allows using the `#[register_tool]` attribute.
    (active, register_tool, "1.41.0", Some(66079), None),
    /// Allows the `#[repr(i128)]` attribute for enums.
    (incomplete, repr128, "1.16.0", Some(56071), None),
    /// Allows `repr(simd)` and importing the various simd intrinsics.
    (active, repr_simd, "1.4.0", Some(27731), None),
    /// Allows return-position `impl Trait` in traits.
    (incomplete, return_position_impl_trait_in_trait, "CURRENT_RUSTC_VERSION", Some(91611), None),
    /// Allows `extern "rust-cold"`.
    (active, rust_cold_cc, "1.63.0", Some(97544), None),
    /// Allows the use of SIMD types in functions declared in `extern` blocks.
    (active, simd_ffi, "1.0.0", Some(27731), None),
    /// Allows specialization of implementations (RFC 1210).
    (incomplete, specialization, "1.7.0", Some(31844), None),
    /// Allows attributes on expressions and non-item statements.
    (active, stmt_expr_attributes, "1.6.0", Some(15701), None),
    /// Allows lints part of the strict provenance effort.
    (active, strict_provenance, "1.61.0", Some(95228), None),
    /// Allows the use of `#[target_feature]` on safe functions.
    (active, target_feature_11, "1.45.0", Some(69098), None),
    /// Allows using `#[thread_local]` on `static` items.
    (active, thread_local, "1.0.0", Some(29594), None),
    /// Allows defining `trait X = A + B;` alias items.
    (active, trait_alias, "1.24.0", Some(41517), None),
    /// Allows upcasting trait objects via supertraits.
    /// Trait upcasting is casting, e.g., `dyn Foo -> dyn Bar` where `Foo: Bar`.
    (incomplete, trait_upcasting, "1.56.0", Some(65991), None),
    /// Allows #[repr(transparent)] on unions (RFC 2645).
    (active, transparent_unions, "1.37.0", Some(60405), None),
    /// Allows inconsistent bounds in where clauses.
    (active, trivial_bounds, "1.28.0", Some(48214), None),
    /// Allows using `try {...}` expressions.
    (active, try_blocks, "1.29.0", Some(31436), None),
    /// Allows `impl Trait` to be used inside type aliases (RFC 2515).
    (active, type_alias_impl_trait, "1.38.0", Some(63063), None),
    /// Allows the use of type ascription in expressions.
    (active, type_ascription, "1.6.0", Some(23416), None),
    /// Allows creation of instances of a struct by moving fields that have
    /// not changed from prior instances of the same struct (RFC #2528)
    (active, type_changing_struct_update, "1.58.0", Some(86555), None),
    /// Enables rustc to generate code that instructs libstd to NOT ignore SIGPIPE.
    (active, unix_sigpipe, "CURRENT_RUSTC_VERSION", Some(97889), None),
    /// Allows unsized fn parameters.
    (active, unsized_fn_params, "1.49.0", Some(48055), None),
    /// Allows unsized rvalues at arguments and parameters.
    (incomplete, unsized_locals, "1.30.0", Some(48055), None),
    /// Allows unsized tuple coercion.
    (active, unsized_tuple_coercion, "1.20.0", Some(42877), None),
    /// Allows using the `#[used(linker)]` (or `#[used(compiler)]`) attribute.
    (active, used_with_arg, "1.60.0", Some(93798), None),
    /// Allows `extern "wasm" fn`
    (active, wasm_abi, "1.53.0", Some(83788), None),
    /// Allows `do yeet` expressions
    (active, yeet_expr, "1.62.0", Some(96373), None),
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!
    // Features are listed in alphabetical order. Tidy will fail if you don't keep it this way.
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!

    // -------------------------------------------------------------------------
    // feature-group-end: actual feature gates
    // -------------------------------------------------------------------------
);

/// Some features are not allowed to be used together at the same time, if
/// the two are present, produce an error.
///
/// Currently empty, but we will probably need this again in the future,
/// so let's keep it in for now.
pub const INCOMPATIBLE_FEATURES: &[(Symbol, Symbol)] = &[];
