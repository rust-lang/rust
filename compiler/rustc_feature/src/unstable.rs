//! List of the unstable feature gates.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rustc_data_structures::fx::FxHashSet;
use rustc_span::{Span, Symbol, sym};

use super::{Feature, to_nonzero};

#[derive(PartialEq)]
enum FeatureStatus {
    Default,
    Incomplete,
    Internal,
}

macro_rules! status_to_enum {
    (unstable) => {
        FeatureStatus::Default
    };
    (incomplete) => {
        FeatureStatus::Incomplete
    };
    (internal) => {
        FeatureStatus::Internal
    };
}

/// A set of features to be used by later passes.
///
/// There are two ways to check if a language feature `foo` is enabled:
/// - Directly with the `foo` method, e.g. `if tcx.features().foo() { ... }`.
/// - With the `enabled` method, e.g. `if tcx.features.enabled(sym::foo) { ... }`.
///
/// The former is preferred. `enabled` should only be used when the feature symbol is not a
/// constant, e.g. a parameter, or when the feature is a library feature.
#[derive(Clone, Default, Debug)]
pub struct Features {
    /// `#![feature]` attrs for language features, for error reporting.
    enabled_lang_features: Vec<EnabledLangFeature>,
    /// `#![feature]` attrs for non-language (library) features.
    enabled_lib_features: Vec<EnabledLibFeature>,
    /// `enabled_lang_features` + `enabled_lib_features`.
    enabled_features: FxHashSet<Symbol>,
}

/// Information about an enabled language feature.
#[derive(Debug, Copy, Clone)]
pub struct EnabledLangFeature {
    /// Name of the feature gate guarding the language feature.
    pub gate_name: Symbol,
    /// Span of the `#[feature(...)]` attribute.
    pub attr_sp: Span,
    /// If the lang feature is stable, the version number when it was stabilized.
    pub stable_since: Option<Symbol>,
}

/// Information about an enabled library feature.
#[derive(Debug, Copy, Clone)]
pub struct EnabledLibFeature {
    pub gate_name: Symbol,
    pub attr_sp: Span,
}

impl Features {
    /// `since` should be set for stable features that are nevertheless enabled with a `#[feature]`
    /// attribute, indicating since when they are stable.
    pub fn set_enabled_lang_feature(&mut self, lang_feat: EnabledLangFeature) {
        self.enabled_lang_features.push(lang_feat);
        self.enabled_features.insert(lang_feat.gate_name);
    }

    pub fn set_enabled_lib_feature(&mut self, lib_feat: EnabledLibFeature) {
        self.enabled_lib_features.push(lib_feat);
        self.enabled_features.insert(lib_feat.gate_name);
    }

    /// Returns a list of [`EnabledLangFeature`] with info about:
    ///
    /// - Feature gate name.
    /// - The span of the `#[feature]` attribute.
    /// - For stable language features, version info for when it was stabilized.
    pub fn enabled_lang_features(&self) -> &Vec<EnabledLangFeature> {
        &self.enabled_lang_features
    }

    pub fn enabled_lib_features(&self) -> &Vec<EnabledLibFeature> {
        &self.enabled_lib_features
    }

    pub fn enabled_features(&self) -> &FxHashSet<Symbol> {
        &self.enabled_features
    }

    /// Is the given feature enabled (via `#[feature(...)]`)?
    pub fn enabled(&self, feature: Symbol) -> bool {
        self.enabled_features.contains(&feature)
    }
}

macro_rules! declare_features {
    ($(
        $(#[doc = $doc:tt])* ($status:ident, $feature:ident, $ver:expr, $issue:expr),
    )+) => {
        /// Unstable language features that are being implemented or being
        /// considered for acceptance (stabilization) or removal.
        pub static UNSTABLE_LANG_FEATURES: &[Feature] = &[
            $(Feature {
                name: sym::$feature,
                since: $ver,
                issue: to_nonzero($issue),
            }),+
        ];

        impl Features {
            $(
                pub fn $feature(&self) -> bool {
                    self.enabled_features.contains(&sym::$feature)
                }
            )*

            /// Some features are known to be incomplete and using them is likely to have
            /// unanticipated results, such as compiler crashes. We warn the user about these
            /// to alert them.
            pub fn incomplete(&self, feature: Symbol) -> bool {
                match feature {
                    $(
                        sym::$feature => status_to_enum!($status) == FeatureStatus::Incomplete,
                    )*
                    _ if self.enabled_features.contains(&feature) => {
                        // Accepted/removed features and library features aren't in this file but
                        // are never incomplete.
                        false
                    }
                    _ => panic!("`{}` was not listed in `declare_features`", feature),
                }
            }

            /// Some features are internal to the compiler and standard library and should not
            /// be used in normal projects. We warn the user about these to alert them.
            pub fn internal(&self, feature: Symbol) -> bool {
                match feature {
                    $(
                        sym::$feature => status_to_enum!($status) == FeatureStatus::Internal,
                    )*
                    _ if self.enabled_features.contains(&feature) => {
                        // This could be accepted/removed, or a libs feature.
                        // Accepted/removed features aren't in this file but are never internal
                        // (a removed feature might have been internal, but that's now irrelevant).
                        // Libs features are internal if they end in `_internal` or `_internals`.
                        // As a special exception we also consider `core_intrinsics` internal;
                        // renaming that age-old feature is just not worth the hassle.
                        // We just always test the name; it's not a big deal if we accidentally hit
                        // an accepted/removed lang feature that way.
                        let name = feature.as_str();
                        name == "core_intrinsics" || name.ends_with("_internal") || name.ends_with("_internals")
                    }
                    _ => panic!("`{}` was not listed in `declare_features`", feature),
                }
            }
        }
    };
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
// was set.
//
// Note that the features are grouped into internal/user-facing and then
// sorted alphabetically inside those groups. This is enforced with tidy.
//
// N.B., `tools/tidy/src/features.rs` parses this information directly out of the
// source, so take care when modifying it.

#[rustfmt::skip]
declare_features! (
    // -------------------------------------------------------------------------
    // feature-group-start: internal feature gates (no tracking issue)
    // -------------------------------------------------------------------------
    // no-tracking-issue-start

    /// Allows using the `unadjusted` ABI; perma-unstable.
    (internal, abi_unadjusted, "1.16.0", None),
    /// Allows using `#![needs_allocator]`, an implementation detail of `#[global_allocator]`.
    (internal, allocator_internals, "1.20.0", None),
    /// Allows using `#[allow_internal_unsafe]`. This is an
    /// attribute on `macro_rules!` and can't use the attribute handling
    /// below (it has to be checked before expansion possibly makes
    /// macros disappear).
    (internal, allow_internal_unsafe, "1.0.0", None),
    /// Allows using `#[allow_internal_unstable]`. This is an
    /// attribute on `macro_rules!` and can't use the attribute handling
    /// below (it has to be checked before expansion possibly makes
    /// macros disappear).
    (internal, allow_internal_unstable, "1.0.0", None),
    /// Allows using anonymous lifetimes in argument-position impl-trait.
    (unstable, anonymous_lifetime_in_impl_trait, "1.63.0", None),
    /// Allows access to the emscripten_wasm_eh config, used by panic_unwind and unwind
    (internal, cfg_emscripten_wasm_eh, "1.86.0", None),
    /// Allows checking whether or not the backend correctly supports unstable float types.
    (internal, cfg_target_has_reliable_f16_f128, "1.88.0", None),
    /// Allows identifying the `compiler_builtins` crate.
    (internal, compiler_builtins, "1.13.0", None),
    /// Allows writing custom MIR
    (internal, custom_mir, "1.65.0", None),
    /// Outputs useful `assert!` messages
    (unstable, generic_assert, "1.63.0", None),
    /// Allows using the #[rustc_intrinsic] attribute.
    (internal, intrinsics, "1.0.0", None),
    /// Allows using `#[lang = ".."]` attribute for linking items to special compiler logic.
    (internal, lang_items, "1.0.0", None),
    /// Allows `#[link(..., cfg(..))]`; perma-unstable per #37406
    (internal, link_cfg, "1.14.0", None),
    /// Allows using `?Trait` trait bounds in more contexts.
    (internal, more_maybe_bounds, "1.82.0", None),
    /// Allows the `multiple_supertrait_upcastable` lint.
    (unstable, multiple_supertrait_upcastable, "1.69.0", None),
    /// Allow negative trait bounds. This is an internal-only feature for testing the trait solver!
    (internal, negative_bounds, "1.71.0", None),
    /// Allows using `#[omit_gdb_pretty_printer_section]`.
    (internal, omit_gdb_pretty_printer_section, "1.5.0", None),
    /// Set the maximum pattern complexity allowed (not limited by default).
    (internal, pattern_complexity_limit, "1.78.0", None),
    /// Allows using pattern types.
    (internal, pattern_types, "1.79.0", Some(123646)),
    /// Allows using `#[prelude_import]` on glob `use` items.
    (internal, prelude_import, "1.2.0", None),
    /// Used to identify crates that contain the profiler runtime.
    (internal, profiler_runtime, "1.18.0", None),
    /// Allows using `rustc_*` attributes (RFC 572).
    (internal, rustc_attrs, "1.0.0", None),
    /// Allows using the `#[stable]` and `#[unstable]` attributes.
    (internal, staged_api, "1.0.0", None),
    /// Added for testing unstable lints; perma-unstable.
    (internal, test_unstable_lint, "1.60.0", None),
    /// Helps with formatting for `group_imports = "StdExternalCrate"`.
    (unstable, unqualified_local_imports, "1.83.0", Some(138299)),
    /// Use for stable + negative coherence and strict coherence depending on trait's
    /// rustc_strict_coherence value.
    (unstable, with_negative_coherence, "1.60.0", None),
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

    /// Allows using the `vectorcall` ABI.
    (unstable, abi_vectorcall, "1.7.0", Some(124485)),
    /// Allows features specific to auto traits.
    /// Renamed from `optin_builtin_traits`.
    (unstable, auto_traits, "1.50.0", Some(13231)),
    /// Allows using `box` in patterns (RFC 469).
    (unstable, box_patterns, "1.0.0", Some(29641)),
    /// Allows builtin # foo() syntax
    (internal, builtin_syntax, "1.71.0", Some(110680)),
    /// Allows `#[doc(notable_trait)]`.
    /// Renamed from `doc_spotlight`.
    (unstable, doc_notable_trait, "1.52.0", Some(45040)),
    /// Allows using the `may_dangle` attribute (RFC 1327).
    (unstable, dropck_eyepatch, "1.10.0", Some(34761)),
    /// Allows using the `#[fundamental]` attribute.
    (unstable, fundamental, "1.0.0", Some(29635)),
    /// Allows using `#[link_name="llvm.*"]`.
    (internal, link_llvm_intrinsics, "1.0.0", Some(29602)),
    /// Allows using the `#[linkage = ".."]` attribute.
    (unstable, linkage, "1.0.0", Some(29603)),
    /// Allows declaring with `#![needs_panic_runtime]` that a panic runtime is needed.
    (internal, needs_panic_runtime, "1.10.0", Some(32837)),
    /// Allows using the `#![panic_runtime]` attribute.
    (internal, panic_runtime, "1.10.0", Some(32837)),
    /// Allows using `#[rustc_allow_const_fn_unstable]`.
    /// This is an attribute on `const fn` for the same
    /// purpose as `#[allow_internal_unstable]`.
    (internal, rustc_allow_const_fn_unstable, "1.49.0", Some(69399)),
    /// Allows using compiler's own crates.
    (unstable, rustc_private, "1.0.0", Some(27812)),
    /// Allows using internal rustdoc features like `doc(keyword)`.
    (internal, rustdoc_internals, "1.58.0", Some(90418)),
    /// Allows using the `rustdoc::missing_doc_code_examples` lint
    (unstable, rustdoc_missing_doc_code_examples, "1.31.0", Some(101730)),
    /// Allows using `#[structural_match]` which indicates that a type is structurally matchable.
    /// FIXME: Subsumed by trait `StructuralPartialEq`, cannot move to removed until a library
    /// feature with the same name exists.
    (unstable, structural_match, "1.8.0", Some(31434)),
    /// Allows using the `rust-call` ABI.
    (unstable, unboxed_closures, "1.0.0", Some(29625)),
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
    (unstable, aarch64_unstable_target_feature, "1.82.0", Some(44839)),
    (unstable, aarch64_ver_target_feature, "1.27.0", Some(44839)),
    (unstable, apx_target_feature, "1.88.0", Some(139284)),
    (unstable, arm_target_feature, "1.27.0", Some(44839)),
    (unstable, bpf_target_feature, "1.54.0", Some(44839)),
    (unstable, csky_target_feature, "1.73.0", Some(44839)),
    (unstable, ermsb_target_feature, "1.49.0", Some(44839)),
    (unstable, hexagon_target_feature, "1.27.0", Some(44839)),
    (unstable, lahfsahf_target_feature, "1.78.0", Some(44839)),
    (unstable, loongarch_target_feature, "1.73.0", Some(44839)),
    (unstable, m68k_target_feature, "1.85.0", Some(134328)),
    (unstable, mips_target_feature, "1.27.0", Some(44839)),
    (unstable, movrs_target_feature, "1.88.0", Some(137976)),
    (unstable, powerpc_target_feature, "1.27.0", Some(44839)),
    (unstable, prfchw_target_feature, "1.78.0", Some(44839)),
    (unstable, riscv_target_feature, "1.45.0", Some(44839)),
    (unstable, rtm_target_feature, "1.35.0", Some(44839)),
    (unstable, s390x_target_feature, "1.82.0", Some(44839)),
    (unstable, sparc_target_feature, "1.84.0", Some(132783)),
    (unstable, sse4a_target_feature, "1.27.0", Some(44839)),
    (unstable, tbm_target_feature, "1.27.0", Some(44839)),
    (unstable, wasm_target_feature, "1.30.0", Some(44839)),
    (unstable, x87_target_feature, "1.85.0", Some(44839)),
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!
    // Features are listed in alphabetical order. Tidy will fail if you don't keep it this way.
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!

    // -------------------------------------------------------------------------
    // feature-group-end: actual feature gates (target features)
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // feature-group-start: actual feature gates
    // -------------------------------------------------------------------------

    /// Allows `extern "avr-interrupt" fn()` and `extern "avr-non-blocking-interrupt" fn()`.
    (unstable, abi_avr_interrupt, "1.45.0", Some(69664)),
    /// Allows `extern "C-cmse-nonsecure-call" fn()`.
    (unstable, abi_c_cmse_nonsecure_call, "1.51.0", Some(81391)),
    /// Allows `extern "custom" fn()`.
    (unstable, abi_custom, "CURRENT_RUSTC_VERSION", Some(140829)),
    /// Allows `extern "gpu-kernel" fn()`.
    (unstable, abi_gpu_kernel, "1.86.0", Some(135467)),
    /// Allows `extern "msp430-interrupt" fn()`.
    (unstable, abi_msp430_interrupt, "1.16.0", Some(38487)),
    /// Allows `extern "ptx-*" fn()`.
    (unstable, abi_ptx, "1.15.0", Some(38788)),
    /// Allows `extern "riscv-interrupt-m" fn()` and `extern "riscv-interrupt-s" fn()`.
    (unstable, abi_riscv_interrupt, "1.73.0", Some(111889)),
    /// Allows `extern "x86-interrupt" fn()`.
    (unstable, abi_x86_interrupt, "1.17.0", Some(40180)),
    /// Allows additional const parameter types, such as `[u8; 10]` or user defined types
    (unstable, adt_const_params, "1.56.0", Some(95174)),
    /// Allows defining an `#[alloc_error_handler]`.
    (unstable, alloc_error_handler, "1.29.0", Some(51540)),
    /// Allows inherent and trait methods with arbitrary self types.
    (unstable, arbitrary_self_types, "1.23.0", Some(44874)),
    /// Allows inherent and trait methods with arbitrary self types that are raw pointers.
    (unstable, arbitrary_self_types_pointers, "1.83.0", Some(44874)),
    /// Allows #[cfg(...)] on inline assembly templates and operands.
    (unstable, asm_cfg, "CURRENT_RUSTC_VERSION", Some(140364)),
    /// Enables experimental inline assembly support for additional architectures.
    (unstable, asm_experimental_arch, "1.58.0", Some(93335)),
    /// Enables experimental register support in inline assembly.
    (unstable, asm_experimental_reg, "1.85.0", Some(133416)),
    /// Allows using `label` operands in inline assembly together with output operands.
    (unstable, asm_goto_with_outputs, "1.85.0", Some(119364)),
    /// Allows the `may_unwind` option in inline assembly.
    (unstable, asm_unwind, "1.58.0", Some(93334)),
    /// Allows users to enforce equality of associated constants `TraitImpl<AssocConst=3>`.
    (unstable, associated_const_equality, "1.58.0", Some(92827)),
    /// Allows associated type defaults.
    (unstable, associated_type_defaults, "1.2.0", Some(29661)),
    /// Allows implementing `AsyncDrop`.
    (incomplete, async_drop, "1.88.0", Some(126482)),
    /// Allows async functions to be called from `dyn Trait`.
    (incomplete, async_fn_in_dyn_trait, "1.85.0", Some(133119)),
    /// Allows `#[track_caller]` on async functions.
    (unstable, async_fn_track_caller, "1.73.0", Some(110011)),
    /// Allows `for await` loops.
    (unstable, async_for_loop, "1.77.0", Some(118898)),
    /// Allows `async` trait bound modifier.
    (unstable, async_trait_bounds, "1.85.0", Some(62290)),
    /// Allows using Intel AVX10 target features and intrinsics
    (unstable, avx10_target_feature, "1.88.0", Some(138843)),
    /// Allows using C-variadics.
    (unstable, c_variadic, "1.34.0", Some(44930)),
    /// Allows the use of `#[cfg(contract_checks)` to check if contract checks are enabled.
    (unstable, cfg_contract_checks, "1.86.0", Some(128044)),
    /// Allows the use of `#[cfg(overflow_checks)` to check if integer overflow behaviour.
    (unstable, cfg_overflow_checks, "1.71.0", Some(111466)),
    /// Provides the relocation model information as cfg entry
    (unstable, cfg_relocation_model, "1.73.0", Some(114929)),
    /// Allows the use of `#[cfg(sanitize = "option")]`; set when -Zsanitizer is used.
    (unstable, cfg_sanitize, "1.41.0", Some(39699)),
    /// Allows `cfg(sanitizer_cfi_generalize_pointers)` and `cfg(sanitizer_cfi_normalize_integers)`.
    (unstable, cfg_sanitizer_cfi, "1.77.0", Some(89653)),
    /// Allows `cfg(target(abi = "..."))`.
    (unstable, cfg_target_compact, "1.63.0", Some(96901)),
    /// Allows `cfg(target_has_atomic_load_store = "...")`.
    (unstable, cfg_target_has_atomic, "1.60.0", Some(94039)),
    /// Allows `cfg(target_has_atomic_equal_alignment = "...")`.
    (unstable, cfg_target_has_atomic_equal_alignment, "1.60.0", Some(93822)),
    /// Allows `cfg(target_thread_local)`.
    (unstable, cfg_target_thread_local, "1.7.0", Some(29594)),
    /// Allows the use of `#[cfg(ub_checks)` to check if UB checks are enabled.
    (unstable, cfg_ub_checks, "1.79.0", Some(123499)),
    /// Allow conditional compilation depending on rust version
    (unstable, cfg_version, "1.45.0", Some(64796)),
    /// Allows to use the `#[cfi_encoding = ""]` attribute.
    (unstable, cfi_encoding, "1.71.0", Some(89653)),
    /// Allows `for<...>` on closures and coroutines.
    (unstable, closure_lifetime_binder, "1.64.0", Some(97362)),
    /// Allows `#[track_caller]` on closures and coroutines.
    (unstable, closure_track_caller, "1.57.0", Some(87417)),
    /// Allows `extern "C-cmse-nonsecure-entry" fn()`.
    (unstable, cmse_nonsecure_entry, "1.48.0", Some(75835)),
    /// Allows `async {}` expressions in const contexts.
    (unstable, const_async_blocks, "1.53.0", Some(85368)),
    /// Allows `const || {}` closures in const contexts.
    (incomplete, const_closures, "1.68.0", Some(106003)),
    /// Allows using `~const Destruct` bounds and calling drop impls in const contexts.
    (unstable, const_destruct, "1.85.0", Some(133214)),
    /// Allows `for _ in _` loops in const contexts.
    (unstable, const_for, "1.56.0", Some(87575)),
    /// Be more precise when looking for live drops in a const context.
    (unstable, const_precise_live_drops, "1.46.0", Some(73255)),
    /// Allows `impl const Trait for T` syntax.
    (unstable, const_trait_impl, "1.42.0", Some(67792)),
    /// Allows the `?` operator in const contexts.
    (unstable, const_try, "1.56.0", Some(74935)),
    /// Allows use of contracts attributes.
    (incomplete, contracts, "1.86.0", Some(128044)),
    /// Allows access to internal machinery used to implement contracts.
    (internal, contracts_internals, "1.86.0", Some(128044)),
    /// Allows coroutines to be cloned.
    (unstable, coroutine_clone, "1.65.0", Some(95360)),
    /// Allows defining coroutines.
    (unstable, coroutines, "1.21.0", Some(43122)),
    /// Allows function attribute `#[coverage(on/off)]`, to control coverage
    /// instrumentation of that function.
    (unstable, coverage_attribute, "1.74.0", Some(84605)),
    /// Allows non-builtin attributes in inner attribute position.
    (unstable, custom_inner_attributes, "1.30.0", Some(54726)),
    /// Allows custom test frameworks with `#![test_runner]` and `#[test_case]`.
    (unstable, custom_test_frameworks, "1.30.0", Some(50297)),
    /// Allows declarative macros 2.0 (`macro`).
    (unstable, decl_macro, "1.17.0", Some(39412)),
    /// Allows the use of default values on struct definitions and the construction of struct
    /// literals with the functional update syntax without a base.
    (unstable, default_field_values, "1.85.0", Some(132162)),
    /// Allows using `#[deprecated_safe]` to deprecate the safeness of a function or trait
    (unstable, deprecated_safe, "1.61.0", Some(94978)),
    /// Allows having using `suggestion` in the `#[deprecated]` attribute.
    (unstable, deprecated_suggestion, "1.61.0", Some(94785)),
    /// Allows deref patterns.
    (incomplete, deref_patterns, "1.79.0", Some(87121)),
    /// Tells rustdoc to automatically generate `#[doc(cfg(...))]`.
    (unstable, doc_auto_cfg, "1.58.0", Some(43781)),
    /// Allows `#[doc(cfg(...))]`.
    (unstable, doc_cfg, "1.21.0", Some(43781)),
    /// Allows `#[doc(cfg_hide(...))]`.
    (unstable, doc_cfg_hide, "1.57.0", Some(43781)),
    /// Allows `#[doc(masked)]`.
    (unstable, doc_masked, "1.21.0", Some(44027)),
    /// Allows `dyn* Trait` objects.
    (incomplete, dyn_star, "1.65.0", Some(102425)),
    /// Allows the .use postfix syntax `x.use` and use closures `use |x| { ... }`
    (incomplete, ergonomic_clones, "1.87.0", Some(132290)),
    /// Allows exhaustive pattern matching on types that contain uninhabited types.
    (unstable, exhaustive_patterns, "1.13.0", Some(51085)),
    /// Disallows `extern` without an explicit ABI.
    (unstable, explicit_extern_abis, "1.88.0", Some(134986)),
    /// Allows explicit tail calls via `become` expression.
    (incomplete, explicit_tail_calls, "1.72.0", Some(112788)),
    /// Allows using `#[export_stable]` which indicates that an item is exportable.
    (incomplete, export_stable, "1.88.0", Some(139939)),
    /// Allows using `aapcs`, `efiapi`, `sysv64` and `win64` as calling conventions
    /// for functions with varargs.
    (unstable, extended_varargs_abi_support, "1.65.0", Some(100189)),
    /// Allows using `system` as a calling convention with varargs.
    (unstable, extern_system_varargs, "1.86.0", Some(136946)),
    /// Allows defining `extern type`s.
    (unstable, extern_types, "1.23.0", Some(43467)),
    /// Allow using 128-bit (quad precision) floating point numbers.
    (unstable, f128, "1.78.0", Some(116909)),
    /// Allow using 16-bit (half precision) floating point numbers.
    (unstable, f16, "1.78.0", Some(116909)),
    /// Allows the use of `#[ffi_const]` on foreign functions.
    (unstable, ffi_const, "1.45.0", Some(58328)),
    /// Allows the use of `#[ffi_pure]` on foreign functions.
    (unstable, ffi_pure, "1.45.0", Some(58329)),
    /// Controlling the behavior of fmt::Debug
    (unstable, fmt_debug, "1.82.0", Some(129709)),
    /// Allows using `#[repr(align(...))]` on function items
    (unstable, fn_align, "1.53.0", Some(82232)),
    /// Support delegating implementation of functions to other already implemented functions.
    (incomplete, fn_delegation, "1.76.0", Some(118212)),
    /// Allows impls for the Freeze trait.
    (internal, freeze_impls, "1.78.0", Some(121675)),
    /// Frontmatter `---` blocks for use by external tools.
    (unstable, frontmatter, "1.88.0", Some(136889)),
    /// Allows defining gen blocks and `gen fn`.
    (unstable, gen_blocks, "1.75.0", Some(117078)),
    /// Infer generic args for both consts and types.
    (unstable, generic_arg_infer, "1.55.0", Some(85077)),
    /// Allows non-trivial generic constants which have to have wfness manually propagated to callers
    (incomplete, generic_const_exprs, "1.56.0", Some(76560)),
    /// Allows generic parameters and where-clauses on free & associated const items.
    (incomplete, generic_const_items, "1.73.0", Some(113521)),
    /// Allows the type of const generics to depend on generic parameters
    (incomplete, generic_const_parameter_types, "1.87.0", Some(137626)),
    /// Allows any generic constants being used as pattern type range ends
    (incomplete, generic_pattern_types, "1.86.0", Some(136574)),
    /// Allows registering static items globally, possibly across crates, to iterate over at runtime.
    (unstable, global_registration, "1.80.0", Some(125119)),
    /// Allows using guards in patterns.
    (incomplete, guard_patterns, "1.85.0", Some(129967)),
    /// Allows using `..=X` as a patterns in slices.
    (unstable, half_open_range_patterns_in_slices, "1.66.0", Some(67264)),
    /// Allows `if let` guard in match arms.
    (unstable, if_let_guard, "1.47.0", Some(51114)),
    /// Allows `impl Trait` to be used inside associated types (RFC 2515).
    (unstable, impl_trait_in_assoc_type, "1.70.0", Some(63063)),
    /// Allows `impl Trait` in bindings (`let`).
    (unstable, impl_trait_in_bindings, "1.64.0", Some(63065)),
    /// Allows `impl Trait` as output type in `Fn` traits in return position of functions.
    (unstable, impl_trait_in_fn_trait_return, "1.64.0", Some(99697)),
    /// Allows `use` associated functions from traits.
    (unstable, import_trait_associated_functions, "1.86.0", Some(134691)),
    /// Allows associated types in inherent impls.
    (incomplete, inherent_associated_types, "1.52.0", Some(8995)),
    /// Allows using `pointer` and `reference` in intra-doc links
    (unstable, intra_doc_pointers, "1.51.0", Some(80896)),
    // Allows setting the threshold for the `large_assignments` lint.
    (unstable, large_assignments, "1.52.0", Some(83518)),
    /// Allow to have type alias types for inter-crate use.
    (incomplete, lazy_type_alias, "1.72.0", Some(112792)),
    /// Allows `if/while p && let q = r && ...` chains.
    (unstable, let_chains, "1.37.0", Some(53667)),
    /// Allows using `#[link(kind = "link-arg", name = "...")]`
    /// to pass custom arguments to the linker.
    (unstable, link_arg_attribute, "1.76.0", Some(99427)),
    /// Give access to additional metadata about declarative macro meta-variables.
    (unstable, macro_metavar_expr, "1.61.0", Some(83527)),
    /// Provides a way to concatenate identifiers using metavariable expressions.
    (unstable, macro_metavar_expr_concat, "1.81.0", Some(124225)),
    /// Allows `#[marker]` on certain traits allowing overlapping implementations.
    (unstable, marker_trait_attr, "1.30.0", Some(29864)),
    /// Enables the generic const args MVP (only bare paths, not arbitrary computation).
    (incomplete, min_generic_const_args, "1.84.0", Some(132980)),
    /// A minimal, sound subset of specialization intended to be used by the
    /// standard library until the soundness issues with specialization
    /// are fixed.
    (unstable, min_specialization, "1.7.0", Some(31844)),
    /// Allows qualified paths in struct expressions, struct patterns and tuple struct patterns.
    (unstable, more_qualified_paths, "1.54.0", Some(86935)),
    /// Allows the `#[must_not_suspend]` attribute.
    (unstable, must_not_suspend, "1.57.0", Some(83310)),
    /// Allows `mut ref` and `mut ref mut` identifier patterns.
    (incomplete, mut_ref, "1.79.0", Some(123076)),
    /// Allows using `#[naked]` on `extern "Rust"` functions.
    (unstable, naked_functions_rustic_abi, "1.88.0", Some(138997)),
    /// Allows using `#[target_feature(enable = "...")]` on `#[naked]` on functions.
    (unstable, naked_functions_target_feature, "1.86.0", Some(138568)),
    /// Allows specifying the as-needed link modifier
    (unstable, native_link_modifiers_as_needed, "1.53.0", Some(81490)),
    /// Allow negative trait implementations.
    (unstable, negative_impls, "1.44.0", Some(68318)),
    /// Allows the `!` pattern.
    (incomplete, never_patterns, "1.76.0", Some(118155)),
    /// Allows the `!` type. Does not imply 'exhaustive_patterns' (below) any more.
    (unstable, never_type, "1.13.0", Some(35121)),
    /// Allows diverging expressions to fall back to `!` rather than `()`.
    (unstable, never_type_fallback, "1.41.0", Some(65992)),
    /// Switch `..` syntax to use the new (`Copy + IntoIterator`) range types.
    (unstable, new_range, "1.86.0", Some(123741)),
    /// Allows `#![no_core]`.
    (unstable, no_core, "1.3.0", Some(29639)),
    /// Allows the use of `no_sanitize` attribute.
    (unstable, no_sanitize, "1.42.0", Some(39699)),
    /// Allows using the `non_exhaustive_omitted_patterns` lint.
    (unstable, non_exhaustive_omitted_patterns_lint, "1.57.0", Some(89554)),
    /// Allows `for<T>` binders in where-clauses
    (incomplete, non_lifetime_binders, "1.69.0", Some(108185)),
    /// Allows using enums in offset_of!
    (unstable, offset_of_enum, "1.75.0", Some(120141)),
    /// Allows using fields with slice type in offset_of!
    (unstable, offset_of_slice, "1.81.0", Some(126151)),
    /// Allows using `#[optimize(X)]`.
    (unstable, optimize_attribute, "1.34.0", Some(54882)),
    /// Allows specifying nop padding on functions for dynamic patching.
    (unstable, patchable_function_entry, "1.81.0", Some(123115)),
    /// Experimental features that make `Pin` more ergonomic.
    (incomplete, pin_ergonomics, "1.83.0", Some(130494)),
    /// Allows postfix match `expr.match { ... }`
    (unstable, postfix_match, "1.79.0", Some(121618)),
    /// Allows macro attributes on expressions, statements and non-inline modules.
    (unstable, proc_macro_hygiene, "1.30.0", Some(54727)),
    /// Allows the use of raw-dylibs on ELF platforms
    (incomplete, raw_dylib_elf, "1.87.0", Some(135694)),
    /// Makes `&` and `&mut` patterns eat only one layer of references in Rust 2024.
    (incomplete, ref_pat_eat_one_layer_2024, "1.79.0", Some(123076)),
    /// Makes `&` and `&mut` patterns eat only one layer of references in Rust 2024—structural variant
    (incomplete, ref_pat_eat_one_layer_2024_structural, "1.81.0", Some(123076)),
    /// Allows using the `#[register_tool]` attribute.
    (unstable, register_tool, "1.41.0", Some(66079)),
    /// Allows `repr(simd)` and importing the various simd intrinsics.
    (unstable, repr_simd, "1.4.0", Some(27731)),
    /// Allows bounding the return type of AFIT/RPITIT.
    (unstable, return_type_notation, "1.70.0", Some(109417)),
    /// Allows `extern "rust-cold"`.
    (unstable, rust_cold_cc, "1.63.0", Some(97544)),
    /// Allows the use of SIMD types in functions declared in `extern` blocks.
    (unstable, simd_ffi, "1.0.0", Some(27731)),
    /// Allows specialization of implementations (RFC 1210).
    (incomplete, specialization, "1.7.0", Some(31844)),
    /// Allows attributes on expressions and non-item statements.
    (unstable, stmt_expr_attributes, "1.6.0", Some(15701)),
    /// Allows lints part of the strict provenance effort.
    (unstable, strict_provenance_lints, "1.61.0", Some(130351)),
    /// Allows string patterns to dereference values to match them.
    (unstable, string_deref_patterns, "1.67.0", Some(87121)),
    /// Allows `super let` statements.
    (unstable, super_let, "1.88.0", Some(139076)),
    /// Allows subtrait items to shadow supertrait items.
    (unstable, supertrait_item_shadowing, "1.86.0", Some(89151)),
    /// Allows using `#[thread_local]` on `static` items.
    (unstable, thread_local, "1.0.0", Some(29594)),
    /// Allows defining `trait X = A + B;` alias items.
    (unstable, trait_alias, "1.24.0", Some(41517)),
    /// Allows for transmuting between arrays with sizes that contain generic consts.
    (unstable, transmute_generic_consts, "1.70.0", Some(109929)),
    /// Allows #[repr(transparent)] on unions (RFC 2645).
    (unstable, transparent_unions, "1.37.0", Some(60405)),
    /// Allows inconsistent bounds in where clauses.
    (unstable, trivial_bounds, "1.28.0", Some(48214)),
    /// Allows using `try {...}` expressions.
    (unstable, try_blocks, "1.29.0", Some(31436)),
    /// Allows `impl Trait` to be used inside type aliases (RFC 2515).
    (unstable, type_alias_impl_trait, "1.38.0", Some(63063)),
    /// Allows creation of instances of a struct by moving fields that have
    /// not changed from prior instances of the same struct (RFC #2528)
    (unstable, type_changing_struct_update, "1.58.0", Some(86555)),
    /// Allows using `unsafe<'a> &'a T` unsafe binder types.
    (incomplete, unsafe_binders, "1.85.0", Some(130516)),
    /// Allows declaring fields `unsafe`.
    (incomplete, unsafe_fields, "1.85.0", Some(132922)),
    /// Allows const generic parameters to be defined with types that
    /// are not `Sized`, e.g. `fn foo<const N: [u8]>() {`.
    (incomplete, unsized_const_params, "1.82.0", Some(95174)),
    /// Allows unsized fn parameters.
    (internal, unsized_fn_params, "1.49.0", Some(48055)),
    /// Allows unsized rvalues at arguments and parameters.
    (incomplete, unsized_locals, "1.30.0", Some(48055)),
    /// Allows using the `#[used(linker)]` (or `#[used(compiler)]`) attribute.
    (unstable, used_with_arg, "1.60.0", Some(93798)),
    /// Allows use of attributes in `where` clauses.
    (unstable, where_clause_attrs, "1.87.0", Some(115590)),
    /// Allows use of x86 `AMX` target-feature attributes and intrinsics
    (unstable, x86_amx_intrinsics, "1.81.0", Some(126622)),
    /// Allows use of the `xop` target-feature
    (unstable, xop_target_feature, "1.81.0", Some(127208)),
    /// Allows `do yeet` expressions
    (unstable, yeet_expr, "1.62.0", Some(96373)),
    (unstable, yield_expr, "1.87.0", Some(43122)),
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!
    // Features are listed in alphabetical order. Tidy will fail if you don't keep it this way.
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!

    // -------------------------------------------------------------------------
    // feature-group-end: actual feature gates
    // -------------------------------------------------------------------------
);

impl Features {
    pub fn dump_feature_usage_metrics(
        &self,
        metrics_path: PathBuf,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(serde::Serialize)]
        struct LibFeature {
            timestamp: u128,
            symbol: String,
        }

        #[derive(serde::Serialize)]
        struct LangFeature {
            timestamp: u128,
            symbol: String,
            since: Option<String>,
        }

        #[derive(serde::Serialize)]
        struct FeatureUsage {
            lib_features: Vec<LibFeature>,
            lang_features: Vec<LangFeature>,
        }

        let metrics_file = std::fs::File::create(metrics_path)?;
        let metrics_file = std::io::BufWriter::new(metrics_file);

        let now = || {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should always be greater than the unix epoch")
                .as_nanos()
        };

        let lib_features = self
            .enabled_lib_features
            .iter()
            .map(|EnabledLibFeature { gate_name, .. }| LibFeature {
                symbol: gate_name.to_string(),
                timestamp: now(),
            })
            .collect();

        let lang_features = self
            .enabled_lang_features
            .iter()
            .map(|EnabledLangFeature { gate_name, stable_since, .. }| LangFeature {
                symbol: gate_name.to_string(),
                since: stable_since.map(|since| since.to_string()),
                timestamp: now(),
            })
            .collect();

        let feature_usage = FeatureUsage { lib_features, lang_features };

        serde_json::to_writer(metrics_file, &feature_usage)?;

        Ok(())
    }
}

/// Some features are not allowed to be used together at the same time, if
/// the two are present, produce an error.
pub const INCOMPATIBLE_FEATURES: &[(Symbol, Symbol)] = &[
    // Experimental match ergonomics rulesets are incompatible with each other, to simplify the
    // boolean logic required to tell which typing rules to use.
    (sym::ref_pat_eat_one_layer_2024, sym::ref_pat_eat_one_layer_2024_structural),
];
