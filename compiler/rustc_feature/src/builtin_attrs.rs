//! Built-in attributes and `cfg` flag gating.

use std::sync::LazyLock;

use AttributeDuplicates::*;
use AttributeGate::*;
use AttributeType::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_span::edition::Edition;
use rustc_span::{Symbol, sym};

use crate::Features;

type GateFn = fn(&Features) -> bool;

pub type GatedCfg = (Symbol, Symbol, GateFn);

/// `cfg(...)`'s that are feature gated.
const GATED_CFGS: &[GatedCfg] = &[
    // (name in cfg, feature, function to check if the feature is enabled)
    (sym::overflow_checks, sym::cfg_overflow_checks, Features::cfg_overflow_checks),
    (sym::ub_checks, sym::cfg_ub_checks, Features::cfg_ub_checks),
    (sym::contract_checks, sym::cfg_contract_checks, Features::cfg_contract_checks),
    (sym::target_thread_local, sym::cfg_target_thread_local, Features::cfg_target_thread_local),
    (
        sym::target_has_atomic_equal_alignment,
        sym::cfg_target_has_atomic_equal_alignment,
        Features::cfg_target_has_atomic_equal_alignment,
    ),
    (
        sym::target_has_atomic_load_store,
        sym::cfg_target_has_atomic,
        Features::cfg_target_has_atomic,
    ),
    (sym::sanitize, sym::cfg_sanitize, Features::cfg_sanitize),
    (sym::version, sym::cfg_version, Features::cfg_version),
    (sym::relocation_model, sym::cfg_relocation_model, Features::cfg_relocation_model),
    (sym::sanitizer_cfi_generalize_pointers, sym::cfg_sanitizer_cfi, Features::cfg_sanitizer_cfi),
    (sym::sanitizer_cfi_normalize_integers, sym::cfg_sanitizer_cfi, Features::cfg_sanitizer_cfi),
    // this is consistent with naming of the compiler flag it's for
    (sym::fmt_debug, sym::fmt_debug, Features::fmt_debug),
    (sym::emscripten_wasm_eh, sym::cfg_emscripten_wasm_eh, Features::cfg_emscripten_wasm_eh),
    (
        sym::target_has_reliable_f16,
        sym::cfg_target_has_reliable_f16_f128,
        Features::cfg_target_has_reliable_f16_f128,
    ),
    (
        sym::target_has_reliable_f16_math,
        sym::cfg_target_has_reliable_f16_f128,
        Features::cfg_target_has_reliable_f16_f128,
    ),
    (
        sym::target_has_reliable_f128,
        sym::cfg_target_has_reliable_f16_f128,
        Features::cfg_target_has_reliable_f16_f128,
    ),
    (
        sym::target_has_reliable_f128_math,
        sym::cfg_target_has_reliable_f16_f128,
        Features::cfg_target_has_reliable_f16_f128,
    ),
];

/// Find a gated cfg determined by the `pred`icate which is given the cfg's name.
pub fn find_gated_cfg(pred: impl Fn(Symbol) -> bool) -> Option<&'static GatedCfg> {
    GATED_CFGS.iter().find(|(cfg_sym, ..)| pred(*cfg_sym))
}

// If you change this, please modify `src/doc/unstable-book` as well. You must
// move that documentation into the relevant place in the other docs, and
// remove the chapter on the flag.

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AttributeType {
    /// Normal, builtin attribute that is consumed
    /// by the compiler before the unused_attribute check
    Normal,

    /// Builtin attribute that is only allowed at the crate level
    CrateLevel,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AttributeSafety {
    /// Normal attribute that does not need `#[unsafe(...)]`
    Normal,

    /// Unsafe attribute that requires safety obligations to be discharged.
    ///
    /// An error is emitted when `#[unsafe(...)]` is omitted, except when the attribute's edition
    /// is less than the one stored in `unsafe_since`. This handles attributes that were safe in
    /// earlier editions, but become unsafe in later ones.
    Unsafe { unsafe_since: Option<Edition> },
}

#[derive(Clone, Debug, Copy)]
pub enum AttributeGate {
    /// A gated attribute which requires a feature gate to be enabled.
    Gated {
        /// The feature gate, for example `#![feature(rustc_attrs)]` for rustc_* attributes.
        feature: Symbol,
        /// The error message displayed when an attempt is made to use the attribute without its feature gate.
        message: &'static str,
        /// Check function to be called during the `PostExpansionVisitor` pass.
        check: fn(&Features) -> bool,
        /// Notes to be displayed when an attempt is made to use the attribute without its feature gate.
        notes: &'static [&'static str],
    },
    /// Ungated attribute, can be used on all release channels
    Ungated,
}

/// A template that the attribute input must match.
/// Only top-level shape (`#[attr]` vs `#[attr(...)]` vs `#[attr = ...]`) is considered now.
#[derive(Clone, Copy, Default)]
pub struct AttributeTemplate {
    /// If `true`, the attribute is allowed to be a bare word like `#[test]`.
    pub word: bool,
    /// If `Some`, the attribute is allowed to take a list of items like `#[allow(..)]`.
    pub list: Option<&'static str>,
    /// If non-empty, the attribute is allowed to take a list containing exactly
    /// one of the listed words, like `#[coverage(off)]`.
    pub one_of: &'static [Symbol],
    /// If `Some`, the attribute is allowed to be a name/value pair where the
    /// value is a string, like `#[must_use = "reason"]`.
    pub name_value_str: Option<&'static str>,
}

/// How to handle multiple duplicate attributes on the same item.
#[derive(Clone, Copy, Default)]
pub enum AttributeDuplicates {
    /// Duplicates of this attribute are allowed.
    ///
    /// This should only be used with attributes where duplicates have semantic
    /// meaning, or some kind of "additive" behavior. For example, `#[warn(..)]`
    /// can be specified multiple times, and it combines all the entries. Or use
    /// this if there is validation done elsewhere.
    #[default]
    DuplicatesOk,
    /// Duplicates after the first attribute will be an unused_attribute warning.
    ///
    /// This is usually used for "word" attributes, where they are used as a
    /// boolean marker, like `#[used]`. It is not necessarily wrong that there
    /// are duplicates, but the others should probably be removed.
    WarnFollowing,
    /// Same as `WarnFollowing`, but only issues warnings for word-style attributes.
    ///
    /// This is only for special cases, for example multiple `#[macro_use]` can
    /// be warned, but multiple `#[macro_use(...)]` should not because the list
    /// form has different meaning from the word form.
    WarnFollowingWordOnly,
    /// Duplicates after the first attribute will be an error.
    ///
    /// This should be used where duplicates would be ignored, but carry extra
    /// meaning that could cause confusion. For example, `#[stable(since="1.0")]
    /// #[stable(since="2.0")]`, which version should be used for `stable`?
    ErrorFollowing,
    /// Duplicates preceding the last instance of the attribute will be an error.
    ///
    /// This is the same as `ErrorFollowing`, except the last attribute is the
    /// one that is "used". This is typically used in cases like codegen
    /// attributes which usually only honor the last attribute.
    ErrorPreceding,
    /// Duplicates after the first attribute will be an unused_attribute warning
    /// with a note that this will be an error in the future.
    ///
    /// This should be used for attributes that should be `ErrorFollowing`, but
    /// because older versions of rustc silently accepted (and ignored) the
    /// attributes, this is used to transition.
    FutureWarnFollowing,
    /// Duplicates preceding the last instance of the attribute will be a
    /// warning, with a note that this will be an error in the future.
    ///
    /// This is the same as `FutureWarnFollowing`, except the last attribute is
    /// the one that is "used". Ideally these can eventually migrate to
    /// `ErrorPreceding`.
    FutureWarnPreceding,
}

/// A convenience macro for constructing attribute templates.
/// E.g., `template!(Word, List: "description")` means that the attribute
/// supports forms `#[attr]` and `#[attr(description)]`.
macro_rules! template {
    (Word) => { template!(@ true, None, &[], None) };
    (List: $descr: expr) => { template!(@ false, Some($descr), &[], None) };
    (OneOf: $one_of: expr) => { template!(@ false, None, $one_of, None) };
    (NameValueStr: $descr: expr) => { template!(@ false, None, &[], Some($descr)) };
    (Word, List: $descr: expr) => { template!(@ true, Some($descr), &[], None) };
    (Word, NameValueStr: $descr: expr) => { template!(@ true, None, &[], Some($descr)) };
    (List: $descr1: expr, NameValueStr: $descr2: expr) => {
        template!(@ false, Some($descr1), &[], Some($descr2))
    };
    (Word, List: $descr1: expr, NameValueStr: $descr2: expr) => {
        template!(@ true, Some($descr1), &[], Some($descr2))
    };
    (@ $word: expr, $list: expr, $one_of: expr, $name_value_str: expr) => { AttributeTemplate {
        word: $word, list: $list, one_of: $one_of, name_value_str: $name_value_str
    } };
}

macro_rules! ungated {
    (unsafe($edition:ident) $attr:ident, $typ:expr, $tpl:expr, $duplicates:expr, $encode_cross_crate:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            type_: $typ,
            safety: AttributeSafety::Unsafe { unsafe_since: Some(Edition::$edition) },
            template: $tpl,
            gate: Ungated,
            duplicates: $duplicates,
        }
    };
    (unsafe $attr:ident, $typ:expr, $tpl:expr, $duplicates:expr, $encode_cross_crate:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            type_: $typ,
            safety: AttributeSafety::Unsafe { unsafe_since: None },
            template: $tpl,
            gate: Ungated,
            duplicates: $duplicates,
        }
    };
    ($attr:ident, $typ:expr, $tpl:expr, $duplicates:expr, $encode_cross_crate:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            type_: $typ,
            safety: AttributeSafety::Normal,
            template: $tpl,
            gate: Ungated,
            duplicates: $duplicates,
        }
    };
}

macro_rules! gated {
    (unsafe $attr:ident, $typ:expr, $tpl:expr, $duplicates:expr, $encode_cross_crate:expr, $gate:ident, $message:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            type_: $typ,
            safety: AttributeSafety::Unsafe { unsafe_since: None },
            template: $tpl,
            duplicates: $duplicates,
            gate: Gated {
                feature: sym::$gate,
                message: $message,
                check: Features::$gate,
                notes: &[],
            },
        }
    };
    (unsafe $attr:ident, $typ:expr, $tpl:expr, $duplicates:expr, $encode_cross_crate:expr, $message:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            type_: $typ,
            safety: AttributeSafety::Unsafe { unsafe_since: None },
            template: $tpl,
            duplicates: $duplicates,
            gate: Gated {
                feature: sym::$attr,
                message: $message,
                check: Features::$attr,
                notes: &[],
            },
        }
    };
    ($attr:ident, $typ:expr, $tpl:expr, $duplicates:expr, $encode_cross_crate:expr, $gate:ident, $message:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            type_: $typ,
            safety: AttributeSafety::Normal,
            template: $tpl,
            duplicates: $duplicates,
            gate: Gated {
                feature: sym::$gate,
                message: $message,
                check: Features::$gate,
                notes: &[],
            },
        }
    };
    ($attr:ident, $typ:expr, $tpl:expr, $duplicates:expr, $encode_cross_crate:expr, $message:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            type_: $typ,
            safety: AttributeSafety::Normal,
            template: $tpl,
            duplicates: $duplicates,
            gate: Gated {
                feature: sym::$attr,
                message: $message,
                check: Features::$attr,
                notes: &[],
            },
        }
    };
}

macro_rules! rustc_attr {
    (TEST, $attr:ident, $typ:expr, $tpl:expr, $duplicate:expr, $encode_cross_crate:expr $(,)?) => {
        rustc_attr!(
            $attr,
            $typ,
            $tpl,
            $duplicate,
            $encode_cross_crate,
            concat!(
                "the `#[",
                stringify!($attr),
                "]` attribute is used for rustc unit tests"
            ),
        )
    };
    ($attr:ident, $typ:expr, $tpl:expr, $duplicates:expr, $encode_cross_crate:expr, $($notes:expr),* $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            type_: $typ,
            safety: AttributeSafety::Normal,
            template: $tpl,
            duplicates: $duplicates,
            gate: Gated {
                feature: sym::rustc_attrs,
                message: "use of an internal attribute",
                check: Features::rustc_attrs,
                notes: &[
                    concat!("the `#[",
                    stringify!($attr),
                    "]` attribute is an internal implementation detail that will never be stable"),
                    $($notes),*
                    ]
            },
        }
    };
}

macro_rules! experimental {
    ($attr:ident) => {
        concat!("the `#[", stringify!($attr), "]` attribute is an experimental feature")
    };
}

#[derive(PartialEq)]
pub enum EncodeCrossCrate {
    Yes,
    No,
}

pub struct BuiltinAttribute {
    pub name: Symbol,
    /// Whether this attribute is encode cross crate.
    ///
    /// If so, it is encoded in the crate metadata.
    /// Otherwise, it can only be used in the local crate.
    pub encode_cross_crate: EncodeCrossCrate,
    pub type_: AttributeType,
    pub safety: AttributeSafety,
    pub template: AttributeTemplate,
    pub duplicates: AttributeDuplicates,
    pub gate: AttributeGate,
}

/// Attributes that have a special meaning to rustc or rustdoc.
#[rustfmt::skip]
pub static BUILTIN_ATTRIBUTES: &[BuiltinAttribute] = &[
    // ==========================================================================
    // Stable attributes:
    // ==========================================================================

    // Conditional compilation:
    ungated!(cfg, Normal, template!(List: "predicate"), DuplicatesOk, EncodeCrossCrate::Yes),
    ungated!(cfg_attr, Normal, template!(List: "predicate, attr1, attr2, ..."), DuplicatesOk, EncodeCrossCrate::Yes),

    // Testing:
    ungated!(
        ignore, Normal, template!(Word, NameValueStr: "reason"), WarnFollowing,
        EncodeCrossCrate::No,
    ),
    ungated!(
        should_panic, Normal,
        template!(Word, List: r#"expected = "reason""#, NameValueStr: "reason"), FutureWarnFollowing,
        EncodeCrossCrate::No,
    ),
    // FIXME(Centril): This can be used on stable but shouldn't.
    ungated!(
        reexport_test_harness_main, CrateLevel, template!(NameValueStr: "name"), ErrorFollowing,
        EncodeCrossCrate::No,
    ),

    // Macros:
    ungated!(automatically_derived, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::Yes),
    ungated!(
        macro_use, Normal, template!(Word, List: "name1, name2, ..."), WarnFollowingWordOnly,
        EncodeCrossCrate::No,
    ),
    ungated!(macro_escape, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No), // Deprecated synonym for `macro_use`.
    ungated!(
        macro_export, Normal, template!(Word, List: "local_inner_macros"),
        WarnFollowing, EncodeCrossCrate::Yes
    ),
    ungated!(proc_macro, Normal, template!(Word), ErrorFollowing, EncodeCrossCrate::No),
    ungated!(
        proc_macro_derive, Normal, template!(List: "TraitName, /*opt*/ attributes(name1, name2, ...)"),
        ErrorFollowing, EncodeCrossCrate::No,
    ),
    ungated!(proc_macro_attribute, Normal, template!(Word), ErrorFollowing, EncodeCrossCrate::No),

    // Lints:
    ungated!(
        warn, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#),
        DuplicatesOk, EncodeCrossCrate::No,
    ),
    ungated!(
        allow, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#),
        DuplicatesOk, EncodeCrossCrate::No,
    ),
    ungated!(
        expect, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#),
        DuplicatesOk, EncodeCrossCrate::No,
    ),
    ungated!(
        forbid, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#),
        DuplicatesOk, EncodeCrossCrate::No
    ),
    ungated!(
        deny, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#),
        DuplicatesOk, EncodeCrossCrate::No
    ),
    ungated!(
        must_use, Normal, template!(Word, NameValueStr: "reason"),
        FutureWarnFollowing, EncodeCrossCrate::Yes
    ),
    gated!(
        must_not_suspend, Normal, template!(Word, NameValueStr: "reason"), WarnFollowing,
        EncodeCrossCrate::Yes, experimental!(must_not_suspend)
    ),
    ungated!(
        deprecated, Normal,
        template!(
            Word,
            List: r#"/*opt*/ since = "version", /*opt*/ note = "reason""#,
            NameValueStr: "reason"
        ),
        ErrorFollowing, EncodeCrossCrate::Yes
    ),

    // Crate properties:
    ungated!(
        crate_name, CrateLevel, template!(NameValueStr: "name"), FutureWarnFollowing,
        EncodeCrossCrate::No,
    ),
    ungated!(
        crate_type, CrateLevel, template!(NameValueStr: "bin|lib|..."), DuplicatesOk,
        EncodeCrossCrate::No,
    ),

    // ABI, linking, symbols, and FFI
    ungated!(
        link, Normal,
        template!(List: r#"name = "...", /*opt*/ kind = "dylib|static|...", /*opt*/ wasm_import_module = "...", /*opt*/ import_name_type = "decorated|noprefix|undecorated""#),
        DuplicatesOk,
        EncodeCrossCrate::No,
    ),
    ungated!(
        link_name, Normal, template!(NameValueStr: "name"),
        FutureWarnPreceding, EncodeCrossCrate::Yes
    ),
    ungated!(no_link, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No),
    ungated!(repr, Normal, template!(List: "C"), DuplicatesOk, EncodeCrossCrate::No),
    ungated!(unsafe(Edition2024) export_name, Normal, template!(NameValueStr: "name"), FutureWarnPreceding, EncodeCrossCrate::No),
    ungated!(unsafe(Edition2024) link_section, Normal, template!(NameValueStr: "name"), FutureWarnPreceding, EncodeCrossCrate::No),
    ungated!(unsafe(Edition2024) no_mangle, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No),
    ungated!(used, Normal, template!(Word, List: "compiler|linker"), WarnFollowing, EncodeCrossCrate::No),
    ungated!(link_ordinal, Normal, template!(List: "ordinal"), ErrorPreceding, EncodeCrossCrate::Yes),
    ungated!(unsafe naked, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No),

    // Limits:
    ungated!(
        recursion_limit, CrateLevel, template!(NameValueStr: "N"), FutureWarnFollowing,
        EncodeCrossCrate::No
    ),
    ungated!(
        type_length_limit, CrateLevel, template!(NameValueStr: "N"), FutureWarnFollowing,
        EncodeCrossCrate::No
    ),
    gated!(
        move_size_limit, CrateLevel, template!(NameValueStr: "N"), ErrorFollowing,
        EncodeCrossCrate::No, large_assignments, experimental!(move_size_limit)
    ),

    // Entry point:
    ungated!(no_main, CrateLevel, template!(Word), WarnFollowing, EncodeCrossCrate::No),

    // Modules, prelude, and resolution:
    ungated!(path, Normal, template!(NameValueStr: "file"), FutureWarnFollowing, EncodeCrossCrate::No),
    ungated!(no_std, CrateLevel, template!(Word), WarnFollowing, EncodeCrossCrate::No),
    ungated!(no_implicit_prelude, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No),
    ungated!(non_exhaustive, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::Yes),

    // Runtime
    ungated!(
        windows_subsystem, CrateLevel,
        template!(NameValueStr: "windows|console"), FutureWarnFollowing,
        EncodeCrossCrate::No
    ),
    ungated!(panic_handler, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::Yes), // RFC 2070

    // Code generation:
    ungated!(inline, Normal, template!(Word, List: "always|never"), FutureWarnFollowing, EncodeCrossCrate::No),
    ungated!(cold, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No),
    ungated!(no_builtins, CrateLevel, template!(Word), WarnFollowing, EncodeCrossCrate::Yes),
    ungated!(
        target_feature, Normal, template!(List: r#"enable = "name""#),
        DuplicatesOk, EncodeCrossCrate::No,
    ),
    ungated!(track_caller, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::Yes),
    ungated!(instruction_set, Normal, template!(List: "set"), ErrorPreceding, EncodeCrossCrate::No),
    gated!(
        no_sanitize, Normal,
        template!(List: "address, kcfi, memory, thread"), DuplicatesOk,
        EncodeCrossCrate::No, experimental!(no_sanitize)
    ),
    gated!(
        coverage, Normal, template!(OneOf: &[sym::off, sym::on]),
        ErrorPreceding, EncodeCrossCrate::No,
        coverage_attribute, experimental!(coverage)
    ),

    ungated!(
        doc, Normal, template!(List: "hidden|inline|...", NameValueStr: "string"), DuplicatesOk,
        EncodeCrossCrate::Yes
    ),

    // Debugging
    ungated!(
        debugger_visualizer, Normal,
        template!(List: r#"natvis_file = "...", gdb_script_file = "...""#),
        DuplicatesOk, EncodeCrossCrate::No
    ),
    ungated!(collapse_debuginfo, Normal, template!(List: "no|external|yes"), ErrorFollowing,
        EncodeCrossCrate::Yes
    ),

    // ==========================================================================
    // Unstable attributes:
    // ==========================================================================

    // Linking:
    gated!(
        export_stable, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No, experimental!(export_stable)
    ),

    // Testing:
    gated!(
        test_runner, CrateLevel, template!(List: "path"), ErrorFollowing,
        EncodeCrossCrate::Yes, custom_test_frameworks,
        "custom test frameworks are an unstable feature",
    ),
    // RFC #1268
    gated!(
        marker, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No,
        marker_trait_attr, experimental!(marker)
    ),
    gated!(
        thread_local, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No,
        "`#[thread_local]` is an experimental feature, and does not currently handle destructors",
    ),
    gated!(
        no_core, CrateLevel, template!(Word), WarnFollowing,
        EncodeCrossCrate::No, experimental!(no_core)
    ),
    // RFC 2412
    gated!(
        optimize, Normal, template!(List: "none|size|speed"), ErrorPreceding,
        EncodeCrossCrate::No, optimize_attribute, experimental!(optimize)
    ),

    gated!(
        unsafe ffi_pure, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No, experimental!(ffi_pure)
    ),
    gated!(
        unsafe ffi_const, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No, experimental!(ffi_const)
    ),
    gated!(
        register_tool, CrateLevel, template!(List: "tool1, tool2, ..."), DuplicatesOk,
        EncodeCrossCrate::No, experimental!(register_tool),
    ),

    // RFC 2632
    gated!(
        const_trait, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No, const_trait_impl,
        "`const_trait` is a temporary placeholder for marking a trait that is suitable for `const` \
        `impls` and all default bodies as `const`, which may be removed or renamed in the \
        future."
    ),
    // lang-team MCP 147
    gated!(
        deprecated_safe, Normal, template!(List: r#"since = "version", note = "...""#), ErrorFollowing,
        EncodeCrossCrate::Yes, experimental!(deprecated_safe),
    ),

    // `#[cfi_encoding = ""]`
    gated!(
        cfi_encoding, Normal, template!(NameValueStr: "encoding"), ErrorPreceding,
        EncodeCrossCrate::Yes, experimental!(cfi_encoding)
    ),

    // `#[coroutine]` attribute to be applied to closures to make them coroutines instead
    gated!(
        coroutine, Normal, template!(Word), ErrorFollowing,
        EncodeCrossCrate::No, coroutines, experimental!(coroutine)
    ),

    // RFC 3543
    // `#[patchable_function_entry(prefix_nops = m, entry_nops = n)]`
    gated!(
        patchable_function_entry, Normal, template!(List: "prefix_nops = m, entry_nops = n"), ErrorPreceding,
        EncodeCrossCrate::Yes, experimental!(patchable_function_entry)
    ),

    // Probably temporary component of min_generic_const_args.
    // `#[type_const] const ASSOC: usize;`
    gated!(
        type_const, Normal, template!(Word), ErrorFollowing,
        EncodeCrossCrate::Yes, min_generic_const_args, experimental!(type_const),
    ),

    // ==========================================================================
    // Internal attributes: Stability, deprecation, and unsafe:
    // ==========================================================================

    ungated!(
        feature, CrateLevel,
        template!(List: "name1, name2, ..."), DuplicatesOk, EncodeCrossCrate::No,
    ),
    // DuplicatesOk since it has its own validation
    ungated!(
        stable, Normal,
        template!(List: r#"feature = "name", since = "version""#), DuplicatesOk, EncodeCrossCrate::No,
    ),
    ungated!(
        unstable, Normal,
        template!(List: r#"feature = "name", reason = "...", issue = "N""#), DuplicatesOk,
        EncodeCrossCrate::Yes
    ),
    ungated!(
        rustc_const_unstable, Normal, template!(List: r#"feature = "name""#),
        DuplicatesOk, EncodeCrossCrate::Yes
    ),
    ungated!(
        rustc_const_stable, Normal,
        template!(List: r#"feature = "name""#), DuplicatesOk, EncodeCrossCrate::No,
    ),
    ungated!(
        rustc_default_body_unstable, Normal,
        template!(List: r#"feature = "name", reason = "...", issue = "N""#),
        DuplicatesOk, EncodeCrossCrate::No
    ),
    gated!(
        allow_internal_unstable, Normal, template!(Word, List: "feat1, feat2, ..."),
        DuplicatesOk, EncodeCrossCrate::Yes,
        "allow_internal_unstable side-steps feature gating and stability checks",
    ),
    gated!(
        allow_internal_unsafe, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No, "allow_internal_unsafe side-steps the unsafe_code lint",
    ),
    rustc_attr!(
        rustc_allowed_through_unstable_modules, Normal, template!(NameValueStr: "deprecation message"),
        WarnFollowing, EncodeCrossCrate::No,
        "rustc_allowed_through_unstable_modules special cases accidental stabilizations of stable items \
        through unstable paths"
    ),
    rustc_attr!(
        rustc_deprecated_safe_2024, Normal, template!(List: r#"audit_that = "...""#),
        ErrorFollowing, EncodeCrossCrate::Yes,
        "`#[rustc_deprecated_safe_2024]` is used to declare functions unsafe across the edition 2024 boundary",
    ),
    rustc_attr!(
        rustc_pub_transparent, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::Yes,
        "used internally to mark types with a `transparent` representation when it is guaranteed by the documentation",
    ),


    // ==========================================================================
    // Internal attributes: Type system related:
    // ==========================================================================

    gated!(fundamental, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::Yes, experimental!(fundamental)),
    gated!(
        may_dangle, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No, dropck_eyepatch,
        "`may_dangle` has unstable semantics and may be removed in the future",
    ),

    rustc_attr!(
        rustc_never_type_options,
        Normal,
        template!(List: r#"/*opt*/ fallback = "unit|niko|never|no""#),
        ErrorFollowing,
        EncodeCrossCrate::No,
        "`rustc_never_type_options` is used to experiment with never type fallback and work on \
         never type stabilization"
    ),

    // ==========================================================================
    // Internal attributes: Runtime related:
    // ==========================================================================

    rustc_attr!(
        rustc_allocator, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_nounwind, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_reallocator, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_deallocator, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_allocator_zeroed, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No,
    ),
    gated!(
        default_lib_allocator, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No, allocator_internals, experimental!(default_lib_allocator),
    ),
    gated!(
        needs_allocator, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No, allocator_internals, experimental!(needs_allocator),
    ),
    gated!(
        panic_runtime, CrateLevel, template!(Word), WarnFollowing,
        EncodeCrossCrate::No, experimental!(panic_runtime)
    ),
    gated!(
        needs_panic_runtime, CrateLevel, template!(Word), WarnFollowing,
        EncodeCrossCrate::No, experimental!(needs_panic_runtime)
    ),
    gated!(
        compiler_builtins, CrateLevel, template!(Word), WarnFollowing,
        EncodeCrossCrate::No,
        "the `#[compiler_builtins]` attribute is used to identify the `compiler_builtins` crate \
        which contains compiler-rt intrinsics and will never be stable",
    ),
    gated!(
        profiler_runtime, CrateLevel, template!(Word), WarnFollowing,
        EncodeCrossCrate::No,
        "the `#[profiler_runtime]` attribute is used to identify the `profiler_builtins` crate \
        which contains the profiler runtime and will never be stable",
    ),

    // ==========================================================================
    // Internal attributes, Linkage:
    // ==========================================================================

    gated!(
        linkage, Normal, template!(NameValueStr: "external|internal|..."),
        ErrorPreceding, EncodeCrossCrate::No,
        "the `linkage` attribute is experimental and not portable across platforms",
    ),
    rustc_attr!(
        rustc_std_internal_symbol, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No,
    ),

    // ==========================================================================
    // Internal attributes, Macro related:
    // ==========================================================================

    rustc_attr!(
        rustc_builtin_macro, Normal,
        template!(Word, List: "name, /*opt*/ attributes(name1, name2, ...)"), ErrorFollowing,
        EncodeCrossCrate::Yes,
    ),
    rustc_attr!(
        rustc_proc_macro_decls, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_macro_transparency, Normal,
        template!(NameValueStr: "transparent|semiopaque|opaque"), ErrorFollowing,
        EncodeCrossCrate::Yes, "used internally for testing macro hygiene",
    ),
    rustc_attr!(
        rustc_autodiff, Normal,
        template!(Word, List: r#""...""#), DuplicatesOk,
        EncodeCrossCrate::Yes,
    ),
    // Traces that are left when `cfg` and `cfg_attr` attributes are expanded.
    // The attributes are not gated, to avoid stability errors, but they cannot be used in stable
    // or unstable code directly because `sym::cfg_(attr_)trace` are not valid identifiers, they
    // can only be generated by the compiler.
    ungated!(
        cfg_trace, Normal, template!(Word /* irrelevant */), DuplicatesOk,
        EncodeCrossCrate::No
    ),
    ungated!(
        cfg_attr_trace, Normal, template!(Word /* irrelevant */), DuplicatesOk,
        EncodeCrossCrate::No
    ),

    // ==========================================================================
    // Internal attributes, Diagnostics related:
    // ==========================================================================

    rustc_attr!(
        rustc_on_unimplemented, Normal,
        template!(
            List: r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#,
            NameValueStr: "message"
        ),
        ErrorFollowing, EncodeCrossCrate::Yes,
        "see `#[diagnostic::on_unimplemented]` for the stable equivalent of this attribute"
    ),
    rustc_attr!(
        rustc_confusables, Normal,
        template!(List: r#""name1", "name2", ..."#),
        ErrorFollowing, EncodeCrossCrate::Yes,
    ),
    // Enumerates "identity-like" conversion methods to suggest on type mismatch.
    rustc_attr!(
        rustc_conversion_suggestion, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::Yes,
    ),
    // Prevents field reads in the marked trait or method to be considered
    // during dead code analysis.
    rustc_attr!(
        rustc_trivial_field_reads, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::Yes,
    ),
    // Used by the `rustc::potential_query_instability` lint to warn methods which
    // might not be stable during incremental compilation.
    rustc_attr!(
        rustc_lint_query_instability, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::Yes,
    ),
    // Used by the `rustc::untracked_query_information` lint to warn methods which
    // might not be stable during incremental compilation.
    rustc_attr!(
        rustc_lint_untracked_query_information, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::Yes,
    ),
    // Used by the `rustc::diagnostic_outside_of_impl` lints to assist in changes to diagnostic
    // APIs. Any function with this attribute will be checked by that lint.
    rustc_attr!(
        rustc_lint_diagnostics, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::Yes,
    ),
    // Used by the `rustc::bad_opt_access` lint to identify `DebuggingOptions` and `CodegenOptions`
    // types (as well as any others in future).
    rustc_attr!(
        rustc_lint_opt_ty, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::Yes,
    ),
    // Used by the `rustc::bad_opt_access` lint on fields
    // types (as well as any others in future).
    rustc_attr!(
        rustc_lint_opt_deny_field_access, Normal, template!(List: "message"),
        WarnFollowing, EncodeCrossCrate::Yes,
    ),

    // ==========================================================================
    // Internal attributes, Const related:
    // ==========================================================================

    rustc_attr!(
        rustc_promotable, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No, ),
    rustc_attr!(
        rustc_legacy_const_generics, Normal, template!(List: "N"), ErrorFollowing,
        EncodeCrossCrate::Yes,
    ),
    // Do not const-check this function's body. It will always get replaced during CTFE.
    rustc_attr!(
        rustc_do_not_const_check, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::Yes, "`#[rustc_do_not_const_check]` skips const-check for this function's body",
    ),
    rustc_attr!(
        rustc_const_panic_str, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::Yes, "`#[rustc_const_panic_str]` ensures the argument to this function is &&str during const-check",
    ),
    rustc_attr!(
        rustc_const_stable_indirect, Normal,
        template!(Word),
        WarnFollowing,
        EncodeCrossCrate::No,
        "this is an internal implementation detail",
    ),
    rustc_attr!(
        rustc_intrinsic_const_stable_indirect, Normal,
        template!(Word), WarnFollowing, EncodeCrossCrate::No,  "this is an internal implementation detail",
    ),
    gated!(
        rustc_allow_const_fn_unstable, Normal,
        template!(Word, List: "feat1, feat2, ..."), DuplicatesOk, EncodeCrossCrate::No,
        "rustc_allow_const_fn_unstable side-steps feature gating and stability checks"
    ),

    // ==========================================================================
    // Internal attributes, Layout related:
    // ==========================================================================

    rustc_attr!(
        rustc_layout_scalar_valid_range_start, Normal, template!(List: "value"), ErrorFollowing,
        EncodeCrossCrate::Yes,
        "the `#[rustc_layout_scalar_valid_range_start]` attribute is just used to enable \
        niche optimizations in the standard library",
    ),
    rustc_attr!(
        rustc_layout_scalar_valid_range_end, Normal, template!(List: "value"), ErrorFollowing,
        EncodeCrossCrate::Yes,
        "the `#[rustc_layout_scalar_valid_range_end]` attribute is just used to enable \
        niche optimizations in the standard library",
    ),
    rustc_attr!(
        rustc_nonnull_optimization_guaranteed, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::Yes,
        "the `#[rustc_nonnull_optimization_guaranteed]` attribute is just used to document \
        guaranteed niche optimizations in the standard library",
        "the compiler does not even check whether the type indeed is being non-null-optimized; \
        it is your responsibility to ensure that the attribute is only used on types that are optimized",
    ),

    // ==========================================================================
    // Internal attributes, Misc:
    // ==========================================================================
    gated!(
        lang, Normal, template!(NameValueStr: "name"), DuplicatesOk, EncodeCrossCrate::No, lang_items,
        "lang items are subject to change",
    ),
    rustc_attr!(
        rustc_as_ptr, Normal, template!(Word), ErrorFollowing,
        EncodeCrossCrate::Yes,
        "`#[rustc_as_ptr]` is used to mark functions returning pointers to their inner allocations."
    ),
    rustc_attr!(
        rustc_pass_by_value, Normal, template!(Word), ErrorFollowing,
        EncodeCrossCrate::Yes,
        "`#[rustc_pass_by_value]` is used to mark types that must be passed by value instead of reference."
    ),
    rustc_attr!(
        rustc_never_returns_null_ptr, Normal, template!(Word), ErrorFollowing,
        EncodeCrossCrate::Yes,
        "`#[rustc_never_returns_null_ptr]` is used to mark functions returning non-null pointers."
    ),
    rustc_attr!(
        rustc_no_implicit_autorefs, AttributeType::Normal, template!(Word), ErrorFollowing, EncodeCrossCrate::Yes,
        "`#[rustc_no_implicit_autorefs]` is used to mark functions for which an autoref to the dereference of a raw pointer should not be used as an argument."
    ),
    rustc_attr!(
        rustc_coherence_is_core, AttributeType::CrateLevel, template!(Word), ErrorFollowing, EncodeCrossCrate::No,
        "`#![rustc_coherence_is_core]` allows inherent methods on builtin types, only intended to be used in `core`."
    ),
    rustc_attr!(
        rustc_coinductive, AttributeType::Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No,
        "`#[rustc_coinductive]` changes a trait to be coinductive, allowing cycles in the trait solver."
    ),
    rustc_attr!(
        rustc_allow_incoherent_impl, AttributeType::Normal, template!(Word), ErrorFollowing, EncodeCrossCrate::No,
        "`#[rustc_allow_incoherent_impl]` has to be added to all impl items of an incoherent inherent impl."
    ),
    rustc_attr!(
        rustc_preserve_ub_checks, AttributeType::CrateLevel, template!(Word), ErrorFollowing, EncodeCrossCrate::No,
        "`#![rustc_preserve_ub_checks]` prevents the designated crate from evaluating whether UB checks are enabled when optimizing MIR",
    ),
    rustc_attr!(
        rustc_deny_explicit_impl,
        AttributeType::Normal,
        template!(Word),
        ErrorFollowing,
        EncodeCrossCrate::No,
        "`#[rustc_deny_explicit_impl]` enforces that a trait can have no user-provided impls"
    ),
    rustc_attr!(
        rustc_do_not_implement_via_object,
        AttributeType::Normal,
        template!(Word),
        ErrorFollowing,
        EncodeCrossCrate::No,
        "`#[rustc_do_not_implement_via_object]` opts out of the automatic trait impl for trait objects \
        (`impl Trait for dyn Trait`)"
    ),
    rustc_attr!(
        rustc_has_incoherent_inherent_impls, AttributeType::Normal, template!(Word),
        ErrorFollowing, EncodeCrossCrate::Yes,
        "`#[rustc_has_incoherent_inherent_impls]` allows the addition of incoherent inherent impls for \
         the given type by annotating all impl items with `#[rustc_allow_incoherent_impl]`."
    ),

    BuiltinAttribute {
        name: sym::rustc_diagnostic_item,
        // FIXME: This can be `true` once we always use `tcx.is_diagnostic_item`.
        encode_cross_crate: EncodeCrossCrate::Yes,
        type_: Normal,
        safety: AttributeSafety::Normal,
        template: template!(NameValueStr: "name"),
        duplicates: ErrorFollowing,
        gate: Gated{
            feature: sym::rustc_attrs,
            message: "use of an internal attribute",
            check: Features::rustc_attrs,
            notes: &["the `#[rustc_diagnostic_item]` attribute allows the compiler to reference types \
            from the standard library for diagnostic purposes"],
        },
    },
    gated!(
        // Used in resolve:
        prelude_import, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::No, "`#[prelude_import]` is for use by rustc only",
    ),
    gated!(
        rustc_paren_sugar, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No,
        unboxed_closures, "unboxed_closures are still evolving",
    ),
    rustc_attr!(
        rustc_inherit_overflow_checks, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No,
        "the `#[rustc_inherit_overflow_checks]` attribute is just used to control \
        overflow checking behavior of several functions in the standard library that are inlined \
        across crates",
    ),
    rustc_attr!(
        rustc_reservation_impl, Normal,
        template!(NameValueStr: "reservation message"), ErrorFollowing, EncodeCrossCrate::Yes,
        "the `#[rustc_reservation_impl]` attribute is internally used \
        for reserving `impl<T> From<!> for T` as part of the effort to stabilize `!`"
    ),
    rustc_attr!(
        rustc_test_marker, Normal, template!(NameValueStr: "name"), WarnFollowing,
        EncodeCrossCrate::No, "the `#[rustc_test_marker]` attribute is used internally to track tests",
    ),
    rustc_attr!(
        rustc_unsafe_specialization_marker, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No,
        "the `#[rustc_unsafe_specialization_marker]` attribute is used to check specializations"
    ),
    rustc_attr!(
        rustc_specialization_trait, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No,
        "the `#[rustc_specialization_trait]` attribute is used to check specializations"
    ),
    rustc_attr!(
        rustc_main, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::No,
        "the `#[rustc_main]` attribute is used internally to specify test entry point function",
    ),
    rustc_attr!(
        rustc_skip_during_method_dispatch, Normal, template!(List: "array, boxed_slice"), WarnFollowing,
        EncodeCrossCrate::No,
        "the `#[rustc_skip_during_method_dispatch]` attribute is used to exclude a trait \
        from method dispatch when the receiver is of the following type, for compatibility in \
        editions < 2021 (array) or editions < 2024 (boxed_slice)."
    ),
    rustc_attr!(
        rustc_must_implement_one_of, Normal, template!(List: "function1, function2, ..."),
        ErrorFollowing, EncodeCrossCrate::No,
        "the `#[rustc_must_implement_one_of]` attribute is used to change minimal complete \
        definition of a trait. Its syntax and semantics are highly experimental and will be \
        subject to change before stabilization",
    ),
    rustc_attr!(
        rustc_doc_primitive, Normal, template!(NameValueStr: "primitive name"), ErrorFollowing,
        EncodeCrossCrate::Yes, "the `#[rustc_doc_primitive]` attribute is used by the standard library \
        to provide a way to generate documentation for primitive types",
    ),
    gated!(
        rustc_intrinsic, Normal, template!(Word), ErrorFollowing, EncodeCrossCrate::Yes, intrinsics,
        "the `#[rustc_intrinsic]` attribute is used to declare intrinsics as function items",
    ),
    rustc_attr!(
        rustc_no_mir_inline, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::Yes,
        "`#[rustc_no_mir_inline]` prevents the MIR inliner from inlining a function while not affecting codegen"
    ),
    rustc_attr!(
        rustc_force_inline, Normal, template!(Word, NameValueStr: "reason"), WarnFollowing, EncodeCrossCrate::Yes,
        "`#[rustc_force_inline]` forces a free function to be inlined"
    ),

    // ==========================================================================
    // Internal attributes, Testing:
    // ==========================================================================

    rustc_attr!(TEST, rustc_effective_visibility, Normal, template!(Word), WarnFollowing, EncodeCrossCrate::Yes),
    rustc_attr!(
        TEST, rustc_outlives, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_capture_analysis, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_insignificant_dtor, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::Yes
    ),
    rustc_attr!(
        TEST, rustc_strict_coherence, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::Yes
    ),
    rustc_attr!(
        TEST, rustc_variance, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_variance_of_opaques, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_hidden_type_of_opaques, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_layout, Normal, template!(List: "field1, field2, ..."),
        WarnFollowing, EncodeCrossCrate::Yes
    ),
    rustc_attr!(
        TEST, rustc_abi, Normal, template!(List: "field1, field2, ..."),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_regions, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_delayed_bug_from_inside_query, Normal,
        template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_user_args, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_evaluate_where_clauses, Normal, template!(Word), WarnFollowing,
        EncodeCrossCrate::Yes
    ),
    rustc_attr!(
        TEST, rustc_if_this_changed, Normal, template!(Word, List: "DepNode"), DuplicatesOk,
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_then_this_would_need, Normal, template!(List: "DepNode"), DuplicatesOk,
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_clean, Normal,
        template!(List: r#"cfg = "...", /*opt*/ label = "...", /*opt*/ except = "...""#),
        DuplicatesOk, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_partition_reused, Normal,
        template!(List: r#"cfg = "...", module = "...""#), DuplicatesOk, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_partition_codegened, Normal,
        template!(List: r#"cfg = "...", module = "...""#), DuplicatesOk, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_expected_cgu_reuse, Normal,
        template!(List: r#"cfg = "...", module = "...", kind = "...""#), DuplicatesOk,
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_symbol_name, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_def_path, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_mir, Normal, template!(List: "arg1, arg2, ..."),
        DuplicatesOk, EncodeCrossCrate::Yes
    ),
    gated!(
        custom_mir, Normal, template!(List: r#"dialect = "...", phase = "...""#),
        ErrorFollowing, EncodeCrossCrate::No,
        "the `#[custom_mir]` attribute is just used for the Rust test suite",
    ),
    rustc_attr!(
        TEST, rustc_dump_item_bounds, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_predicates, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_def_parents, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_object_lifetime_default, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_vtable, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dummy, Normal, template!(Word /* doesn't matter*/),
        DuplicatesOk, EncodeCrossCrate::No
    ),
    gated!(
        omit_gdb_pretty_printer_section, Normal, template!(Word),
        WarnFollowing, EncodeCrossCrate::No,
        "the `#[omit_gdb_pretty_printer_section]` attribute is just used for the Rust test suite",
    ),
    rustc_attr!(
        TEST, pattern_complexity_limit, CrateLevel, template!(NameValueStr: "N"),
        ErrorFollowing, EncodeCrossCrate::No,
    ),
];

pub fn is_builtin_attr_name(name: Symbol) -> bool {
    BUILTIN_ATTRIBUTE_MAP.get(&name).is_some()
}

/// Whether this builtin attribute is encoded cross crate.
/// This means it can be used cross crate.
pub fn encode_cross_crate(name: Symbol) -> bool {
    if let Some(attr) = BUILTIN_ATTRIBUTE_MAP.get(&name) {
        attr.encode_cross_crate == EncodeCrossCrate::Yes
    } else {
        true
    }
}

pub fn is_valid_for_get_attr(name: Symbol) -> bool {
    BUILTIN_ATTRIBUTE_MAP.get(&name).is_some_and(|attr| match attr.duplicates {
        WarnFollowing | ErrorFollowing | ErrorPreceding | FutureWarnFollowing
        | FutureWarnPreceding => true,
        DuplicatesOk | WarnFollowingWordOnly => false,
    })
}

pub static BUILTIN_ATTRIBUTE_MAP: LazyLock<FxHashMap<Symbol, &BuiltinAttribute>> =
    LazyLock::new(|| {
        let mut map = FxHashMap::default();
        for attr in BUILTIN_ATTRIBUTES.iter() {
            if map.insert(attr.name, attr).is_some() {
                panic!("duplicate builtin attribute `{}`", attr.name);
            }
        }
        map
    });

pub fn is_stable_diagnostic_attribute(sym: Symbol, _features: &Features) -> bool {
    match sym {
        sym::on_unimplemented | sym::do_not_recommend => true,
        _ => false,
    }
}
