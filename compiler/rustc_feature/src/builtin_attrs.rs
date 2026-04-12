//! Built-in attributes and `cfg` flag gating.

use std::sync::LazyLock;

use AttributeGate::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::AttrStyle;
use rustc_hir::attrs::EncodeCrossCrate;
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
    (sym::target_object_format, sym::cfg_target_object_format, Features::cfg_target_object_format),
];

/// Find a gated cfg determined by the `pred`icate which is given the cfg's name.
pub fn find_gated_cfg(pred: impl Fn(Symbol) -> bool) -> Option<&'static GatedCfg> {
    GATED_CFGS.iter().find(|(cfg_sym, ..)| pred(*cfg_sym))
}

// If you change this, please modify `src/doc/unstable-book` as well. You must
// move that documentation into the relevant place in the other docs, and
// remove the chapter on the flag.

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

// FIXME(jdonszelmann): move to rustc_hir::attrs
/// A template that the attribute input must match.
/// Only top-level shape (`#[attr]` vs `#[attr(...)]` vs `#[attr = ...]`) is considered now.
#[derive(Clone, Copy, Default)]
pub struct AttributeTemplate {
    /// If `true`, the attribute is allowed to be a bare word like `#[test]`.
    pub word: bool,
    /// If `Some`, the attribute is allowed to take a list of items like `#[allow(..)]`.
    pub list: Option<&'static [&'static str]>,
    /// If non-empty, the attribute is allowed to take a list containing exactly
    /// one of the listed words, like `#[coverage(off)]`.
    pub one_of: &'static [Symbol],
    /// If `Some`, the attribute is allowed to be a name/value pair where the
    /// value is a string, like `#[must_use = "reason"]`.
    pub name_value_str: Option<&'static [&'static str]>,
    /// A link to the document for this attribute.
    pub docs: Option<&'static str>,
}

pub enum AttrSuggestionStyle {
    /// The suggestion is styled for a normal attribute.
    /// The `AttrStyle` determines whether this is an inner or outer attribute.
    Attribute(AttrStyle),
    /// The suggestion is styled for an attribute embedded into another attribute.
    /// For example, attributes inside `#[cfg_attr(true, attr(...)]`.
    EmbeddedAttribute,
    /// The suggestion is styled for macros that are parsed with attribute parsers.
    /// For example, the `cfg!(predicate)` macro.
    Macro,
}

impl AttributeTemplate {
    pub fn suggestions(
        &self,
        style: AttrSuggestionStyle,
        name: impl std::fmt::Display,
    ) -> Vec<String> {
        let (start, macro_call, end) = match style {
            AttrSuggestionStyle::Attribute(AttrStyle::Outer) => ("#[", "", "]"),
            AttrSuggestionStyle::Attribute(AttrStyle::Inner) => ("#![", "", "]"),
            AttrSuggestionStyle::Macro => ("", "!", ""),
            AttrSuggestionStyle::EmbeddedAttribute => ("", "", ""),
        };

        let mut suggestions = vec![];

        if self.word {
            debug_assert!(macro_call.is_empty(), "Macro suggestions use list style");
            suggestions.push(format!("{start}{name}{end}"));
        }
        if let Some(descr) = self.list {
            for descr in descr {
                suggestions.push(format!("{start}{name}{macro_call}({descr}){end}"));
            }
        }
        suggestions.extend(self.one_of.iter().map(|&word| format!("{start}{name}({word}){end}")));
        if let Some(descr) = self.name_value_str {
            debug_assert!(macro_call.is_empty(), "Macro suggestions use list style");
            for descr in descr {
                suggestions.push(format!("{start}{name} = \"{descr}\"{end}"));
            }
        }
        suggestions.sort();

        suggestions
    }
}

/// A convenience macro for constructing attribute templates.
/// E.g., `template!(Word, List: "description")` means that the attribute
/// supports forms `#[attr]` and `#[attr(description)]`.
#[macro_export]
macro_rules! template {
    (Word) => { $crate::template!(@ true, None, &[], None, None) };
    (Word, $link: literal) => { $crate::template!(@ true, None, &[], None, Some($link)) };
    (List: $descr: expr) => { $crate::template!(@ false, Some($descr), &[], None, None) };
    (List: $descr: expr, $link: literal) => { $crate::template!(@ false, Some($descr), &[], None, Some($link)) };
    (OneOf: $one_of: expr) => { $crate::template!(@ false, None, $one_of, None, None) };
    (NameValueStr: [$($descr: literal),* $(,)?]) => { $crate::template!(@ false, None, &[], Some(&[$($descr,)*]), None) };
    (NameValueStr: [$($descr: literal),* $(,)?], $link: literal) => { $crate::template!(@ false, None, &[], Some(&[$($descr,)*]), Some($link)) };
    (NameValueStr: $descr: literal) => { $crate::template!(@ false, None, &[], Some(&[$descr]), None) };
    (NameValueStr: $descr: literal, $link: literal) => { $crate::template!(@ false, None, &[], Some(&[$descr]), Some($link)) };
    (Word, List: $descr: expr) => { $crate::template!(@ true, Some($descr), &[], None, None) };
    (Word, List: $descr: expr, $link: literal) => { $crate::template!(@ true, Some($descr), &[], None, Some($link)) };
    (Word, NameValueStr: $descr: expr) => { $crate::template!(@ true, None, &[], Some(&[$descr]), None) };
    (Word, NameValueStr: $descr: expr, $link: literal) => { $crate::template!(@ true, None, &[], Some(&[$descr]), Some($link)) };
    (List: $descr1: expr, NameValueStr: $descr2: expr) => {
        $crate::template!(@ false, Some($descr1), &[], Some(&[$descr2]), None)
    };
    (List: $descr1: expr, NameValueStr: $descr2: expr, $link: literal) => {
        $crate::template!(@ false, Some($descr1), &[], Some(&[$descr2]), Some($link))
    };
    (Word, List: $descr1: expr, NameValueStr: $descr2: expr) => {
        $crate::template!(@ true, Some($descr1), &[], Some(&[$descr2]), None)
    };
    (Word, List: $descr1: expr, NameValueStr: $descr2: expr, $link: literal) => {
        $crate::template!(@ true, Some($descr1), &[], Some(&[$descr2]), Some($link))
    };
    (@ $word: expr, $list: expr, $one_of: expr, $name_value_str: expr, $link: expr) => { $crate::AttributeTemplate {
        word: $word, list: $list, one_of: $one_of, name_value_str: $name_value_str, docs: $link,
    } };
}

macro_rules! ungated {
    (unsafe($edition:ident) $attr:ident, $tpl:expr, $encode_cross_crate:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            safety: AttributeSafety::Unsafe { unsafe_since: Some(Edition::$edition) },
            template: $tpl,
            gate: Ungated,
        }
    };
    (unsafe $attr:ident, $tpl:expr, $encode_cross_crate:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            safety: AttributeSafety::Unsafe { unsafe_since: None },
            template: $tpl,
            gate: Ungated,
        }
    };
    ($attr:ident, $tpl:expr, $encode_cross_crate:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            safety: AttributeSafety::Normal,
            template: $tpl,
            gate: Ungated,
        }
    };
}

macro_rules! gated {
    (unsafe $attr:ident, $tpl:expr, $encode_cross_crate:expr, $gate:ident, $message:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            safety: AttributeSafety::Unsafe { unsafe_since: None },
            template: $tpl,
            gate: Gated {
                feature: sym::$gate,
                message: $message,
                check: Features::$gate,
                notes: &[],
            },
        }
    };
    (unsafe $attr:ident, $tpl:expr, $encode_cross_crate:expr, $message:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            safety: AttributeSafety::Unsafe { unsafe_since: None },
            template: $tpl,
            gate: Gated {
                feature: sym::$attr,
                message: $message,
                check: Features::$attr,
                notes: &[],
            },
        }
    };
    ($attr:ident, $tpl:expr, $encode_cross_crate:expr, $gate:ident, $message:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            safety: AttributeSafety::Normal,
            template: $tpl,
            gate: Gated {
                feature: sym::$gate,
                message: $message,
                check: Features::$gate,
                notes: &[],
            },
        }
    };
    ($attr:ident, $tpl:expr, $encode_cross_crate:expr, $message:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            safety: AttributeSafety::Normal,
            template: $tpl,
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
    (TEST, $attr:ident, $tpl:expr, $encode_cross_crate:expr $(,)?) => {
        rustc_attr!(
            $attr,
            $tpl,
            $encode_cross_crate,
            concat!(
                "the `#[",
                stringify!($attr),
                "]` attribute is used for rustc unit tests"
            ),
        )
    };
    ($attr:ident, $tpl:expr, $encode_cross_crate:expr, $($notes:expr),* $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            encode_cross_crate: $encode_cross_crate,
            safety: AttributeSafety::Normal,
            template: $tpl,
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

pub struct BuiltinAttribute {
    pub name: Symbol,
    /// Whether this attribute is encode cross crate.
    ///
    /// If so, it is encoded in the crate metadata.
    /// Otherwise, it can only be used in the local crate.
    pub encode_cross_crate: EncodeCrossCrate,
    pub safety: AttributeSafety,
    pub template: AttributeTemplate,
    pub gate: AttributeGate,
}

/// Attributes that have a special meaning to rustc or rustdoc.
#[rustfmt::skip]
pub static BUILTIN_ATTRIBUTES: &[BuiltinAttribute] = &[
    // ==========================================================================
    // Stable attributes:
    // ==========================================================================

    // Conditional compilation:
    ungated!(
        cfg,
        template!(
            List: &["predicate"],
            "https://doc.rust-lang.org/reference/conditional-compilation.html#the-cfg-attribute"
        ),
        EncodeCrossCrate::No
    ),
    ungated!(
        cfg_attr,
        template!(
            List: &["predicate, attr1, attr2, ..."],
            "https://doc.rust-lang.org/reference/conditional-compilation.html#the-cfg_attr-attribute"
        ),
        EncodeCrossCrate::No
    ),

    // Testing:
    ungated!(
        ignore,
        template!(
            Word,
            NameValueStr: "reason",
            "https://doc.rust-lang.org/reference/attributes/testing.html#the-ignore-attribute"
        ),
        EncodeCrossCrate::No,
    ),
    ungated!(
        should_panic,
        template!(
            Word,
            List: &[r#"expected = "reason""#],
            NameValueStr: "reason",
            "https://doc.rust-lang.org/reference/attributes/testing.html#the-should_panic-attribute"
        ),
        EncodeCrossCrate::No,
    ),

    // Macros:
    ungated!(
        automatically_derived,
        template!(
            Word,
            "https://doc.rust-lang.org/reference/attributes/derive.html#the-automatically_derived-attribute"
        ),
        EncodeCrossCrate::Yes
    ),
    ungated!(
        macro_use,
        template!(
            Word,
            List: &["name1, name2, ..."],
            "https://doc.rust-lang.org/reference/macros-by-example.html#the-macro_use-attribute"
        ),
        EncodeCrossCrate::No,
    ),
    ungated!(macro_escape, template!(Word), EncodeCrossCrate::No), // Deprecated synonym for `macro_use`.
    ungated!(
        macro_export,
        template!(
            Word,
            List: &["local_inner_macros"],
            "https://doc.rust-lang.org/reference/macros-by-example.html#path-based-scope"
        ),
        EncodeCrossCrate::Yes
    ),
    ungated!(
        proc_macro,
        template!(
            Word,
            "https://doc.rust-lang.org/reference/procedural-macros.html#function-like-procedural-macros"),
        EncodeCrossCrate::No
    ),
    ungated!(
        proc_macro_derive,
        template!(
            List: &["TraitName", "TraitName, attributes(name1, name2, ...)"],
            "https://doc.rust-lang.org/reference/procedural-macros.html#derive-macros"
        ),
        EncodeCrossCrate::No,
    ),
    ungated!(
        proc_macro_attribute,
        template!(Word, "https://doc.rust-lang.org/reference/procedural-macros.html#attribute-macros"),
        EncodeCrossCrate::No
    ),

    // Lints:
    ungated!(
        warn,
        template!(
            List: &["lint1", "lint1, lint2, ...", r#"lint1, lint2, lint3, reason = "...""#],
            "https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes"
        ),
        EncodeCrossCrate::No,
    ),
    ungated!(
        allow,
        template!(
            List: &["lint1", "lint1, lint2, ...", r#"lint1, lint2, lint3, reason = "...""#],
            "https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes"
        ),
        EncodeCrossCrate::No,
    ),
    ungated!(
        expect,
        template!(
            List: &["lint1", "lint1, lint2, ...", r#"lint1, lint2, lint3, reason = "...""#],
            "https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes"
        ),
        EncodeCrossCrate::No,
    ),
    ungated!(
        forbid,
        template!(
            List: &["lint1", "lint1, lint2, ...", r#"lint1, lint2, lint3, reason = "...""#],
            "https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes"
        ),
        EncodeCrossCrate::No
    ),
    ungated!(
        deny,
        template!(
            List: &["lint1", "lint1, lint2, ...", r#"lint1, lint2, lint3, reason = "...""#],
            "https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes"
        ),
        EncodeCrossCrate::No
    ),
    ungated!(
        must_use,
        template!(
            Word,
            NameValueStr: "reason",
            "https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute"
        ),
        EncodeCrossCrate::Yes
    ),
    gated!(
        must_not_suspend, template!(Word, NameValueStr: "reason"),
        EncodeCrossCrate::Yes, experimental!(must_not_suspend)
    ),
    ungated!(
        deprecated,
        template!(
            Word,
            List: &[r#"/*opt*/ since = "version", /*opt*/ note = "reason""#],
            NameValueStr: "reason",
            "https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-deprecated-attribute"
        ),
        EncodeCrossCrate::Yes
    ),

    // Crate properties:
    ungated!(
        crate_name,
        template!(
            NameValueStr: "name",
            "https://doc.rust-lang.org/reference/crates-and-source-files.html#the-crate_name-attribute"
        ),
        EncodeCrossCrate::No,
    ),
    ungated!(
        crate_type,
        template!(
            NameValueStr: ["bin", "lib", "dylib", "cdylib", "rlib", "staticlib", "sdylib", "proc-macro"],
            "https://doc.rust-lang.org/reference/linkage.html"
        ),
        EncodeCrossCrate::No,
    ),

    // ABI, linking, symbols, and FFI
    ungated!(
        link,
        template!(List: &[
            r#"name = "...""#,
            r#"name = "...", kind = "dylib|static|...""#,
            r#"name = "...", wasm_import_module = "...""#,
            r#"name = "...", import_name_type = "decorated|noprefix|undecorated""#,
            r#"name = "...", kind = "dylib|static|...", wasm_import_module = "...", import_name_type = "decorated|noprefix|undecorated""#,
        ], "https://doc.rust-lang.org/reference/items/external-blocks.html#the-link-attribute"),
        EncodeCrossCrate::No,
    ),
    ungated!(
        link_name,
        template!(NameValueStr: "name", "https://doc.rust-lang.org/reference/items/external-blocks.html#the-link_name-attribute"),
        EncodeCrossCrate::Yes
    ),
    ungated!(
        no_link,
        template!(Word, "https://doc.rust-lang.org/reference/items/extern-crates.html#the-no_link-attribute"),
        EncodeCrossCrate::No
    ),
    ungated!(
        repr,
        template!(
            List: &["C", "Rust", "transparent", "align(...)", "packed(...)", "<integer type>"],
            "https://doc.rust-lang.org/reference/type-layout.html#representations"
        ),
        EncodeCrossCrate::No
    ),
    // FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
    gated!(rustc_align, template!(List: &["alignment"]), EncodeCrossCrate::No, fn_align, experimental!(rustc_align)),
    gated!(rustc_align_static, template!(List: &["alignment"]), EncodeCrossCrate::No, static_align, experimental!(rustc_align_static)),
    ungated!(
        unsafe(Edition2024) export_name,
        template!(NameValueStr: "name", "https://doc.rust-lang.org/reference/abi.html#the-export_name-attribute"),
        EncodeCrossCrate::No
    ),
    ungated!(
        unsafe(Edition2024) link_section,
        template!(NameValueStr: "name", "https://doc.rust-lang.org/reference/abi.html#the-link_section-attribute"),
        EncodeCrossCrate::No
    ),
    ungated!(
        unsafe(Edition2024) no_mangle,
        template!(Word, "https://doc.rust-lang.org/reference/abi.html#the-no_mangle-attribute"),
        EncodeCrossCrate::No
    ),
    ungated!(
        used,
        template!(Word, List: &["compiler", "linker"], "https://doc.rust-lang.org/reference/abi.html#the-used-attribute"),
        EncodeCrossCrate::No
    ),
    ungated!(
        link_ordinal,
        template!(List: &["ordinal"], "https://doc.rust-lang.org/reference/items/external-blocks.html#the-link_ordinal-attribute"),
        EncodeCrossCrate::Yes
    ),
    ungated!(
        unsafe naked,
        template!(Word, "https://doc.rust-lang.org/reference/attributes/codegen.html#the-naked-attribute"),
        EncodeCrossCrate::No
    ),
    // See `TyAndLayout::pass_indirectly_in_non_rustic_abis` for details.
    rustc_attr!(
        rustc_pass_indirectly_in_non_rustic_abis, template!(Word),
        EncodeCrossCrate::No,
        "types marked with `#[rustc_pass_indirectly_in_non_rustic_abis]` are always passed indirectly by non-Rustic ABIs"
    ),

    // Limits:
    ungated!(
        recursion_limit,
        template!(NameValueStr: "N", "https://doc.rust-lang.org/reference/attributes/limits.html#the-recursion_limit-attribute"),
        EncodeCrossCrate::No
    ),
    ungated!(
        type_length_limit,
        template!(NameValueStr: "N", "https://doc.rust-lang.org/reference/attributes/limits.html#the-type_length_limit-attribute"),
        EncodeCrossCrate::No
    ),
    gated!(
        move_size_limit, template!(NameValueStr: "N"),
        EncodeCrossCrate::No, large_assignments, experimental!(move_size_limit)
    ),

    // Entry point:
    ungated!(
        no_main,
        template!(Word, "https://doc.rust-lang.org/reference/crates-and-source-files.html#the-no_main-attribute"),
        EncodeCrossCrate::No
    ),

    // Modules, prelude, and resolution:
    ungated!(
        path,
        template!(NameValueStr: "file", "https://doc.rust-lang.org/reference/items/modules.html#the-path-attribute"),
        EncodeCrossCrate::No
    ),
    ungated!(
        no_std,
        template!(Word, "https://doc.rust-lang.org/reference/names/preludes.html#the-no_std-attribute"),
        EncodeCrossCrate::No
    ),
    ungated!(
        no_implicit_prelude,
        template!(Word, "https://doc.rust-lang.org/reference/names/preludes.html#the-no_implicit_prelude-attribute"),
        EncodeCrossCrate::No
    ),
    ungated!(
        non_exhaustive,
        template!(Word, "https://doc.rust-lang.org/reference/attributes/type_system.html#the-non_exhaustive-attribute"),
        EncodeCrossCrate::Yes
    ),

    // Runtime
    ungated!(
        windows_subsystem,
        template!(NameValueStr: ["windows", "console"], "https://doc.rust-lang.org/reference/runtime.html#the-windows_subsystem-attribute"),
        EncodeCrossCrate::No
    ),
    ungated!( // RFC 2070
        panic_handler,
        template!(Word, "https://doc.rust-lang.org/reference/panic.html#the-panic_handler-attribute"),
        EncodeCrossCrate::Yes
    ),

    // Code generation:
    ungated!(
        inline,
        template!(
            Word,
            List: &["always", "never"],
            "https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute"
        ),
        EncodeCrossCrate::No
    ),
    ungated!(
        cold,
        template!(Word, "https://doc.rust-lang.org/reference/attributes/codegen.html#the-cold-attribute"),
        EncodeCrossCrate::No
    ),
    ungated!(
        no_builtins,
        template!(Word, "https://doc.rust-lang.org/reference/attributes/codegen.html#the-no_builtins-attribute"),
        EncodeCrossCrate::Yes
    ),
    ungated!(
        target_feature,
        template!(List: &[r#"enable = "name""#], "https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute"),
        EncodeCrossCrate::No,
    ),
    ungated!(
        track_caller,
        template!(Word, "https://doc.rust-lang.org/reference/attributes/codegen.html#the-track_caller-attribute"),
        EncodeCrossCrate::Yes
    ),
    ungated!(
        instruction_set,
        template!(List: &["set"], "https://doc.rust-lang.org/reference/attributes/codegen.html#the-instruction_set-attribute"),
        EncodeCrossCrate::No
    ),
    gated!(
        unsafe force_target_feature, template!(List: &[r#"enable = "name""#]),
        EncodeCrossCrate::No, effective_target_features, experimental!(force_target_feature)
    ),
    gated!(
        sanitize, template!(List: &[r#"address = "on|off""#, r#"kernel_address = "on|off""#, r#"cfi = "on|off""#, r#"hwaddress = "on|off""#, r#"kernel_hwaddress = "on|off""#, r#"kcfi = "on|off""#, r#"memory = "on|off""#, r#"memtag = "on|off""#, r#"shadow_call_stack = "on|off""#, r#"thread = "on|off""#]),
        EncodeCrossCrate::No, sanitize, experimental!(sanitize),
    ),
    gated!(
        coverage, template!(OneOf: &[sym::off, sym::on]),
        EncodeCrossCrate::No,
        coverage_attribute, experimental!(coverage)
    ),

    ungated!(
        doc,
        template!(
            List: &["hidden", "inline"],
            NameValueStr: "string",
            "https://doc.rust-lang.org/rustdoc/write-documentation/the-doc-attribute.html"
        ),
        EncodeCrossCrate::Yes
    ),

    // Debugging
    ungated!(
        debugger_visualizer,
        template!(
            List: &[r#"natvis_file = "...", gdb_script_file = "...""#],
            "https://doc.rust-lang.org/reference/attributes/debugger.html#the-debugger_visualizer-attribute"
        ),
        EncodeCrossCrate::No
    ),
    ungated!(
        collapse_debuginfo,
        template!(
            List: &["no", "external", "yes"],
            "https://doc.rust-lang.org/reference/attributes/debugger.html#the-collapse_debuginfo-attribute"
        ),
        EncodeCrossCrate::Yes
    ),

    // ==========================================================================
    // Unstable attributes:
    // ==========================================================================

    // Linking:
    gated!(
        export_stable, template!(Word),
        EncodeCrossCrate::No, experimental!(export_stable)
    ),

    // Testing:
    gated!(
        test_runner, template!(List: &["path"]),
        EncodeCrossCrate::Yes, custom_test_frameworks,
        "custom test frameworks are an unstable feature",
    ),

    gated!(
        reexport_test_harness_main, template!(NameValueStr: "name"),
        EncodeCrossCrate::No, custom_test_frameworks,
        "custom test frameworks are an unstable feature",
    ),

    // RFC #1268
    gated!(
        marker, template!(Word), EncodeCrossCrate::No,
        marker_trait_attr, experimental!(marker)
    ),
    gated!(
        thread_local, template!(Word), EncodeCrossCrate::No,
        "`#[thread_local]` is an experimental feature, and does not currently handle destructors",
    ),
    gated!(
        no_core, template!(Word),
        EncodeCrossCrate::No, experimental!(no_core)
    ),
    // RFC 2412
    gated!(
        optimize, template!(List: &["none", "size", "speed"]),
        EncodeCrossCrate::No, optimize_attribute, experimental!(optimize)
    ),

    gated!(
        unsafe ffi_pure, template!(Word),
        EncodeCrossCrate::No, experimental!(ffi_pure)
    ),
    gated!(
        unsafe ffi_const, template!(Word),
        EncodeCrossCrate::No, experimental!(ffi_const)
    ),
    gated!(
        register_tool, template!(List: &["tool1, tool2, ..."]),
        EncodeCrossCrate::No, experimental!(register_tool),
    ),
    // `#[cfi_encoding = ""]`
    gated!(
        cfi_encoding, template!(NameValueStr: "encoding"),
        EncodeCrossCrate::Yes, experimental!(cfi_encoding)
    ),

    // `#[coroutine]` attribute to be applied to closures to make them coroutines instead
    gated!(
        coroutine, template!(Word),
        EncodeCrossCrate::No, coroutines, experimental!(coroutine)
    ),

    // RFC 3543
    // `#[patchable_function_entry(prefix_nops = m, entry_nops = n)]`
    gated!(
        patchable_function_entry, template!(List: &["prefix_nops = m, entry_nops = n"]),
        EncodeCrossCrate::Yes, experimental!(patchable_function_entry)
    ),

    // The `#[loop_match]` and `#[const_continue]` attributes are part of the
    // lang experiment for RFC 3720 tracked in:
    //
    // - https://github.com/rust-lang/rust/issues/132306
    gated!(
        const_continue, template!(Word),
        EncodeCrossCrate::No, loop_match, experimental!(const_continue)
    ),
    gated!(
        loop_match, template!(Word),
        EncodeCrossCrate::No, loop_match, experimental!(loop_match)
    ),

    // The `#[pin_v2]` attribute is part of the `pin_ergonomics` experiment
    // that allows structurally pinning, tracked in:
    //
    // - https://github.com/rust-lang/rust/issues/130494
    gated!(
        pin_v2, template!(Word),
        EncodeCrossCrate::Yes, pin_ergonomics, experimental!(pin_v2),
    ),

    // ==========================================================================
    // Internal attributes: Stability, deprecation, and unsafe:
    // ==========================================================================

    ungated!(
        feature,
        template!(List: &["name1, name2, ..."]), EncodeCrossCrate::No,
    ),
    // DuplicatesOk since it has its own validation
    ungated!(
        stable,
        template!(List: &[r#"feature = "name", since = "version""#]), EncodeCrossCrate::No,
    ),
    ungated!(
        unstable,
        template!(List: &[r#"feature = "name", reason = "...", issue = "N""#]),
        EncodeCrossCrate::Yes
    ),
    ungated!(
        unstable_feature_bound, template!(Word, List: &["feat1, feat2, ..."]),
        EncodeCrossCrate::No,
    ),
    ungated!(
        unstable_removed, CrateLevel,
        template!(List: &[r#"feature = "name", reason = "...", link = "...", since = "version""#]),
        DuplicatesOk, EncodeCrossCrate::Yes
    ),
    ungated!(
        rustc_const_unstable, template!(List: &[r#"feature = "name""#]),
        EncodeCrossCrate::Yes
    ),
    ungated!(
        rustc_const_stable,
        template!(List: &[r#"feature = "name""#]), EncodeCrossCrate::No,
    ),
    ungated!(
        rustc_default_body_unstable,
        template!(List: &[r#"feature = "name", reason = "...", issue = "N""#]),
        EncodeCrossCrate::No
    ),
    gated!(
        allow_internal_unstable, template!(Word, List: &["feat1, feat2, ..."]),
        EncodeCrossCrate::Yes,
        "allow_internal_unstable side-steps feature gating and stability checks",
    ),
    gated!(
        allow_internal_unsafe, template!(Word),
        EncodeCrossCrate::No, "allow_internal_unsafe side-steps the unsafe_code lint",
    ),
    gated!(
        rustc_eii_foreign_item, template!(Word),
        EncodeCrossCrate::Yes, eii_internals,
        "used internally to mark types with a `transparent` representation when it is guaranteed by the documentation",
    ),
    rustc_attr!(
        rustc_allowed_through_unstable_modules, template!(NameValueStr: "deprecation message"),
        EncodeCrossCrate::No,
        "rustc_allowed_through_unstable_modules special cases accidental stabilizations of stable items \
        through unstable paths"
    ),
    rustc_attr!(
        rustc_deprecated_safe_2024, template!(List: &[r#"audit_that = "...""#]),
        EncodeCrossCrate::Yes,
        "`#[rustc_deprecated_safe_2024]` is used to declare functions unsafe across the edition 2024 boundary",
    ),
    rustc_attr!(
        rustc_pub_transparent, template!(Word),
        EncodeCrossCrate::Yes,
        "used internally to mark types with a `transparent` representation when it is guaranteed by the documentation",
    ),


    // ==========================================================================
    // Internal attributes: Type system related:
    // ==========================================================================

    gated!(fundamental, template!(Word), EncodeCrossCrate::Yes, experimental!(fundamental)),
    gated!(
        may_dangle, template!(Word),
        EncodeCrossCrate::No, dropck_eyepatch,
        "`may_dangle` has unstable semantics and may be removed in the future",
    ),

    rustc_attr!(
        rustc_never_type_options,
        template!(List: &[
            "",
            r#"fallback = "unit""#,
            r#"fallback = "niko""#,
            r#"fallback = "never""#,
            r#"fallback = "no""#,
        ]),
        EncodeCrossCrate::No,
        "`rustc_never_type_options` is used to experiment with never type fallback and work on \
         never type stabilization"
    ),

    // ==========================================================================
    // Internal attributes: Runtime related:
    // ==========================================================================

    rustc_attr!(
        rustc_allocator, template!(Word), EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_nounwind, template!(Word), EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_reallocator, template!(Word), EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_deallocator, template!(Word), EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_allocator_zeroed, template!(Word), EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_allocator_zeroed_variant, template!(NameValueStr: "function"),
        EncodeCrossCrate::Yes,
    ),
    gated!(
        default_lib_allocator, template!(Word),
        EncodeCrossCrate::No, allocator_internals, experimental!(default_lib_allocator),
    ),
    gated!(
        needs_allocator, template!(Word),
        EncodeCrossCrate::No, allocator_internals, experimental!(needs_allocator),
    ),
    gated!(
        panic_runtime, template!(Word),
        EncodeCrossCrate::No, experimental!(panic_runtime)
    ),
    gated!(
        needs_panic_runtime, template!(Word),
        EncodeCrossCrate::No, experimental!(needs_panic_runtime)
    ),
    gated!(
        compiler_builtins, template!(Word),
        EncodeCrossCrate::No,
        "the `#[compiler_builtins]` attribute is used to identify the `compiler_builtins` crate \
        which contains compiler-rt intrinsics and will never be stable",
    ),
    gated!(
        profiler_runtime, template!(Word),
        EncodeCrossCrate::No,
        "the `#[profiler_runtime]` attribute is used to identify the `profiler_builtins` crate \
        which contains the profiler runtime and will never be stable",
    ),

    // ==========================================================================
    // Internal attributes, Linkage:
    // ==========================================================================

    gated!(
        linkage, template!(NameValueStr: [
            "available_externally",
            "common",
            "extern_weak",
            "external",
            "internal",
            "linkonce",
            "linkonce_odr",
            "weak",
            "weak_odr",
        ], "https://doc.rust-lang.org/reference/linkage.html"),
        EncodeCrossCrate::No,
        "the `linkage` attribute is experimental and not portable across platforms",
    ),
    rustc_attr!(
        rustc_std_internal_symbol, template!(Word), EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_objc_class, template!(NameValueStr: "ClassName"), EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_objc_selector, template!(NameValueStr: "methodName"), EncodeCrossCrate::No,
    ),

    // ==========================================================================
    // Internal attributes, Macro related:
    // ==========================================================================

    rustc_attr!(
        rustc_builtin_macro,
        template!(Word, List: &["name", "name, /*opt*/ attributes(name1, name2, ...)"]),
        EncodeCrossCrate::Yes,
    ),
    rustc_attr!(
        rustc_proc_macro_decls, template!(Word),
        EncodeCrossCrate::No,
    ),
    rustc_attr!(
        rustc_macro_transparency,
        template!(NameValueStr: ["transparent", "semiopaque", "opaque"]),
        EncodeCrossCrate::Yes, "used internally for testing macro hygiene",
    ),
    rustc_attr!(
        rustc_autodiff,
        template!(Word, List: &[r#""...""#]),
        EncodeCrossCrate::Yes,
    ),
    rustc_attr!(
        rustc_offload_kernel,
        template!(Word), EncodeCrossCrate::Yes,
    ),
    // Traces that are left when `cfg` and `cfg_attr` attributes are expanded.
    // The attributes are not gated, to avoid stability errors, but they cannot be used in stable
    // or unstable code directly because `sym::cfg_(attr_)trace` are not valid identifiers, they
    // can only be generated by the compiler.
    ungated!(
        cfg_trace, template!(Word /* irrelevant */), EncodeCrossCrate::Yes
    ),
    ungated!(
        cfg_attr_trace, template!(Word /* irrelevant */), EncodeCrossCrate::No
    ),

    // ==========================================================================
    // Internal attributes, Diagnostics related:
    // ==========================================================================

    rustc_attr!(
        rustc_on_unimplemented,
        template!(
            List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#],
            NameValueStr: "message"
        ),
        EncodeCrossCrate::Yes,
        "see `#[diagnostic::on_unimplemented]` for the stable equivalent of this attribute"
    ),
    rustc_attr!(
        rustc_confusables,
        template!(List: &[r#""name1", "name2", ..."#]),
        EncodeCrossCrate::Yes,
    ),
    // Enumerates "identity-like" conversion methods to suggest on type mismatch.
    rustc_attr!(
        rustc_conversion_suggestion, template!(Word),
        EncodeCrossCrate::Yes,
    ),
    // Prevents field reads in the marked trait or method to be considered
    // during dead code analysis.
    rustc_attr!(
        rustc_trivial_field_reads, template!(Word),
        EncodeCrossCrate::Yes,
    ),
    // Used by the `rustc::potential_query_instability` lint to warn methods which
    // might not be stable during incremental compilation.
    rustc_attr!(
        rustc_lint_query_instability, template!(Word),
        EncodeCrossCrate::Yes,
    ),
    // Used by the `rustc::untracked_query_information` lint to warn methods which
    // might not be stable during incremental compilation.
    rustc_attr!(
        rustc_lint_untracked_query_information, template!(Word),
        EncodeCrossCrate::Yes,
    ),
    // Used by the `rustc::bad_opt_access` lint to identify `DebuggingOptions` and `CodegenOptions`
    // types (as well as any others in future).
    rustc_attr!(
        rustc_lint_opt_ty, template!(Word),
        EncodeCrossCrate::Yes,
    ),
    // Used by the `rustc::bad_opt_access` lint on fields
    // types (as well as any others in future).
    rustc_attr!(
        rustc_lint_opt_deny_field_access, template!(List: &["message"]),
        EncodeCrossCrate::Yes,
    ),

    // ==========================================================================
    // Internal attributes, Const related:
    // ==========================================================================

    rustc_attr!(
        rustc_promotable, template!(Word),
        EncodeCrossCrate::No, ),
    rustc_attr!(
        rustc_legacy_const_generics, template!(List: &["N"]),
        EncodeCrossCrate::Yes,
    ),
    // Do not const-check this function's body. It will always get replaced during CTFE via `hook_special_const_fn`.
    rustc_attr!(
        rustc_do_not_const_check, template!(Word),
        EncodeCrossCrate::Yes, "`#[rustc_do_not_const_check]` skips const-check for this function's body",
    ),
    rustc_attr!(
        rustc_const_stable_indirect,
        template!(Word),
       
        EncodeCrossCrate::No,
        "this is an internal implementation detail",
    ),
    rustc_attr!(
        rustc_intrinsic_const_stable_indirect,
        template!(Word), EncodeCrossCrate::No,  "this is an internal implementation detail",
    ),
    rustc_attr!(
        rustc_allow_const_fn_unstable,
        template!(Word, List: &["feat1, feat2, ..."]), EncodeCrossCrate::No,
        "rustc_allow_const_fn_unstable side-steps feature gating and stability checks"
    ),

    // ==========================================================================
    // Internal attributes, Layout related:
    // ==========================================================================

    rustc_attr!(
        rustc_layout_scalar_valid_range_start, template!(List: &["value"]),
        EncodeCrossCrate::Yes,
        "the `#[rustc_layout_scalar_valid_range_start]` attribute is just used to enable \
        niche optimizations in the standard library",
    ),
    rustc_attr!(
        rustc_layout_scalar_valid_range_end, template!(List: &["value"]),
        EncodeCrossCrate::Yes,
        "the `#[rustc_layout_scalar_valid_range_end]` attribute is just used to enable \
        niche optimizations in the standard library",
    ),
    rustc_attr!(
        rustc_simd_monomorphize_lane_limit, template!(NameValueStr: "N"),
        EncodeCrossCrate::Yes,
        "the `#[rustc_simd_monomorphize_lane_limit]` attribute is just used by std::simd \
        for better error messages",
    ),
    rustc_attr!(
        rustc_nonnull_optimization_guaranteed, template!(Word),
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
        lang, template!(NameValueStr: "name"), EncodeCrossCrate::No, lang_items,
        "lang items are subject to change",
    ),
    rustc_attr!(
        rustc_as_ptr, template!(Word),
        EncodeCrossCrate::Yes,
        "`#[rustc_as_ptr]` is used to mark functions returning pointers to their inner allocations"
    ),
    rustc_attr!(
        rustc_should_not_be_called_on_const_items, template!(Word),
        EncodeCrossCrate::Yes,
        "`#[rustc_should_not_be_called_on_const_items]` is used to mark methods that don't make sense to be called on interior mutable consts"
    ),
    rustc_attr!(
        rustc_pass_by_value, template!(Word),
        EncodeCrossCrate::Yes,
        "`#[rustc_pass_by_value]` is used to mark types that must be passed by value instead of reference"
    ),
    rustc_attr!(
        rustc_never_returns_null_ptr, template!(Word),
        EncodeCrossCrate::Yes,
        "`#[rustc_never_returns_null_ptr]` is used to mark functions returning non-null pointers"
    ),
    rustc_attr!(
        rustc_no_implicit_autorefs, template!(Word), EncodeCrossCrate::Yes,
        "`#[rustc_no_implicit_autorefs]` is used to mark functions for which an autoref to the dereference of a raw pointer should not be used as an argument"
    ),
    rustc_attr!(
        rustc_coherence_is_core, template!(Word), EncodeCrossCrate::No,
        "`#![rustc_coherence_is_core]` allows inherent methods on builtin types, only intended to be used in `core`"
    ),
    rustc_attr!(
        rustc_coinductive, template!(Word), EncodeCrossCrate::No,
        "`#[rustc_coinductive]` changes a trait to be coinductive, allowing cycles in the trait solver"
    ),
    rustc_attr!(
        rustc_allow_incoherent_impl, template!(Word), EncodeCrossCrate::No,
        "`#[rustc_allow_incoherent_impl]` has to be added to all impl items of an incoherent inherent impl"
    ),
    rustc_attr!(
        rustc_preserve_ub_checks, template!(Word), EncodeCrossCrate::No,
        "`#![rustc_preserve_ub_checks]` prevents the designated crate from evaluating whether UB checks are enabled when optimizing MIR",
    ),
    rustc_attr!(
        rustc_deny_explicit_impl,
        template!(Word),
        EncodeCrossCrate::No,
        "`#[rustc_deny_explicit_impl]` enforces that a trait can have no user-provided impls"
    ),
    rustc_attr!(
        rustc_dyn_incompatible_trait,
        template!(Word),
        EncodeCrossCrate::No,
        "`#[rustc_dyn_incompatible_trait]` marks a trait as dyn-incompatible, \
        even if it otherwise satisfies the requirements to be dyn-compatible."
    ),
    rustc_attr!(
        rustc_has_incoherent_inherent_impls, template!(Word),
        EncodeCrossCrate::Yes,
        "`#[rustc_has_incoherent_inherent_impls]` allows the addition of incoherent inherent impls for \
         the given type by annotating all impl items with `#[rustc_allow_incoherent_impl]`"
    ),
    rustc_attr!(
        rustc_non_const_trait_method, template!(Word),
        EncodeCrossCrate::No,
        "`#[rustc_non_const_trait_method]` should only used by the standard library to mark trait methods \
        as non-const to allow large traits an easier transition to const"
    ),

    BuiltinAttribute {
        name: sym::rustc_diagnostic_item,
        // FIXME: This can be `true` once we always use `tcx.is_diagnostic_item`.
        encode_cross_crate: EncodeCrossCrate::Yes,
        safety: AttributeSafety::Normal,
        template: template!(NameValueStr: "name"),
        gate: Gated {
            feature: sym::rustc_attrs,
            message: "use of an internal attribute",
            check: Features::rustc_attrs,
            notes: &["the `#[rustc_diagnostic_item]` attribute allows the compiler to reference types \
            from the standard library for diagnostic purposes"],
        },
    },
    gated!(
        // Used in resolve:
        prelude_import, template!(Word),
        EncodeCrossCrate::No, "`#[prelude_import]` is for use by rustc only",
    ),
    gated!(
        rustc_paren_sugar, template!(Word), EncodeCrossCrate::No,
        unboxed_closures, "unboxed_closures are still evolving",
    ),
    rustc_attr!(
        rustc_inherit_overflow_checks, template!(Word), EncodeCrossCrate::No,
        "the `#[rustc_inherit_overflow_checks]` attribute is just used to control \
        overflow checking behavior of several functions in the standard library that are inlined \
        across crates",
    ),
    rustc_attr!(
        rustc_reservation_impl,
        template!(NameValueStr: "reservation message"), EncodeCrossCrate::Yes,
        "the `#[rustc_reservation_impl]` attribute is internally used \
        for reserving `impl<T> From<!> for T` as part of the effort to stabilize `!`"
    ),
    rustc_attr!(
        rustc_test_marker, template!(NameValueStr: "name"),
        EncodeCrossCrate::No, "the `#[rustc_test_marker]` attribute is used internally to track tests",
    ),
    rustc_attr!(
        rustc_unsafe_specialization_marker, template!(Word),
        EncodeCrossCrate::No,
        "the `#[rustc_unsafe_specialization_marker]` attribute is used to check specializations"
    ),
    rustc_attr!(
        rustc_specialization_trait, template!(Word),
        EncodeCrossCrate::No,
        "the `#[rustc_specialization_trait]` attribute is used to check specializations"
    ),
    rustc_attr!(
        rustc_main, template!(Word), EncodeCrossCrate::No,
        "the `#[rustc_main]` attribute is used internally to specify test entry point function",
    ),
    rustc_attr!(
        rustc_skip_during_method_dispatch, template!(List: &["array, boxed_slice"]),
        EncodeCrossCrate::No,
        "the `#[rustc_skip_during_method_dispatch]` attribute is used to exclude a trait \
        from method dispatch when the receiver is of the following type, for compatibility in \
        editions < 2021 (array) or editions < 2024 (boxed_slice)"
    ),
    rustc_attr!(
        rustc_must_implement_one_of, template!(List: &["function1, function2, ..."]),
        EncodeCrossCrate::No,
        "the `#[rustc_must_implement_one_of]` attribute is used to change minimal complete \
        definition of a trait. Its syntax and semantics are highly experimental and will be \
        subject to change before stabilization",
    ),
    rustc_attr!(
        rustc_doc_primitive, template!(NameValueStr: "primitive name"),
        EncodeCrossCrate::Yes, "the `#[rustc_doc_primitive]` attribute is used by the standard library \
        to provide a way to generate documentation for primitive types",
    ),
    gated!(
        rustc_intrinsic, template!(Word), EncodeCrossCrate::Yes, intrinsics,
        "the `#[rustc_intrinsic]` attribute is used to declare intrinsics as function items",
    ),
    rustc_attr!(
        rustc_no_mir_inline, template!(Word), EncodeCrossCrate::Yes,
        "`#[rustc_no_mir_inline]` prevents the MIR inliner from inlining a function while not affecting codegen"
    ),
    rustc_attr!(
        rustc_force_inline, template!(Word, NameValueStr: "reason"), EncodeCrossCrate::Yes,
        "`#[rustc_force_inline]` forces a free function to be inlined"
    ),
    rustc_attr!(
        rustc_scalable_vector, template!(List: &["count"]), EncodeCrossCrate::Yes,
        "`#[rustc_scalable_vector]` defines a scalable vector type"
    ),
    rustc_attr!(
        rustc_must_match_exhaustively, template!(Word), EncodeCrossCrate::Yes,
        "enums with `#[rustc_must_match_exhaustively]` must be matched on with a match block that mentions all variants explicitly"
    ),

    // ==========================================================================
    // Internal attributes, Testing:
    // ==========================================================================

    rustc_attr!(TEST, rustc_effective_visibility, template!(Word), EncodeCrossCrate::Yes),
    rustc_attr!(
        TEST, rustc_dump_inferred_outlives, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_capture_analysis, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_insignificant_dtor, template!(Word),
        EncodeCrossCrate::Yes
    ),
    rustc_attr!(
        TEST, rustc_no_implicit_bounds, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_strict_coherence, template!(Word),
        EncodeCrossCrate::Yes
    ),
    rustc_attr!(
        TEST, rustc_dump_variances, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_variances_of_opaques, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_hidden_type_of_opaques, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_layout, template!(List: &["field1, field2, ..."]),
        EncodeCrossCrate::Yes
    ),
    rustc_attr!(
        TEST, rustc_abi, template!(List: &["field1, field2, ..."]),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_regions, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_delayed_bug_from_inside_query,
        template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_user_args, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_evaluate_where_clauses, template!(Word),
        EncodeCrossCrate::Yes
    ),
    rustc_attr!(
        TEST, rustc_if_this_changed, template!(Word, List: &["DepNode"]),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_then_this_would_need, template!(List: &["DepNode"]),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_clean,
        template!(List: &[r#"cfg = "...", /*opt*/ label = "...", /*opt*/ except = "...""#]),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_partition_reused,
        template!(List: &[r#"cfg = "...", module = "...""#]), EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_partition_codegened,
        template!(List: &[r#"cfg = "...", module = "...""#]), EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_expected_cgu_reuse,
        template!(List: &[r#"cfg = "...", module = "...", kind = "...""#]),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_symbol_name, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_def_path, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_mir, template!(List: &["arg1, arg2, ..."]),
        EncodeCrossCrate::Yes
    ),
    gated!(
        custom_mir, template!(List: &[r#"dialect = "...", phase = "...""#]),
        EncodeCrossCrate::No,
        "the `#[custom_mir]` attribute is just used for the Rust test suite",
    ),
    rustc_attr!(
        TEST, rustc_dump_item_bounds, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_predicates, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_def_parents, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_object_lifetime_defaults, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dump_vtable, template!(Word),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, rustc_dummy, template!(Word /* doesn't matter*/),
        EncodeCrossCrate::No
    ),
    rustc_attr!(
        TEST, pattern_complexity_limit, template!(NameValueStr: "N"),
        EncodeCrossCrate::No,
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

pub fn is_stable_diagnostic_attribute(sym: Symbol, features: &Features) -> bool {
    match sym {
        sym::on_unimplemented | sym::do_not_recommend => true,
        sym::on_const => features.diagnostic_on_const(),
        sym::on_move => features.diagnostic_on_move(),
        sym::on_unknown => features.diagnostic_on_unknown(),
        _ => false,
    }
}
