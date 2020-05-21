//! Built-in attributes and `cfg` flag gating.

use AttributeGate::*;
use AttributeType::*;

use crate::{Features, Stability};

use lazy_static::lazy_static;
use rustc_data_structures::fx::FxHashMap;
use rustc_span::symbol::{sym, Symbol};

type GateFn = fn(&Features) -> bool;

macro_rules! cfg_fn {
    ($field: ident) => {
        (|features| features.$field) as GateFn
    };
}

pub type GatedCfg = (Symbol, Symbol, GateFn);

/// `cfg(...)`'s that are feature gated.
const GATED_CFGS: &[GatedCfg] = &[
    // (name in cfg, feature, function to check if the feature is enabled)
    (sym::target_thread_local, sym::cfg_target_thread_local, cfg_fn!(cfg_target_thread_local)),
    (sym::target_has_atomic, sym::cfg_target_has_atomic, cfg_fn!(cfg_target_has_atomic)),
    (sym::target_has_atomic_load_store, sym::cfg_target_has_atomic, cfg_fn!(cfg_target_has_atomic)),
    (sym::sanitize, sym::cfg_sanitize, cfg_fn!(cfg_sanitize)),
    (sym::version, sym::cfg_version, cfg_fn!(cfg_version)),
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

    /// Builtin attribute that may not be consumed by the compiler
    /// before the unused_attribute check. These attributes
    /// will be ignored by the unused_attribute lint
    Whitelisted,

    /// Builtin attribute that is only allowed at the crate level
    CrateLevel,
}

#[derive(Clone, Copy)]
pub enum AttributeGate {
    /// Is gated by a given feature gate, reason
    /// and function to check if enabled
    Gated(Stability, Symbol, &'static str, fn(&Features) -> bool),

    /// Ungated attribute, can be used on all release channels
    Ungated,
}

// fn() is not Debug
impl std::fmt::Debug for AttributeGate {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Gated(ref stab, name, expl, _) => {
                write!(fmt, "Gated({:?}, {}, {})", stab, name, expl)
            }
            Self::Ungated => write!(fmt, "Ungated"),
        }
    }
}

impl AttributeGate {
    fn is_deprecated(&self) -> bool {
        match *self {
            Self::Gated(Stability::Deprecated(_, _), ..) => true,
            _ => false,
        }
    }
}

/// A template that the attribute input must match.
/// Only top-level shape (`#[attr]` vs `#[attr(...)]` vs `#[attr = ...]`) is considered now.
#[derive(Clone, Copy, Default)]
pub struct AttributeTemplate {
    pub word: bool,
    pub list: Option<&'static str>,
    pub name_value_str: Option<&'static str>,
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

macro_rules! ungated {
    ($attr:ident, $typ:expr, $tpl:expr $(,)?) => {
        (sym::$attr, $typ, $tpl, Ungated)
    };
}

macro_rules! gated {
    ($attr:ident, $typ:expr, $tpl:expr, $gate:ident, $msg:expr $(,)?) => {
        (sym::$attr, $typ, $tpl, Gated(Stability::Unstable, sym::$gate, $msg, cfg_fn!($gate)))
    };
    ($attr:ident, $typ:expr, $tpl:expr, $msg:expr $(,)?) => {
        (sym::$attr, $typ, $tpl, Gated(Stability::Unstable, sym::$attr, $msg, cfg_fn!($attr)))
    };
}

macro_rules! rustc_attr {
    (TEST, $attr:ident, $typ:expr, $tpl:expr $(,)?) => {
        rustc_attr!(
            $attr,
            $typ,
            $tpl,
            concat!(
                "the `#[",
                stringify!($attr),
                "]` attribute is just used for rustc unit tests \
                and will never be stable",
            ),
        )
    };
    ($attr:ident, $typ:expr, $tpl:expr, $msg:expr $(,)?) => {
        (
            sym::$attr,
            $typ,
            $tpl,
            Gated(Stability::Unstable, sym::rustc_attrs, $msg, cfg_fn!(rustc_attrs)),
        )
    };
}

macro_rules! experimental {
    ($attr:ident) => {
        concat!("the `#[", stringify!($attr), "]` attribute is an experimental feature")
    };
}

const IMPL_DETAIL: &str = "internal implementation detail";
const INTERNAL_UNSTABLE: &str = "this is an internal attribute that will never be stable";

pub type BuiltinAttribute = (Symbol, AttributeType, AttributeTemplate, AttributeGate);

/// Attributes that have a special meaning to rustc or rustdoc.
#[rustfmt::skip]
pub const BUILTIN_ATTRIBUTES: &[BuiltinAttribute] = &[
    // ==========================================================================
    // Stable attributes:
    // ==========================================================================

    // Conditional compilation:
    ungated!(cfg, Normal, template!(List: "predicate")),
    ungated!(cfg_attr, Normal, template!(List: "predicate, attr1, attr2, ...")),

    // Testing:
    ungated!(ignore, Normal, template!(Word, NameValueStr: "reason")),
    ungated!(
        should_panic, Normal,
        template!(Word, List: r#"expected = "reason"#, NameValueStr: "reason"),
    ),
    // FIXME(Centril): This can be used on stable but shouldn't.
    ungated!(reexport_test_harness_main, Normal, template!(NameValueStr: "name")),

    // Macros:
    ungated!(derive, Normal, template!(List: "Trait1, Trait2, ...")),
    ungated!(automatically_derived, Normal, template!(Word)),
    // FIXME(#14407)
    ungated!(macro_use, Normal, template!(Word, List: "name1, name2, ...")),
    ungated!(macro_escape, Normal, template!(Word)), // Deprecated synonym for `macro_use`.
    ungated!(macro_export, Normal, template!(Word, List: "local_inner_macros")),
    ungated!(proc_macro, Normal, template!(Word)),
    ungated!(
        proc_macro_derive, Normal,
        template!(List: "TraitName, /*opt*/ attributes(name1, name2, ...)"),
    ),
    ungated!(proc_macro_attribute, Normal, template!(Word)),

    // Lints:
    ungated!(warn, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#)),
    ungated!(allow, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#)),
    ungated!(forbid, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#)),
    ungated!(deny, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#)),
    ungated!(must_use, Whitelisted, template!(Word, NameValueStr: "reason")),
    // FIXME(#14407)
    ungated!(
        deprecated, Normal,
        template!(
            Word,
            List: r#"/*opt*/ since = "version", /*opt*/ note = "reason""#,
            NameValueStr: "reason"
        ),
    ),

    // Crate properties:
    ungated!(crate_name, CrateLevel, template!(NameValueStr: "name")),
    ungated!(crate_type, CrateLevel, template!(NameValueStr: "bin|lib|...")),
    ungated!(crate_id, CrateLevel, template!(NameValueStr: "ignored")),

    // ABI, linking, symbols, and FFI
    ungated!(
        link, Whitelisted,
        template!(List: r#"name = "...", /*opt*/ kind = "dylib|static|...", /*opt*/ wasm_import_module = "...""#),
    ),
    ungated!(link_name, Whitelisted, template!(NameValueStr: "name")),
    ungated!(no_link, Normal, template!(Word)),
    ungated!(repr, Normal, template!(List: "C")),
    ungated!(export_name, Whitelisted, template!(NameValueStr: "name")),
    ungated!(link_section, Whitelisted, template!(NameValueStr: "name")),
    ungated!(no_mangle, Whitelisted, template!(Word)),
    ungated!(used, Whitelisted, template!(Word)),

    // Limits:
    ungated!(recursion_limit, CrateLevel, template!(NameValueStr: "N")),
    ungated!(type_length_limit, CrateLevel, template!(NameValueStr: "N")),
    gated!(
        const_eval_limit, CrateLevel, template!(NameValueStr: "N"), const_eval_limit,
        experimental!(const_eval_limit)
    ),

    // Entry point:
    ungated!(main, Normal, template!(Word)),
    ungated!(start, Normal, template!(Word)),
    ungated!(no_start, CrateLevel, template!(Word)),
    ungated!(no_main, CrateLevel, template!(Word)),

    // Modules, prelude, and resolution:
    ungated!(path, Normal, template!(NameValueStr: "file")),
    ungated!(no_std, CrateLevel, template!(Word)),
    ungated!(no_implicit_prelude, Normal, template!(Word)),
    ungated!(non_exhaustive, Whitelisted, template!(Word)),

    // Runtime
    ungated!(windows_subsystem, Whitelisted, template!(NameValueStr: "windows|console")),
    ungated!(panic_handler, Normal, template!(Word)), // RFC 2070

    // Code generation:
    ungated!(inline, Whitelisted, template!(Word, List: "always|never")),
    ungated!(cold, Whitelisted, template!(Word)),
    ungated!(no_builtins, Whitelisted, template!(Word)),
    ungated!(target_feature, Whitelisted, template!(List: r#"enable = "name""#)),
    gated!(
        no_sanitize, Whitelisted,
        template!(List: "address, memory, thread"),
        experimental!(no_sanitize)
    ),

    // FIXME: #14408 whitelist docs since rustdoc looks at them
    ungated!(doc, Whitelisted, template!(List: "hidden|inline|...", NameValueStr: "string")),

    // ==========================================================================
    // Unstable attributes:
    // ==========================================================================

    // Linking:
    gated!(naked, Whitelisted, template!(Word), naked_functions, experimental!(naked)),
    gated!(
        link_args, Normal, template!(NameValueStr: "args"),
        "the `link_args` attribute is experimental and not portable across platforms, \
        it is recommended to use `#[link(name = \"foo\")] instead",
    ),
    gated!(
        link_ordinal, Whitelisted, template!(List: "ordinal"), raw_dylib,
        experimental!(link_ordinal)
    ),

    // Plugins:
    (
        sym::plugin_registrar, Normal, template!(Word),
        Gated(
            Stability::Deprecated(
                "https://github.com/rust-lang/rust/pull/64675",
                Some("may be removed in a future compiler version"),
            ),
            sym::plugin_registrar,
            "compiler plugins are deprecated",
            cfg_fn!(plugin_registrar)
        )
    ),
    (
        sym::plugin, CrateLevel, template!(List: "name"),
        Gated(
            Stability::Deprecated(
                "https://github.com/rust-lang/rust/pull/64675",
                Some("may be removed in a future compiler version"),
            ),
            sym::plugin,
            "compiler plugins are deprecated",
            cfg_fn!(plugin)
        )
    ),

    // Testing:
    gated!(allow_fail, Normal, template!(Word), experimental!(allow_fail)),
    gated!(
        test_runner, CrateLevel, template!(List: "path"), custom_test_frameworks,
        "custom test frameworks are an unstable feature",
    ),
    // RFC #1268
    gated!(marker, Normal, template!(Word), marker_trait_attr, experimental!(marker)),
    gated!(
        thread_local, Whitelisted, template!(Word),
        "`#[thread_local]` is an experimental feature, and does not currently handle destructors",
    ),
    gated!(no_core, CrateLevel, template!(Word), experimental!(no_core)),
    // RFC 2412
    gated!(
        optimize, Whitelisted, template!(List: "size|speed"), optimize_attribute,
        experimental!(optimize),
    ),

    gated!(ffi_returns_twice, Whitelisted, template!(Word), experimental!(ffi_returns_twice)),
    gated!(ffi_pure, Whitelisted, template!(Word), experimental!(ffi_pure)),
    gated!(ffi_const, Whitelisted, template!(Word), experimental!(ffi_const)),
    gated!(track_caller, Whitelisted, template!(Word), experimental!(track_caller)),
    gated!(
        register_attr, CrateLevel, template!(List: "attr1, attr2, ..."),
        experimental!(register_attr),
    ),
    gated!(
        register_tool, CrateLevel, template!(List: "tool1, tool2, ..."),
        experimental!(register_tool),
    ),

    // ==========================================================================
    // Internal attributes: Stability, deprecation, and unsafe:
    // ==========================================================================

    ungated!(feature, CrateLevel, template!(List: "name1, name1, ...")),
    // FIXME(#14407) -- only looked at on-demand so we can't
    // guarantee they'll have already been checked.
    ungated!(
        rustc_deprecated, Whitelisted,
        template!(List: r#"since = "version", reason = "...""#)
    ),
    // FIXME(#14407)
    ungated!(stable, Whitelisted, template!(List: r#"feature = "name", since = "version""#)),
    // FIXME(#14407)
    ungated!(
        unstable, Whitelisted,
        template!(List: r#"feature = "name", reason = "...", issue = "N""#),
    ),
    // FIXME(#14407)
    ungated!(rustc_const_unstable, Whitelisted, template!(List: r#"feature = "name""#)),
    // FIXME(#14407)
    ungated!(rustc_const_stable, Whitelisted, template!(List: r#"feature = "name""#)),
    gated!(
        allow_internal_unstable, Normal, template!(Word, List: "feat1, feat2, ..."),
        "allow_internal_unstable side-steps feature gating and stability checks",
    ),
    gated!(
        allow_internal_unsafe, Normal, template!(Word),
        "allow_internal_unsafe side-steps the unsafe_code lint",
    ),

    // ==========================================================================
    // Internal attributes: Type system related:
    // ==========================================================================

    gated!(fundamental, Whitelisted, template!(Word), experimental!(fundamental)),
    gated!(
        may_dangle, Normal, template!(Word), dropck_eyepatch,
        "`may_dangle` has unstable semantics and may be removed in the future",
    ),

    // ==========================================================================
    // Internal attributes: Runtime related:
    // ==========================================================================

    rustc_attr!(rustc_allocator, Whitelisted, template!(Word), IMPL_DETAIL),
    rustc_attr!(rustc_allocator_nounwind, Whitelisted, template!(Word), IMPL_DETAIL),
    gated!(alloc_error_handler, Normal, template!(Word), experimental!(alloc_error_handler)),
    gated!(
        default_lib_allocator, Whitelisted, template!(Word), allocator_internals,
        experimental!(default_lib_allocator),
    ),
    gated!(
        needs_allocator, Normal, template!(Word), allocator_internals,
        experimental!(needs_allocator),
    ),
    gated!(panic_runtime, Whitelisted, template!(Word), experimental!(panic_runtime)),
    gated!(needs_panic_runtime, Whitelisted, template!(Word), experimental!(needs_panic_runtime)),
    gated!(
        unwind, Whitelisted, template!(List: "allowed|aborts"), unwind_attributes,
        experimental!(unwind),
    ),
    gated!(
        compiler_builtins, Whitelisted, template!(Word),
        "the `#[compiler_builtins]` attribute is used to identify the `compiler_builtins` crate \
        which contains compiler-rt intrinsics and will never be stable",
    ),
    gated!(
        profiler_runtime, Whitelisted, template!(Word),
        "the `#[profiler_runtime]` attribute is used to identify the `profiler_builtins` crate \
        which contains the profiler runtime and will never be stable",
    ),

    // ==========================================================================
    // Internal attributes, Linkage:
    // ==========================================================================

    gated!(
        linkage, Whitelisted, template!(NameValueStr: "external|internal|..."),
        "the `linkage` attribute is experimental and not portable across platforms",
    ),
    rustc_attr!(rustc_std_internal_symbol, Whitelisted, template!(Word), INTERNAL_UNSTABLE),

    // ==========================================================================
    // Internal attributes, Macro related:
    // ==========================================================================

    rustc_attr!(rustc_builtin_macro, Whitelisted, template!(Word), IMPL_DETAIL),
    rustc_attr!(rustc_proc_macro_decls, Normal, template!(Word), INTERNAL_UNSTABLE),
    rustc_attr!(
        rustc_macro_transparency, Whitelisted,
        template!(NameValueStr: "transparent|semitransparent|opaque"),
        "used internally for testing macro hygiene",
    ),

    // ==========================================================================
    // Internal attributes, Diagnostics related:
    // ==========================================================================

    rustc_attr!(
        rustc_on_unimplemented, Whitelisted,
        template!(
            List: r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#,
            NameValueStr: "message"
        ),
        INTERNAL_UNSTABLE
    ),
    // Whitelists "identity-like" conversion methods to suggest on type mismatch.
    rustc_attr!(rustc_conversion_suggestion, Whitelisted, template!(Word), INTERNAL_UNSTABLE),

    // ==========================================================================
    // Internal attributes, Const related:
    // ==========================================================================

    rustc_attr!(rustc_promotable, Whitelisted, template!(Word), IMPL_DETAIL),
    rustc_attr!(rustc_allow_const_fn_ptr, Whitelisted, template!(Word), IMPL_DETAIL),
    rustc_attr!(rustc_args_required_const, Whitelisted, template!(List: "N"), INTERNAL_UNSTABLE),

    // ==========================================================================
    // Internal attributes, Layout related:
    // ==========================================================================

    rustc_attr!(
        rustc_layout_scalar_valid_range_start, Whitelisted, template!(List: "value"),
        "the `#[rustc_layout_scalar_valid_range_start]` attribute is just used to enable \
        niche optimizations in libcore and will never be stable",
    ),
    rustc_attr!(
        rustc_layout_scalar_valid_range_end, Whitelisted, template!(List: "value"),
        "the `#[rustc_layout_scalar_valid_range_end]` attribute is just used to enable \
        niche optimizations in libcore and will never be stable",
    ),
    rustc_attr!(
        rustc_nonnull_optimization_guaranteed, Whitelisted, template!(Word),
        "the `#[rustc_nonnull_optimization_guaranteed]` attribute is just used to enable \
        niche optimizations in libcore and will never be stable",
    ),

    // ==========================================================================
    // Internal attributes, Misc:
    // ==========================================================================
    gated!(
        lang, Normal, template!(NameValueStr: "name"), lang_items,
        "language items are subject to change",
    ),
    (
        sym::rustc_diagnostic_item,
        Normal,
        template!(NameValueStr: "name"),
        Gated(
            Stability::Unstable,
            sym::rustc_attrs,
            "diagnostic items compiler internal support for linting",
            cfg_fn!(rustc_attrs),
        ),
    ),
    gated!(
        // Used in resolve:
        prelude_import, Whitelisted, template!(Word),
        "`#[prelude_import]` is for use by rustc only",
    ),
    gated!(
        rustc_paren_sugar, Normal, template!(Word), unboxed_closures,
        "unboxed_closures are still evolving",
    ),
    rustc_attr!(
        rustc_inherit_overflow_checks, Whitelisted, template!(Word),
        "the `#[rustc_inherit_overflow_checks]` attribute is just used to control \
        overflow checking behavior of several libcore functions that are inlined \
        across crates and will never be stable",
    ),
    rustc_attr!(rustc_reservation_impl, Normal, template!(NameValueStr: "reservation message"),
                "the `#[rustc_reservation_impl]` attribute is internally used \
                 for reserving for `for<T> From<!> for T` impl"
    ),
    rustc_attr!(
        rustc_test_marker, Normal, template!(Word),
        "the `#[rustc_test_marker]` attribute is used internally to track tests",
    ),
    rustc_attr!(
        rustc_unsafe_specialization_marker, Normal, template!(Word),
        "the `#[rustc_unsafe_specialization_marker]` attribute is used to check specializations"
    ),
    rustc_attr!(
        rustc_specialization_trait, Normal, template!(Word),
        "the `#[rustc_specialization_trait]` attribute is used to check specializations"
    ),

    // ==========================================================================
    // Internal attributes, Testing:
    // ==========================================================================

    rustc_attr!(TEST, rustc_outlives, Normal, template!(Word)),
    rustc_attr!(TEST, rustc_variance, Normal, template!(Word)),
    rustc_attr!(TEST, rustc_layout, Normal, template!(List: "field1, field2, ...")),
    rustc_attr!(TEST, rustc_regions, Normal, template!(Word)),
    rustc_attr!(
        TEST, rustc_error, Whitelisted,
        template!(Word, List: "delay_span_bug_from_inside_query")
    ),
    rustc_attr!(TEST, rustc_dump_user_substs, Whitelisted, template!(Word)),
    rustc_attr!(TEST, rustc_if_this_changed, Whitelisted, template!(Word, List: "DepNode")),
    rustc_attr!(TEST, rustc_then_this_would_need, Whitelisted, template!(List: "DepNode")),
    rustc_attr!(
        TEST, rustc_dirty, Whitelisted,
        template!(List: r#"cfg = "...", /*opt*/ label = "...", /*opt*/ except = "...""#),
    ),
    rustc_attr!(
        TEST, rustc_clean, Whitelisted,
        template!(List: r#"cfg = "...", /*opt*/ label = "...", /*opt*/ except = "...""#),
    ),
    rustc_attr!(
        TEST, rustc_partition_reused, Whitelisted,
        template!(List: r#"cfg = "...", module = "...""#),
    ),
    rustc_attr!(
        TEST, rustc_partition_codegened, Whitelisted,
        template!(List: r#"cfg = "...", module = "...""#),
    ),
    rustc_attr!(
        TEST, rustc_expected_cgu_reuse, Whitelisted,
        template!(List: r#"cfg = "...", module = "...", kind = "...""#),
    ),
    rustc_attr!(TEST, rustc_synthetic, Whitelisted, template!(Word)),
    rustc_attr!(TEST, rustc_symbol_name, Whitelisted, template!(Word)),
    rustc_attr!(TEST, rustc_def_path, Whitelisted, template!(Word)),
    rustc_attr!(TEST, rustc_mir, Whitelisted, template!(List: "arg1, arg2, ...")),
    rustc_attr!(TEST, rustc_dump_program_clauses, Whitelisted, template!(Word)),
    rustc_attr!(TEST, rustc_dump_env_program_clauses, Whitelisted, template!(Word)),
    rustc_attr!(TEST, rustc_object_lifetime_default, Whitelisted, template!(Word)),
    rustc_attr!(TEST, rustc_dummy, Normal, template!(Word /* doesn't matter*/)),
    gated!(
        omit_gdb_pretty_printer_section, Whitelisted, template!(Word),
        "the `#[omit_gdb_pretty_printer_section]` attribute is just used for the Rust test suite",
    ),
];

pub fn deprecated_attributes() -> Vec<&'static BuiltinAttribute> {
    BUILTIN_ATTRIBUTES.iter().filter(|(.., gate)| gate.is_deprecated()).collect()
}

pub fn is_builtin_attr_name(name: Symbol) -> bool {
    BUILTIN_ATTRIBUTE_MAP.get(&name).is_some()
}

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
