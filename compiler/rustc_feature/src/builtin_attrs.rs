//! Built-in attributes and `cfg` flag gating.

use AttributeDuplicates::*;
use AttributeGate::*;
use AttributeType::*;

use crate::{Features, Stability};

use rustc_data_structures::fx::FxHashMap;
use rustc_span::symbol::{sym, Symbol};

use std::sync::LazyLock;

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
    (sym::target_abi, sym::cfg_target_abi, cfg_fn!(cfg_target_abi)),
    (sym::target_thread_local, sym::cfg_target_thread_local, cfg_fn!(cfg_target_thread_local)),
    (
        sym::target_has_atomic_equal_alignment,
        sym::cfg_target_has_atomic_equal_alignment,
        cfg_fn!(cfg_target_has_atomic_equal_alignment),
    ),
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
        matches!(*self, Self::Gated(Stability::Deprecated(_, _), ..))
    }
}

/// A template that the attribute input must match.
/// Only top-level shape (`#[attr]` vs `#[attr(...)]` vs `#[attr = ...]`) is considered now.
#[derive(Clone, Copy, Default)]
pub struct AttributeTemplate {
    /// If `true`, the attribute is allowed to be a bare word like `#[test]`.
    pub word: bool,
    /// If `Some`, the attribute is allowed to take a list of items like `#[allow(..)]`.
    pub list: Option<&'static str>,
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

/// A conveniece macro to deal with `$($expr)?`.
macro_rules! or_default {
    ($default:expr,) => {
        $default
    };
    ($default:expr, $next:expr) => {
        $next
    };
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
    ($attr:ident, $typ:expr, $tpl:expr, $duplicates:expr $(, @only_local: $only_local:expr)? $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            only_local: or_default!(false, $($only_local)?),
            type_: $typ,
            template: $tpl,
            gate: Ungated,
            duplicates: $duplicates,
        }
    };
}

macro_rules! gated {
    ($attr:ident, $typ:expr, $tpl:expr, $duplicates:expr $(, @only_local: $only_local:expr)?, $gate:ident, $msg:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            only_local: or_default!(false, $($only_local)?),
            type_: $typ,
            template: $tpl,
            duplicates: $duplicates,
            gate: Gated(Stability::Unstable, sym::$gate, $msg, cfg_fn!($gate)),
        }
    };
    ($attr:ident, $typ:expr, $tpl:expr, $duplicates:expr $(, @only_local: $only_local:expr)?, $msg:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            only_local: or_default!(false, $($only_local)?),
            type_: $typ,
            template: $tpl,
            duplicates: $duplicates,
            gate: Gated(Stability::Unstable, sym::$attr, $msg, cfg_fn!($attr)),
        }
    };
}

macro_rules! rustc_attr {
    (TEST, $attr:ident, $typ:expr, $tpl:expr, $duplicate:expr $(, @only_local: $only_local:expr)? $(,)?) => {
        rustc_attr!(
            $attr,
            $typ,
            $tpl,
            $duplicate,
            $(@only_local: $only_local,)?
            concat!(
                "the `#[",
                stringify!($attr),
                "]` attribute is just used for rustc unit tests \
                and will never be stable",
            ),
        )
    };
    ($attr:ident, $typ:expr, $tpl:expr, $duplicates:expr $(, @only_local: $only_local:expr)?, $msg:expr $(,)?) => {
        BuiltinAttribute {
            name: sym::$attr,
            only_local: or_default!(false, $($only_local)?),
            type_: $typ,
            template: $tpl,
            duplicates: $duplicates,
            gate: Gated(Stability::Unstable, sym::rustc_attrs, $msg, cfg_fn!(rustc_attrs)),
        }
    };
}

macro_rules! experimental {
    ($attr:ident) => {
        concat!("the `#[", stringify!($attr), "]` attribute is an experimental feature")
    };
}

const IMPL_DETAIL: &str = "internal implementation detail";
const INTERNAL_UNSTABLE: &str = "this is an internal attribute that will never be stable";

pub struct BuiltinAttribute {
    pub name: Symbol,
    /// Whether this attribute is only used in the local crate.
    ///
    /// If so, it is not encoded in the crate metadata.
    pub only_local: bool,
    pub type_: AttributeType,
    pub template: AttributeTemplate,
    pub duplicates: AttributeDuplicates,
    pub gate: AttributeGate,
}

/// Attributes that have a special meaning to rustc or rustdoc.
#[rustfmt::skip]
pub const BUILTIN_ATTRIBUTES: &[BuiltinAttribute] = &[
    // ==========================================================================
    // Stable attributes:
    // ==========================================================================

    // Conditional compilation:
    ungated!(cfg, Normal, template!(List: "predicate"), DuplicatesOk),
    ungated!(cfg_attr, Normal, template!(List: "predicate, attr1, attr2, ..."), DuplicatesOk),

    // Testing:
    ungated!(ignore, Normal, template!(Word, NameValueStr: "reason"), WarnFollowing),
    ungated!(
        should_panic, Normal,
        template!(Word, List: r#"expected = "reason""#, NameValueStr: "reason"), FutureWarnFollowing,
    ),
    // FIXME(Centril): This can be used on stable but shouldn't.
    ungated!(reexport_test_harness_main, CrateLevel, template!(NameValueStr: "name"), ErrorFollowing),

    // Macros:
    ungated!(automatically_derived, Normal, template!(Word), WarnFollowing),
    ungated!(macro_use, Normal, template!(Word, List: "name1, name2, ..."), WarnFollowingWordOnly),
    ungated!(macro_escape, Normal, template!(Word), WarnFollowing), // Deprecated synonym for `macro_use`.
    ungated!(macro_export, Normal, template!(Word, List: "local_inner_macros"), WarnFollowing),
    ungated!(proc_macro, Normal, template!(Word), ErrorFollowing),
    ungated!(
        proc_macro_derive, Normal,
        template!(List: "TraitName, /*opt*/ attributes(name1, name2, ...)"), ErrorFollowing,
    ),
    ungated!(proc_macro_attribute, Normal, template!(Word), ErrorFollowing),

    // Lints:
    ungated!(
        warn, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#), DuplicatesOk
    ),
    ungated!(
        allow, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#), DuplicatesOk
    ),
    gated!(
        expect, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#), DuplicatesOk,
        lint_reasons, experimental!(expect)
    ),
    ungated!(
        forbid, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#), DuplicatesOk
    ),
    ungated!(
        deny, Normal, template!(List: r#"lint1, lint2, ..., /*opt*/ reason = "...""#), DuplicatesOk
    ),
    ungated!(must_use, Normal, template!(Word, NameValueStr: "reason"), FutureWarnFollowing),
    gated!(
        must_not_suspend, Normal, template!(Word, NameValueStr: "reason"), WarnFollowing,
        experimental!(must_not_suspend)
    ),
    ungated!(
        deprecated, Normal,
        template!(
            Word,
            List: r#"/*opt*/ since = "version", /*opt*/ note = "reason""#,
            NameValueStr: "reason"
        ),
        ErrorFollowing
    ),

    // Crate properties:
    ungated!(crate_name, CrateLevel, template!(NameValueStr: "name"), FutureWarnFollowing),
    ungated!(crate_type, CrateLevel, template!(NameValueStr: "bin|lib|..."), DuplicatesOk),
    // crate_id is deprecated
    ungated!(crate_id, CrateLevel, template!(NameValueStr: "ignored"), FutureWarnFollowing),

    // ABI, linking, symbols, and FFI
    ungated!(
        link, Normal,
        template!(List: r#"name = "...", /*opt*/ kind = "dylib|static|...", /*opt*/ wasm_import_module = "...", /*opt*/ import_name_type = "decorated|noprefix|undecorated""#),
        DuplicatesOk,
    ),
    ungated!(link_name, Normal, template!(NameValueStr: "name"), FutureWarnPreceding),
    ungated!(no_link, Normal, template!(Word), WarnFollowing),
    ungated!(repr, Normal, template!(List: "C"), DuplicatesOk),
    ungated!(export_name, Normal, template!(NameValueStr: "name"), FutureWarnPreceding),
    ungated!(link_section, Normal, template!(NameValueStr: "name"), FutureWarnPreceding),
    ungated!(no_mangle, Normal, template!(Word), WarnFollowing, @only_local: true),
    ungated!(used, Normal, template!(Word, List: "compiler|linker"), WarnFollowing, @only_local: true),
    ungated!(link_ordinal, Normal, template!(List: "ordinal"), ErrorPreceding),

    // Limits:
    ungated!(recursion_limit, CrateLevel, template!(NameValueStr: "N"), FutureWarnFollowing),
    ungated!(type_length_limit, CrateLevel, template!(NameValueStr: "N"), FutureWarnFollowing),
    gated!(
        const_eval_limit, CrateLevel, template!(NameValueStr: "N"), ErrorFollowing,
        const_eval_limit, experimental!(const_eval_limit)
    ),
    gated!(
        move_size_limit, CrateLevel, template!(NameValueStr: "N"), ErrorFollowing,
        large_assignments, experimental!(move_size_limit)
    ),

    // Entry point:
    gated!(unix_sigpipe, Normal, template!(Word, NameValueStr: "inherit|sig_ign|sig_dfl"), ErrorFollowing, experimental!(unix_sigpipe)),
    ungated!(start, Normal, template!(Word), WarnFollowing),
    ungated!(no_start, CrateLevel, template!(Word), WarnFollowing),
    ungated!(no_main, CrateLevel, template!(Word), WarnFollowing),

    // Modules, prelude, and resolution:
    ungated!(path, Normal, template!(NameValueStr: "file"), FutureWarnFollowing),
    ungated!(no_std, CrateLevel, template!(Word), WarnFollowing),
    ungated!(no_implicit_prelude, Normal, template!(Word), WarnFollowing),
    ungated!(non_exhaustive, Normal, template!(Word), WarnFollowing),

    // Runtime
    ungated!(
        windows_subsystem, CrateLevel,
        template!(NameValueStr: "windows|console"), FutureWarnFollowing
    ),
    ungated!(panic_handler, Normal, template!(Word), WarnFollowing), // RFC 2070

    // Code generation:
    ungated!(inline, Normal, template!(Word, List: "always|never"), FutureWarnFollowing, @only_local: true),
    ungated!(cold, Normal, template!(Word), WarnFollowing, @only_local: true),
    ungated!(no_builtins, CrateLevel, template!(Word), WarnFollowing),
    ungated!(target_feature, Normal, template!(List: r#"enable = "name""#), DuplicatesOk),
    ungated!(track_caller, Normal, template!(Word), WarnFollowing),
    gated!(
        no_sanitize, Normal,
        template!(List: "address, memory, thread"), DuplicatesOk,
        experimental!(no_sanitize)
    ),
    gated!(no_coverage, Normal, template!(Word), WarnFollowing, experimental!(no_coverage)),

    ungated!(
        doc, Normal, template!(List: "hidden|inline|...", NameValueStr: "string"), DuplicatesOk
    ),

    // ==========================================================================
    // Unstable attributes:
    // ==========================================================================

    // RFC #3191: #[debugger_visualizer] support
    gated!(
        debugger_visualizer, Normal, template!(List: r#"natvis_file = "...", gdb_script_file = "...""#),
        DuplicatesOk, experimental!(debugger_visualizer)
    ),

    // Linking:
    gated!(naked, Normal, template!(Word), WarnFollowing, @only_local: true, naked_functions, experimental!(naked)),

    // Plugins:
    BuiltinAttribute {
        name: sym::plugin,
        only_local: false,
        type_: CrateLevel,
        template: template!(List: "name"),
        duplicates: DuplicatesOk,
        gate: Gated(
            Stability::Deprecated(
                "https://github.com/rust-lang/rust/pull/64675",
                Some("may be removed in a future compiler version"),
            ),
            sym::plugin,
            "compiler plugins are deprecated",
            cfg_fn!(plugin)
        ),
    },

    // Testing:
    gated!(
        test_runner, CrateLevel, template!(List: "path"), ErrorFollowing, custom_test_frameworks,
        "custom test frameworks are an unstable feature",
    ),
    // RFC #1268
    gated!(
        marker, Normal, template!(Word), WarnFollowing, marker_trait_attr, experimental!(marker)
    ),
    gated!(
        thread_local, Normal, template!(Word), WarnFollowing,
        "`#[thread_local]` is an experimental feature, and does not currently handle destructors",
    ),
    gated!(no_core, CrateLevel, template!(Word), WarnFollowing, experimental!(no_core)),
    // RFC 2412
    gated!(
        optimize, Normal, template!(List: "size|speed"), ErrorPreceding, optimize_attribute,
        experimental!(optimize),
    ),
    // RFC 2867
    gated!(
        instruction_set, Normal, template!(List: "set"), ErrorPreceding,
        isa_attribute, experimental!(instruction_set)
    ),

    gated!(
        ffi_returns_twice, Normal, template!(Word), WarnFollowing, experimental!(ffi_returns_twice)
    ),
    gated!(ffi_pure, Normal, template!(Word), WarnFollowing, experimental!(ffi_pure)),
    gated!(ffi_const, Normal, template!(Word), WarnFollowing, experimental!(ffi_const)),
    gated!(
        register_tool, CrateLevel, template!(List: "tool1, tool2, ..."), DuplicatesOk,
        experimental!(register_tool),
    ),

    gated!(
        cmse_nonsecure_entry, Normal, template!(Word), WarnFollowing,
        experimental!(cmse_nonsecure_entry)
    ),
    // RFC 2632
    gated!(
        const_trait, Normal, template!(Word), WarnFollowing, const_trait_impl,
        "`const` is a temporary placeholder for marking a trait that is suitable for `const` \
        `impls` and all default bodies as `const`, which may be removed or renamed in the \
        future."
    ),
    // lang-team MCP 147
    gated!(
        deprecated_safe, Normal, template!(List: r#"since = "version", note = "...""#), ErrorFollowing,
        experimental!(deprecated_safe),
    ),

    // `#[collapse_debuginfo]`
    gated!(
        collapse_debuginfo, Normal, template!(Word), WarnFollowing,
        experimental!(collapse_debuginfo)
    ),

    // ==========================================================================
    // Internal attributes: Stability, deprecation, and unsafe:
    // ==========================================================================

    ungated!(feature, CrateLevel, template!(List: "name1, name2, ..."), DuplicatesOk),
    // DuplicatesOk since it has its own validation
    ungated!(
        stable, Normal, template!(List: r#"feature = "name", since = "version""#), DuplicatesOk,
    ),
    ungated!(
        unstable, Normal,
        template!(List: r#"feature = "name", reason = "...", issue = "N""#), DuplicatesOk,
    ),
    ungated!(rustc_const_unstable, Normal, template!(List: r#"feature = "name""#), DuplicatesOk),
    ungated!(rustc_const_stable, Normal, template!(List: r#"feature = "name""#), DuplicatesOk),
    ungated!(
        rustc_default_body_unstable, Normal,
        template!(List: r#"feature = "name", reason = "...", issue = "N""#), DuplicatesOk
    ),
    gated!(
        allow_internal_unstable, Normal, template!(Word, List: "feat1, feat2, ..."), DuplicatesOk,
        "allow_internal_unstable side-steps feature gating and stability checks",
    ),
    gated!(
        rustc_allow_const_fn_unstable, Normal,
        template!(Word, List: "feat1, feat2, ..."), DuplicatesOk,
        "rustc_allow_const_fn_unstable side-steps feature gating and stability checks"
    ),
    gated!(
        allow_internal_unsafe, Normal, template!(Word), WarnFollowing,
        "allow_internal_unsafe side-steps the unsafe_code lint",
    ),
    rustc_attr!(rustc_allowed_through_unstable_modules, Normal, template!(Word), WarnFollowing,
    "rustc_allowed_through_unstable_modules special cases accidental stabilizations of stable items \
    through unstable paths"),

    // ==========================================================================
    // Internal attributes: Type system related:
    // ==========================================================================

    gated!(fundamental, Normal, template!(Word), WarnFollowing, experimental!(fundamental)),
    gated!(
        may_dangle, Normal, template!(Word), WarnFollowing, dropck_eyepatch,
        "`may_dangle` has unstable semantics and may be removed in the future",
    ),

    // ==========================================================================
    // Internal attributes: Runtime related:
    // ==========================================================================

    rustc_attr!(rustc_allocator, Normal, template!(Word), WarnFollowing, IMPL_DETAIL),
    rustc_attr!(rustc_allocator_nounwind, Normal, template!(Word), WarnFollowing, IMPL_DETAIL),
    rustc_attr!(rustc_reallocator, Normal, template!(Word), WarnFollowing, IMPL_DETAIL),
    rustc_attr!(rustc_deallocator, Normal, template!(Word), WarnFollowing, IMPL_DETAIL),
    rustc_attr!(rustc_allocator_zeroed, Normal, template!(Word), WarnFollowing, IMPL_DETAIL),
    gated!(
        alloc_error_handler, Normal, template!(Word), WarnFollowing,
        experimental!(alloc_error_handler)
    ),
    gated!(
        default_lib_allocator, Normal, template!(Word), WarnFollowing, allocator_internals,
        experimental!(default_lib_allocator),
    ),
    gated!(
        needs_allocator, Normal, template!(Word), WarnFollowing, allocator_internals,
        experimental!(needs_allocator),
    ),
    gated!(panic_runtime, Normal, template!(Word), WarnFollowing, experimental!(panic_runtime)),
    gated!(
        needs_panic_runtime, Normal, template!(Word), WarnFollowing,
        experimental!(needs_panic_runtime)
    ),
    gated!(
        compiler_builtins, Normal, template!(Word), WarnFollowing,
        "the `#[compiler_builtins]` attribute is used to identify the `compiler_builtins` crate \
        which contains compiler-rt intrinsics and will never be stable",
    ),
    gated!(
        profiler_runtime, Normal, template!(Word), WarnFollowing,
        "the `#[profiler_runtime]` attribute is used to identify the `profiler_builtins` crate \
        which contains the profiler runtime and will never be stable",
    ),

    // ==========================================================================
    // Internal attributes, Linkage:
    // ==========================================================================

    gated!(
        linkage, Normal, template!(NameValueStr: "external|internal|..."), ErrorPreceding, @only_local: true,
        "the `linkage` attribute is experimental and not portable across platforms",
    ),
    rustc_attr!(
        rustc_std_internal_symbol, Normal, template!(Word), WarnFollowing, @only_local: true, INTERNAL_UNSTABLE
    ),

    // ==========================================================================
    // Internal attributes, Macro related:
    // ==========================================================================

    rustc_attr!(
        rustc_builtin_macro, Normal,
        template!(Word, List: "name, /*opt*/ attributes(name1, name2, ...)"), ErrorFollowing,
        IMPL_DETAIL,
    ),
    rustc_attr!(rustc_proc_macro_decls, Normal, template!(Word), WarnFollowing, INTERNAL_UNSTABLE),
    rustc_attr!(
        rustc_macro_transparency, Normal,
        template!(NameValueStr: "transparent|semitransparent|opaque"), ErrorFollowing,
        "used internally for testing macro hygiene",
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
        ErrorFollowing,
        INTERNAL_UNSTABLE
    ),
    // Enumerates "identity-like" conversion methods to suggest on type mismatch.
    rustc_attr!(
        rustc_conversion_suggestion, Normal, template!(Word), WarnFollowing, INTERNAL_UNSTABLE
    ),
    // Prevents field reads in the marked trait or method to be considered
    // during dead code analysis.
    rustc_attr!(
        rustc_trivial_field_reads, Normal, template!(Word), WarnFollowing, INTERNAL_UNSTABLE
    ),
    // Used by the `rustc::potential_query_instability` lint to warn methods which
    // might not be stable during incremental compilation.
    rustc_attr!(rustc_lint_query_instability, Normal, template!(Word), WarnFollowing, INTERNAL_UNSTABLE),
    // Used by the `rustc::untranslatable_diagnostic` and `rustc::diagnostic_outside_of_impl` lints
    // to assist in changes to diagnostic APIs.
    rustc_attr!(rustc_lint_diagnostics, Normal, template!(Word), WarnFollowing, INTERNAL_UNSTABLE),
    // Used by the `rustc::bad_opt_access` lint to identify `DebuggingOptions` and `CodegenOptions`
    // types (as well as any others in future).
    rustc_attr!(rustc_lint_opt_ty, Normal, template!(Word), WarnFollowing, INTERNAL_UNSTABLE),
    // Used by the `rustc::bad_opt_access` lint on fields
    // types (as well as any others in future).
    rustc_attr!(rustc_lint_opt_deny_field_access, Normal, template!(List: "message"), WarnFollowing, INTERNAL_UNSTABLE),

    // ==========================================================================
    // Internal attributes, Const related:
    // ==========================================================================

    rustc_attr!(rustc_promotable, Normal, template!(Word), WarnFollowing, IMPL_DETAIL),
    rustc_attr!(
        rustc_legacy_const_generics, Normal, template!(List: "N"), ErrorFollowing,
        INTERNAL_UNSTABLE
    ),
    // Do not const-check this function's body. It will always get replaced during CTFE.
    rustc_attr!(
        rustc_do_not_const_check, Normal, template!(Word), WarnFollowing, INTERNAL_UNSTABLE
    ),

    // ==========================================================================
    // Internal attributes, Layout related:
    // ==========================================================================

    rustc_attr!(
        rustc_layout_scalar_valid_range_start, Normal, template!(List: "value"), ErrorFollowing,
        "the `#[rustc_layout_scalar_valid_range_start]` attribute is just used to enable \
        niche optimizations in libcore and libstd and will never be stable",
    ),
    rustc_attr!(
        rustc_layout_scalar_valid_range_end, Normal, template!(List: "value"), ErrorFollowing,
        "the `#[rustc_layout_scalar_valid_range_end]` attribute is just used to enable \
        niche optimizations in libcore and libstd and will never be stable",
    ),
    rustc_attr!(
        rustc_nonnull_optimization_guaranteed, Normal, template!(Word), WarnFollowing,
        "the `#[rustc_nonnull_optimization_guaranteed]` attribute is just used to enable \
        niche optimizations in libcore and libstd and will never be stable",
    ),

    // ==========================================================================
    // Internal attributes, Misc:
    // ==========================================================================
    gated!(
        lang, Normal, template!(NameValueStr: "name"), DuplicatesOk, @only_local: true, lang_items,
        "language items are subject to change",
    ),
    rustc_attr!(
        rustc_pass_by_value, Normal,
        template!(Word), ErrorFollowing,
        "#[rustc_pass_by_value] is used to mark types that must be passed by value instead of reference."
    ),
    rustc_attr!(
        rustc_coherence_is_core, AttributeType::CrateLevel, template!(Word), ErrorFollowing, @only_local: true,
        "#![rustc_coherence_is_core] allows inherent methods on builtin types, only intended to be used in `core`."
    ),
    rustc_attr!(
        rustc_allow_incoherent_impl, AttributeType::Normal, template!(Word), ErrorFollowing, @only_local: true,
        "#[rustc_allow_incoherent_impl] has to be added to all impl items of an incoherent inherent impl."
    ),
    rustc_attr!(
        rustc_has_incoherent_inherent_impls, AttributeType::Normal, template!(Word), ErrorFollowing,
        "#[rustc_has_incoherent_inherent_impls] allows the addition of incoherent inherent impls for \
         the given type by annotating all impl items with #[rustc_allow_incoherent_impl]."
    ),
    rustc_attr!(
        rustc_box, AttributeType::Normal, template!(Word), ErrorFollowing,
        "#[rustc_box] allows creating boxes \
        and it is only intended to be used in `alloc`."
    ),

    BuiltinAttribute {
        name: sym::rustc_diagnostic_item,
        // FIXME: This can be `true` once we always use `tcx.is_diagnostic_item`.
        only_local: false,
        type_: Normal,
        template: template!(NameValueStr: "name"),
        duplicates: ErrorFollowing,
        gate: Gated(
            Stability::Unstable,
            sym::rustc_attrs,
            "diagnostic items compiler internal support for linting",
            cfg_fn!(rustc_attrs),
        ),
    },
    gated!(
        // Used in resolve:
        prelude_import, Normal, template!(Word), WarnFollowing,
        "`#[prelude_import]` is for use by rustc only",
    ),
    gated!(
        rustc_paren_sugar, Normal, template!(Word), WarnFollowing, unboxed_closures,
        "unboxed_closures are still evolving",
    ),
    rustc_attr!(
        rustc_inherit_overflow_checks, Normal, template!(Word), WarnFollowing, @only_local: true,
        "the `#[rustc_inherit_overflow_checks]` attribute is just used to control \
        overflow checking behavior of several libcore functions that are inlined \
        across crates and will never be stable",
    ),
    rustc_attr!(
        rustc_reservation_impl, Normal,
        template!(NameValueStr: "reservation message"), ErrorFollowing,
        "the `#[rustc_reservation_impl]` attribute is internally used \
         for reserving for `for<T> From<!> for T` impl"
    ),
    rustc_attr!(
        rustc_test_marker, Normal, template!(Word), WarnFollowing,
        "the `#[rustc_test_marker]` attribute is used internally to track tests",
    ),
    rustc_attr!(
        rustc_unsafe_specialization_marker, Normal, template!(Word), WarnFollowing,
        "the `#[rustc_unsafe_specialization_marker]` attribute is used to check specializations"
    ),
    rustc_attr!(
        rustc_specialization_trait, Normal, template!(Word), WarnFollowing,
        "the `#[rustc_specialization_trait]` attribute is used to check specializations"
    ),
    rustc_attr!(
        rustc_main, Normal, template!(Word), WarnFollowing,
        "the `#[rustc_main]` attribute is used internally to specify test entry point function",
    ),
    rustc_attr!(
        rustc_skip_array_during_method_dispatch, Normal, template!(Word), WarnFollowing,
        "the `#[rustc_skip_array_during_method_dispatch]` attribute is used to exclude a trait \
        from method dispatch when the receiver is an array, for compatibility in editions < 2021."
    ),
    rustc_attr!(
        rustc_must_implement_one_of, Normal, template!(List: "function1, function2, ..."), ErrorFollowing,
        "the `#[rustc_must_implement_one_of]` attribute is used to change minimal complete \
        definition of a trait, it's currently in experimental form and should be changed before \
        being exposed outside of the std"
    ),

    // ==========================================================================
    // Internal attributes, Testing:
    // ==========================================================================

    rustc_attr!(TEST, rustc_access_level, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_outlives, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_capture_analysis, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_insignificant_dtor, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_strict_coherence, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_variance, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_layout, Normal, template!(List: "field1, field2, ..."), WarnFollowing),
    rustc_attr!(TEST, rustc_regions, Normal, template!(Word), WarnFollowing),
    rustc_attr!(
        TEST, rustc_error, Normal,
        template!(Word, List: "delay_span_bug_from_inside_query"), WarnFollowingWordOnly
    ),
    rustc_attr!(TEST, rustc_dump_user_substs, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_evaluate_where_clauses, Normal, template!(Word), WarnFollowing),
    rustc_attr!(
        TEST, rustc_if_this_changed, Normal, template!(Word, List: "DepNode"), DuplicatesOk
    ),
    rustc_attr!(
        TEST, rustc_then_this_would_need, Normal, template!(List: "DepNode"), DuplicatesOk
    ),
    rustc_attr!(
        TEST, rustc_clean, Normal,
        template!(List: r#"cfg = "...", /*opt*/ label = "...", /*opt*/ except = "...""#),
        DuplicatesOk,
    ),
    rustc_attr!(
        TEST, rustc_partition_reused, Normal,
        template!(List: r#"cfg = "...", module = "...""#), DuplicatesOk,
    ),
    rustc_attr!(
        TEST, rustc_partition_codegened, Normal,
        template!(List: r#"cfg = "...", module = "...""#), DuplicatesOk,
    ),
    rustc_attr!(
        TEST, rustc_expected_cgu_reuse, Normal,
        template!(List: r#"cfg = "...", module = "...", kind = "...""#), DuplicatesOk,
    ),
    rustc_attr!(TEST, rustc_symbol_name, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_polymorphize_error, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_def_path, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_mir, Normal, template!(List: "arg1, arg2, ..."), DuplicatesOk),
    rustc_attr!(TEST, rustc_dump_program_clauses, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_dump_env_program_clauses, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_object_lifetime_default, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_dump_vtable, Normal, template!(Word), WarnFollowing),
    rustc_attr!(TEST, rustc_dummy, Normal, template!(Word /* doesn't matter*/), DuplicatesOk),
    gated!(
        omit_gdb_pretty_printer_section, Normal, template!(Word), WarnFollowing,
        "the `#[omit_gdb_pretty_printer_section]` attribute is just used for the Rust test suite",
    ),
];

pub fn deprecated_attributes() -> Vec<&'static BuiltinAttribute> {
    BUILTIN_ATTRIBUTES.iter().filter(|attr| attr.gate.is_deprecated()).collect()
}

pub fn is_builtin_attr_name(name: Symbol) -> bool {
    BUILTIN_ATTRIBUTE_MAP.get(&name).is_some()
}

pub fn is_builtin_only_local(name: Symbol) -> bool {
    BUILTIN_ATTRIBUTE_MAP.get(&name).map_or(false, |attr| attr.only_local)
}

pub fn is_valid_for_get_attr(name: Symbol) -> bool {
    BUILTIN_ATTRIBUTE_MAP.get(&name).map_or(false, |attr| match attr.duplicates {
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
