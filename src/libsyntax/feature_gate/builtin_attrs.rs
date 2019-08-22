//! Built-in attributes and `cfg` flag gating.

use AttributeType::*;
use AttributeGate::*;

use super::{emit_feature_err, GateIssue};
use super::{Stability, EXPLAIN_ALLOW_INTERNAL_UNSAFE, EXPLAIN_ALLOW_INTERNAL_UNSTABLE};
use super::active::Features;

use crate::ast;
use crate::attr::AttributeTemplate;
use crate::symbol::{Symbol, sym};
use crate::parse::ParseSess;

use syntax_pos::Span;
use rustc_data_structures::fx::FxHashMap;
use lazy_static::lazy_static;

type GateFn = fn(&Features) -> bool;

macro_rules! cfg_fn {
    ($field: ident) => {
        (|features| { features.$field }) as GateFn
    }
}

/// `cfg(...)`'s that are feature gated.
const GATED_CFGS: &[(Symbol, Symbol, GateFn)] = &[
    // (name in cfg, feature, function to check if the feature is enabled)
    (sym::target_thread_local, sym::cfg_target_thread_local, cfg_fn!(cfg_target_thread_local)),
    (sym::target_has_atomic, sym::cfg_target_has_atomic, cfg_fn!(cfg_target_has_atomic)),
    (sym::rustdoc, sym::doc_cfg, cfg_fn!(doc_cfg)),
    (sym::doctest, sym::cfg_doctest, cfg_fn!(cfg_doctest)),
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

// fn() is not Debug
impl std::fmt::Debug for AttributeGate {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Gated(ref stab, name, expl, _) =>
                write!(fmt, "Gated({:?}, {}, {})", stab, name, expl),
            Self::Ungated => write!(fmt, "Ungated")
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

pub type BuiltinAttribute = (Symbol, AttributeType, AttributeTemplate, AttributeGate);

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
    (sym::rustc_dump_env_program_clauses, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                                    sym::rustc_attrs,
                                                    "the `#[rustc_dump_env_program_clauses]` \
                                                    attribute is just used for rustc unit \
                                                    tests and will never be stable",
                                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_object_lifetime_default, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                                    sym::rustc_attrs,
                                                    "the `#[rustc_object_lifetime_default]` \
                                                    attribute is just used for rustc unit \
                                                    tests and will never be stable",
                                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_test_marker, Normal, template!(Word), Gated(Stability::Unstable,
                                    sym::rustc_attrs,
                                    "the `#[rustc_test_marker]` attribute \
                                    is used internally to track tests",
                                    cfg_fn!(rustc_attrs))),
    (sym::rustc_macro_transparency, Whitelisted, template!(NameValueStr:
                                                           "transparent|semitransparent|opaque"),
                                                Gated(Stability::Unstable,
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

    (sym::rustc_allocator, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                                sym::rustc_attrs,
                                                "internal implementation detail",
                                                cfg_fn!(rustc_attrs))),

    (sym::rustc_allocator_nounwind, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                                sym::rustc_attrs,
                                                "internal implementation detail",
                                                cfg_fn!(rustc_attrs))),

    (sym::rustc_builtin_macro, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                                sym::rustc_attrs,
                                                "internal implementation detail",
                                                cfg_fn!(rustc_attrs))),

    (sym::rustc_promotable, Whitelisted, template!(Word), Gated(Stability::Unstable,
                                                sym::rustc_attrs,
                                                "internal implementation detail",
                                                cfg_fn!(rustc_attrs))),

    (sym::rustc_allow_const_fn_ptr, Whitelisted, template!(Word), Gated(Stability::Unstable,
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
    (sym::may_dangle,
    Normal,
    template!(Word),
    Gated(Stability::Unstable,
        sym::dropck_eyepatch,
        "`may_dangle` has unstable semantics and may be removed in the future",
        cfg_fn!(dropck_eyepatch))),
    (sym::unwind, Whitelisted, template!(List: "allowed|aborts"), Gated(Stability::Unstable,
                                sym::unwind_attributes,
                                "`#[unwind]` is experimental",
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
                        "`#[alloc_error_handler]` is an unstable feature",
                        cfg_fn!(alloc_error_handler))),

    // RFC 2412
    (sym::optimize, Whitelisted, template!(List: "size|speed"), Gated(Stability::Unstable,
                            sym::optimize_attribute,
                            "`#[optimize]` attribute is an unstable feature",
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
                    "custom test frameworks are an unstable feature",
                    cfg_fn!(custom_test_frameworks))),
];

pub fn deprecated_attributes() -> Vec<&'static BuiltinAttribute> {
    BUILTIN_ATTRIBUTES.iter().filter(|(.., gate)| gate.is_deprecated()).collect()
}

pub fn is_builtin_attr_name(name: ast::Name) -> bool {
    BUILTIN_ATTRIBUTE_MAP.get(&name).is_some()
}

pub fn is_builtin_attr(attr: &ast::Attribute) -> bool {
    attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name)).is_some()
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
