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

use self::Status::*;
use self::AttributeType::*;
use self::AttributeGate::*;

use abi::Abi;
use ast::{NodeId, PatKind};
use ast;
use attr;
use attr::AttrMetaMethods;
use codemap::{CodeMap, Span};
use errors::Handler;
use visit;
use visit::{FnKind, Visitor};
use parse::token::InternedString;

use std::ascii::AsciiExt;
use std::cmp;

enum Status {
    /// Represents an active feature that is currently being implemented or
    /// currently being considered for addition/removal.
    Active,

    /// Represents a feature which has since been removed (it was once Active)
    Removed,

    /// This language feature has since been Accepted (it was once Active)
    Accepted,
}

// If you change this list without updating src/doc/reference.md, @cmr will be sad
// Don't ever remove anything from this list; set them to 'Removed'.
// The version numbers here correspond to the version in which the current status
// was set. This is most important for knowing when a particular feature became
// stable (active).
// NB: The tidy tool parses this information directly out of the source so take
// care when modifying it.
const KNOWN_FEATURES: &'static [(&'static str, &'static str, Option<u32>, Status)] = &[
    ("globs", "1.0.0", None, Accepted),
    ("macro_rules", "1.0.0", None, Accepted),
    ("struct_variant", "1.0.0", None, Accepted),
    ("asm", "1.0.0", Some(29722), Active),
    ("managed_boxes", "1.0.0", None, Removed),
    ("non_ascii_idents", "1.0.0", Some(28979), Active),
    ("thread_local", "1.0.0", Some(29594), Active),
    ("link_args", "1.0.0", Some(29596), Active),
    ("plugin_registrar", "1.0.0", Some(29597), Active),
    ("log_syntax", "1.0.0", Some(29598), Active),
    ("trace_macros", "1.0.0", Some(29598), Active),
    ("concat_idents", "1.0.0", Some(29599), Active),

    // rustc internal, for now:
    ("intrinsics", "1.0.0", None, Active),
    ("lang_items", "1.0.0", None, Active),

    ("simd", "1.0.0", Some(27731), Active),
    ("default_type_params", "1.0.0", None, Accepted),
    ("quote", "1.0.0", Some(29601), Active),
    ("link_llvm_intrinsics", "1.0.0", Some(29602), Active),
    ("linkage", "1.0.0", Some(29603), Active),
    ("struct_inherit", "1.0.0", None, Removed),

    ("quad_precision_float", "1.0.0", None, Removed),

    // rustc internal
    ("rustc_diagnostic_macros", "1.0.0", None, Active),
    ("unboxed_closures", "1.0.0", Some(29625), Active),
    ("reflect", "1.0.0", Some(27749), Active),
    ("import_shadowing", "1.0.0", None, Removed),
    ("advanced_slice_patterns", "1.0.0", Some(23121), Active),
    ("tuple_indexing", "1.0.0", None, Accepted),
    ("associated_types", "1.0.0", None, Accepted),
    ("visible_private_types", "1.0.0", None, Removed),
    ("slicing_syntax", "1.0.0", None, Accepted),
    ("box_syntax", "1.0.0", Some(27779), Active),
    ("placement_in_syntax", "1.0.0", Some(27779), Active),

    // rustc internal.
    ("pushpop_unsafe", "1.2.0", None, Active),

    ("on_unimplemented", "1.0.0", Some(29628), Active),
    ("simd_ffi", "1.0.0", Some(27731), Active),
    ("allocator", "1.0.0", Some(27389), Active),
    ("needs_allocator", "1.4.0", Some(27389), Active),
    ("linked_from", "1.3.0", Some(29629), Active),

    ("if_let", "1.0.0", None, Accepted),
    ("while_let", "1.0.0", None, Accepted),

    ("plugin", "1.0.0", Some(29597), Active),
    ("start", "1.0.0", Some(29633), Active),
    ("main", "1.0.0", Some(29634), Active),

    ("fundamental", "1.0.0", Some(29635), Active),

    // A temporary feature gate used to enable parser extensions needed
    // to bootstrap fix for #5723.
    ("issue_5723_bootstrap", "1.0.0", None, Accepted),

    ("structural_match", "1.8.0", Some(31434), Active),

    // A way to temporarily opt out of opt in copy. This will *never* be accepted.
    ("opt_out_copy", "1.0.0", None, Removed),

    // OIBIT specific features
    ("optin_builtin_traits", "1.0.0", Some(13231), Active),

    // macro reexport needs more discussion and stabilization
    ("macro_reexport", "1.0.0", Some(29638), Active),

    // These are used to test this portion of the compiler, they don't actually
    // mean anything
    ("test_accepted_feature", "1.0.0", None, Accepted),
    ("test_removed_feature", "1.0.0", None, Removed),

    // Allows use of #[staged_api]
    // rustc internal
    ("staged_api", "1.0.0", None, Active),

    // Allows using items which are missing stability attributes
    // rustc internal
    ("unmarked_api", "1.0.0", None, Active),

    // Allows using #![no_std]
    ("no_std", "1.0.0", None, Accepted),

    // Allows using #![no_core]
    ("no_core", "1.3.0", Some(29639), Active),

    // Allows using `box` in patterns; RFC 469
    ("box_patterns", "1.0.0", Some(29641), Active),

    // Allows using the unsafe_no_drop_flag attribute (unlikely to
    // switch to Accepted; see RFC 320)
    ("unsafe_no_drop_flag", "1.0.0", None, Active),

    // Allows using the unsafe_destructor_blind_to_params attribute;
    // RFC 1238
    ("dropck_parametricity", "1.3.0", Some(28498), Active),

    // Allows the use of custom attributes; RFC 572
    ("custom_attribute", "1.0.0", Some(29642), Active),

    // Allows the use of #[derive(Anything)] as sugar for
    // #[derive_Anything].
    ("custom_derive", "1.0.0", Some(29644), Active),

    // Allows the use of rustc_* attributes; RFC 572
    ("rustc_attrs", "1.0.0", Some(29642), Active),

    // Allows the use of #[allow_internal_unstable]. This is an
    // attribute on macro_rules! and can't use the attribute handling
    // below (it has to be checked before expansion possibly makes
    // macros disappear).
    //
    // rustc internal
    ("allow_internal_unstable", "1.0.0", None, Active),

    // #23121. Array patterns have some hazards yet.
    ("slice_patterns", "1.0.0", Some(23121), Active),

    // Allows use of unary negate on unsigned integers, e.g. -e for e: u8
    ("negate_unsigned", "1.0.0", Some(29645), Removed),

    // Allows the definition of associated constants in `trait` or `impl`
    // blocks.
    ("associated_consts", "1.0.0", Some(29646), Active),

    // Allows the definition of `const fn` functions.
    ("const_fn", "1.2.0", Some(24111), Active),

    // Allows indexing into constant arrays.
    ("const_indexing", "1.4.0", Some(29947), Active),

    // Allows using #[prelude_import] on glob `use` items.
    //
    // rustc internal
    ("prelude_import", "1.2.0", None, Active),

    // Allows the definition recursive static items.
    ("static_recursion", "1.3.0", Some(29719), Active),

    // Allows default type parameters to influence type inference.
    ("default_type_parameter_fallback", "1.3.0", Some(27336), Active),

    // Allows associated type defaults
    ("associated_type_defaults", "1.2.0", Some(29661), Active),

    // Allows macros to appear in the type position.
    ("type_macros", "1.3.0", Some(27245), Active),

    // allow `repr(simd)`, and importing the various simd intrinsics
    ("repr_simd", "1.4.0", Some(27731), Active),

    // Allows cfg(target_feature = "...").
    ("cfg_target_feature", "1.4.0", Some(29717), Active),

    // allow `extern "platform-intrinsic" { ... }`
    ("platform_intrinsics", "1.4.0", Some(27731), Active),

    // allow `#[unwind]`
    // rust runtime internal
    ("unwind_attributes", "1.4.0", None, Active),

    // allow the use of `#[naked]` on functions.
    ("naked_functions", "1.9.0", Some(32408), Active),

    // allow empty structs and enum variants with braces
    ("braced_empty_structs", "1.8.0", Some(29720), Accepted),

    // allow overloading augmented assignment operations like `a += b`
    ("augmented_assignments", "1.8.0", Some(28235), Accepted),

    // allow `#[no_debug]`
    ("no_debug", "1.5.0", Some(29721), Active),

    // allow `#[omit_gdb_pretty_printer_section]`
    // rustc internal.
    ("omit_gdb_pretty_printer_section", "1.5.0", None, Active),

    // Allows cfg(target_vendor = "...").
    ("cfg_target_vendor", "1.5.0", Some(29718), Active),

    // Allow attributes on expressions and non-item statements
    ("stmt_expr_attributes", "1.6.0", Some(15701), Active),

    // Allows `#[deprecated]` attribute
    ("deprecated", "1.9.0", Some(29935), Accepted),

    // allow using type ascription in expressions
    ("type_ascription", "1.6.0", Some(23416), Active),

    // Allows cfg(target_thread_local)
    ("cfg_target_thread_local", "1.7.0", Some(29594), Active),

    // rustc internal
    ("abi_vectorcall", "1.7.0", None, Active),

    // a...b and ...b
    ("inclusive_range_syntax", "1.7.0", Some(28237), Active),

    // `expr?`
    ("question_mark", "1.9.0", Some(31436), Active),

    // impl specialization (RFC 1210)
    ("specialization", "1.7.0", Some(31844), Active),

    // pub(restricted) visibilities (RFC 1422)
    ("pub_restricted", "1.9.0", Some(32409), Active),
];
// (changing above list without updating src/doc/reference.md makes @cmr sad)

#[derive(PartialEq, Copy, Clone, Debug)]
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
    Gated(&'static str, &'static str, fn(&Features) -> bool),

    /// Ungated attribute, can be used on all release channels
    Ungated,
}

// fn() is not Debug
impl ::std::fmt::Debug for AttributeGate {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            Gated(ref name, ref expl, _) => write!(fmt, "Gated({}, {})", name, expl),
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

// Attributes that have a special meaning to rustc or rustdoc
pub const KNOWN_ATTRIBUTES: &'static [(&'static str, AttributeType, AttributeGate)] = &[
    // Normal attributes

    ("warn", Normal, Ungated),
    ("allow", Normal, Ungated),
    ("forbid", Normal, Ungated),
    ("deny", Normal, Ungated),

    ("macro_reexport", Normal, Ungated),
    ("macro_use", Normal, Ungated),
    ("macro_export", Normal, Ungated),
    ("plugin_registrar", Normal, Ungated),

    ("cfg", Normal, Ungated),
    ("cfg_attr", Normal, Ungated),
    ("main", Normal, Ungated),
    ("start", Normal, Ungated),
    ("test", Normal, Ungated),
    ("bench", Normal, Ungated),
    ("simd", Normal, Ungated),
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
    ("link_args", Normal, Ungated),
    ("macro_escape", Normal, Ungated),

    // RFC #1445.
    ("structural_match", Whitelisted, Gated("structural_match",
                                            "the semantics of constant patterns is \
                                             not yet settled",
                                            cfg_fn!(structural_match))),

    // Not used any more, but we can't feature gate it
    ("no_stack_check", Normal, Ungated),

    ("plugin", CrateLevel, Gated("plugin",
                                 "compiler plugins are experimental \
                                  and possibly buggy",
                                 cfg_fn!(plugin))),

    ("no_std", CrateLevel, Ungated),
    ("no_core", CrateLevel, Gated("no_core",
                                  "no_core is experimental",
                                  cfg_fn!(no_core))),
    ("lang", Normal, Gated("lang_items",
                           "language items are subject to change",
                           cfg_fn!(lang_items))),
    ("linkage", Whitelisted, Gated("linkage",
                                   "the `linkage` attribute is experimental \
                                    and not portable across platforms",
                                   cfg_fn!(linkage))),
    ("thread_local", Whitelisted, Gated("thread_local",
                                        "`#[thread_local]` is an experimental feature, and does \
                                         not currently handle destructors. There is no \
                                         corresponding `#[task_local]` mapping to the task \
                                         model",
                                        cfg_fn!(thread_local))),

    ("rustc_on_unimplemented", Normal, Gated("on_unimplemented",
                                             "the `#[rustc_on_unimplemented]` attribute \
                                              is an experimental feature",
                                             cfg_fn!(on_unimplemented))),
    ("allocator", Whitelisted, Gated("allocator",
                                     "the `#[allocator]` attribute is an experimental feature",
                                     cfg_fn!(allocator))),
    ("needs_allocator", Normal, Gated("needs_allocator",
                                      "the `#[needs_allocator]` \
                                       attribute is an experimental \
                                       feature",
                                      cfg_fn!(needs_allocator))),
    ("rustc_variance", Normal, Gated("rustc_attrs",
                                     "the `#[rustc_variance]` attribute \
                                      is just used for rustc unit tests \
                                      and will never be stable",
                                     cfg_fn!(rustc_attrs))),
    ("rustc_error", Whitelisted, Gated("rustc_attrs",
                                       "the `#[rustc_error]` attribute \
                                        is just used for rustc unit tests \
                                        and will never be stable",
                                       cfg_fn!(rustc_attrs))),
    ("rustc_if_this_changed", Whitelisted, Gated("rustc_attrs",
                                                 "the `#[rustc_if_this_changed]` attribute \
                                                  is just used for rustc unit tests \
                                                  and will never be stable",
                                                 cfg_fn!(rustc_attrs))),
    ("rustc_then_this_would_need", Whitelisted, Gated("rustc_attrs",
                                                      "the `#[rustc_if_this_changed]` attribute \
                                                       is just used for rustc unit tests \
                                                       and will never be stable",
                                                      cfg_fn!(rustc_attrs))),
    ("rustc_dirty", Whitelisted, Gated("rustc_attrs",
                                       "the `#[rustc_dirty]` attribute \
                                        is just used for rustc unit tests \
                                        and will never be stable",
                                       cfg_fn!(rustc_attrs))),
    ("rustc_clean", Whitelisted, Gated("rustc_attrs",
                                       "the `#[rustc_clean]` attribute \
                                        is just used for rustc unit tests \
                                        and will never be stable",
                                       cfg_fn!(rustc_attrs))),
    ("rustc_symbol_name", Whitelisted, Gated("rustc_attrs",
                                             "internal rustc attributes will never be stable",
                                             cfg_fn!(rustc_attrs))),
    ("rustc_item_path", Whitelisted, Gated("rustc_attrs",
                                           "internal rustc attributes will never be stable",
                                           cfg_fn!(rustc_attrs))),
    ("rustc_move_fragments", Normal, Gated("rustc_attrs",
                                           "the `#[rustc_move_fragments]` attribute \
                                            is just used for rustc unit tests \
                                            and will never be stable",
                                           cfg_fn!(rustc_attrs))),
    ("rustc_mir", Whitelisted, Gated("rustc_attrs",
                                     "the `#[rustc_mir]` attribute \
                                      is just used for rustc unit tests \
                                      and will never be stable",
                                     cfg_fn!(rustc_attrs))),
    ("rustc_no_mir", Whitelisted, Gated("rustc_attrs",
                                        "the `#[rustc_no_mir]` attribute \
                                         is just used to make tests pass \
                                         and will never be stable",
                                        cfg_fn!(rustc_attrs))),

    ("allow_internal_unstable", Normal, Gated("allow_internal_unstable",
                                              EXPLAIN_ALLOW_INTERNAL_UNSTABLE,
                                              cfg_fn!(allow_internal_unstable))),

    ("fundamental", Whitelisted, Gated("fundamental",
                                       "the `#[fundamental]` attribute \
                                        is an experimental feature",
                                       cfg_fn!(fundamental))),

    ("linked_from", Normal, Gated("linked_from",
                                  "the `#[linked_from]` attribute \
                                   is an experimental feature",
                                  cfg_fn!(linked_from))),

    // FIXME: #14408 whitelist docs since rustdoc looks at them
    ("doc", Whitelisted, Ungated),

    // FIXME: #14406 these are processed in trans, which happens after the
    // lint pass
    ("cold", Whitelisted, Ungated),
    ("naked", Whitelisted, Gated("naked_functions",
                                 "the `#[naked]` attribute \
                                  is an experimental feature",
                                 cfg_fn!(naked))),
    ("export_name", Whitelisted, Ungated),
    ("inline", Whitelisted, Ungated),
    ("link", Whitelisted, Ungated),
    ("link_name", Whitelisted, Ungated),
    ("link_section", Whitelisted, Ungated),
    ("no_builtins", Whitelisted, Ungated),
    ("no_mangle", Whitelisted, Ungated),
    ("no_debug", Whitelisted, Gated("no_debug",
                                    "the `#[no_debug]` attribute \
                                     is an experimental feature",
                                    cfg_fn!(no_debug))),
    ("omit_gdb_pretty_printer_section", Whitelisted, Gated("omit_gdb_pretty_printer_section",
                                                       "the `#[omit_gdb_pretty_printer_section]` \
                                                        attribute is just used for the Rust test \
                                                        suite",
                                                       cfg_fn!(omit_gdb_pretty_printer_section))),
    ("unsafe_no_drop_flag", Whitelisted, Gated("unsafe_no_drop_flag",
                                               "unsafe_no_drop_flag has unstable semantics \
                                                and may be removed in the future",
                                               cfg_fn!(unsafe_no_drop_flag))),
    ("unsafe_destructor_blind_to_params",
     Normal,
     Gated("dropck_parametricity",
           "unsafe_destructor_blind_to_params has unstable semantics \
            and may be removed in the future",
           cfg_fn!(dropck_parametricity))),
    ("unwind", Whitelisted, Gated("unwind_attributes", "#[unwind] is experimental",
                                  cfg_fn!(unwind_attributes))),

    // used in resolve
    ("prelude_import", Whitelisted, Gated("prelude_import",
                                          "`#[prelude_import]` is for use by rustc only",
                                          cfg_fn!(prelude_import))),

    // FIXME: #14407 these are only looked at on-demand so we can't
    // guarantee they'll have already been checked
    ("rustc_deprecated", Whitelisted, Ungated),
    ("must_use", Whitelisted, Ungated),
    ("stable", Whitelisted, Ungated),
    ("unstable", Whitelisted, Ungated),
    ("deprecated", Normal, Ungated),

    ("rustc_paren_sugar", Normal, Gated("unboxed_closures",
                                        "unboxed_closures are still evolving",
                                        cfg_fn!(unboxed_closures))),
    ("rustc_reflect_like", Whitelisted, Gated("reflect",
                                              "defining reflective traits is still evolving",
                                              cfg_fn!(reflect))),

    // Crate level attributes
    ("crate_name", CrateLevel, Ungated),
    ("crate_type", CrateLevel, Ungated),
    ("crate_id", CrateLevel, Ungated),
    ("feature", CrateLevel, Ungated),
    ("no_start", CrateLevel, Ungated),
    ("no_main", CrateLevel, Ungated),
    ("no_builtins", CrateLevel, Ungated),
    ("recursion_limit", CrateLevel, Ungated),
];

// cfg(...)'s that are feature gated
const GATED_CFGS: &'static [(&'static str, &'static str, fn(&Features) -> bool)] = &[
    // (name in cfg, feature, function to check if the feature is enabled)
    ("target_feature", "cfg_target_feature", cfg_fn!(cfg_target_feature)),
    ("target_vendor", "cfg_target_vendor", cfg_fn!(cfg_target_vendor)),
    ("target_thread_local", "cfg_target_thread_local", cfg_fn!(cfg_target_thread_local)),
];

#[derive(Debug, Eq, PartialEq)]
pub enum GatedCfgAttr {
    GatedCfg(GatedCfg),
    GatedAttr(Span),
}

#[derive(Debug, Eq, PartialEq)]
pub struct GatedCfg {
    span: Span,
    index: usize,
}

impl Ord for GatedCfgAttr {
    fn cmp(&self, other: &GatedCfgAttr) -> cmp::Ordering {
        let to_tup = |s: &GatedCfgAttr| match *s {
            GatedCfgAttr::GatedCfg(ref gated_cfg) => {
                (gated_cfg.span.lo.0, gated_cfg.span.hi.0, gated_cfg.index)
            }
            GatedCfgAttr::GatedAttr(ref span) => {
                (span.lo.0, span.hi.0, GATED_CFGS.len())
            }
        };
        to_tup(self).cmp(&to_tup(other))
    }
}

impl PartialOrd for GatedCfgAttr {
    fn partial_cmp(&self, other: &GatedCfgAttr) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl GatedCfgAttr {
    pub fn check_and_emit(&self,
                          diagnostic: &Handler,
                          features: &Features,
                          codemap: &CodeMap) {
        match *self {
            GatedCfgAttr::GatedCfg(ref cfg) => {
                cfg.check_and_emit(diagnostic, features, codemap);
            }
            GatedCfgAttr::GatedAttr(span) => {
                if !features.stmt_expr_attributes {
                    emit_feature_err(diagnostic,
                                     "stmt_expr_attributes",
                                     span,
                                     GateIssue::Language,
                                     EXPLAIN_STMT_ATTR_SYNTAX);
                }
            }
        }
    }
}

impl GatedCfg {
    pub fn gate(cfg: &ast::MetaItem) -> Option<GatedCfg> {
        let name = cfg.name();
        GATED_CFGS.iter()
                  .position(|info| info.0 == name)
                  .map(|idx| {
                      GatedCfg {
                          span: cfg.span,
                          index: idx
                      }
                  })
    }
    fn check_and_emit(&self,
                      diagnostic: &Handler,
                      features: &Features,
                      codemap: &CodeMap) {
        let (cfg, feature, has_feature) = GATED_CFGS[self.index];
        if !has_feature(features) && !codemap.span_allows_unstable(self.span) {
            let explain = format!("`cfg({})` is experimental and subject to change", cfg);
            emit_feature_err(diagnostic, feature, self.span, GateIssue::Language, &explain);
        }
    }
}

/// A set of features to be used by later passes.
pub struct Features {
    pub unboxed_closures: bool,
    pub rustc_diagnostic_macros: bool,
    pub allow_quote: bool,
    pub allow_asm: bool,
    pub allow_log_syntax: bool,
    pub allow_concat_idents: bool,
    pub allow_trace_macros: bool,
    pub allow_internal_unstable: bool,
    pub allow_custom_derive: bool,
    pub allow_placement_in: bool,
    pub allow_box: bool,
    pub allow_pushpop_unsafe: bool,
    pub allow_inclusive_range: bool,
    pub simd_ffi: bool,
    pub unmarked_api: bool,
    /// spans of #![feature] attrs for stable language features. for error reporting
    pub declared_stable_lang_features: Vec<Span>,
    /// #![feature] attrs for non-language (library) features
    pub declared_lib_features: Vec<(InternedString, Span)>,
    pub const_fn: bool,
    pub const_indexing: bool,
    pub static_recursion: bool,
    pub default_type_parameter_fallback: bool,
    pub rustc_attrs: bool,
    pub type_macros: bool,
    pub cfg_target_feature: bool,
    pub cfg_target_vendor: bool,
    pub cfg_target_thread_local: bool,
    pub staged_api: bool,
    pub stmt_expr_attributes: bool,
    pub deprecated: bool,
    pub question_mark: bool,
    pub specialization: bool,
    pub pub_restricted: bool,
    pub structural_match: bool,
    pub plugin: bool,
    pub lang_items: bool,
    pub linkage: bool,
    pub thread_local: bool,
    pub on_unimplemented: bool,
    pub allocator: bool,
    pub needs_allocator: bool,
    pub fundamental: bool,
    pub linked_from: bool,
    pub naked: bool,
    pub no_debug: bool,
    pub omit_gdb_pretty_printer_section: bool,
    pub dropck_parametricity: bool,
    pub unwind_attributes: bool,
    pub prelude_import: bool,
    pub reflect: bool,
    pub no_core: bool,
    pub unsafe_no_drop_flag: bool,
    pub custom_derive: bool,
    pub custom_attribute: bool,
    pub asm: bool,
    pub log_syntax: bool,
    pub trace_macros: bool,
    pub concat_idents: bool,
    pub box_syntax: bool,
    pub placement_in_syntax: bool,
    pub non_ascii_idents: bool,
    pub macro_reexport: bool,
    pub link_args: bool,
    pub intrinsics: bool,
    pub platform_intrinsics: bool,
    pub abi_vectorcall: bool,
    pub plugin_registrar: bool,
    pub start: bool,
    pub main: bool,
    pub simd: bool,
    pub repr_simd: bool,
    pub optin_builtin_traits: bool,
    pub link_llvm_intrinsics: bool,
    pub type_ascription: bool,
    pub inclusive_range_syntax: bool,
    pub advanced_slice_patterns: bool,
    pub slice_patterns: bool,
    pub box_patterns: bool,
    pub associated_consts: bool,
    pub associated_type_defaults: bool
}

impl Features {
    pub fn new() -> Features {
        Features {
            unboxed_closures: false,
            rustc_diagnostic_macros: false,
            allow_quote: false,
            allow_asm: false,
            allow_log_syntax: false,
            allow_concat_idents: false,
            allow_trace_macros: false,
            allow_internal_unstable: false,
            allow_custom_derive: false,
            allow_placement_in: false,
            allow_box: false,
            allow_pushpop_unsafe: false,
            allow_inclusive_range: false,
            simd_ffi: false,
            unmarked_api: false,
            declared_stable_lang_features: Vec::new(),
            declared_lib_features: Vec::new(),
            const_fn: false,
            const_indexing: false,
            static_recursion: false,
            default_type_parameter_fallback: false,
            rustc_attrs: false,
            type_macros: false,
            cfg_target_feature: false,
            cfg_target_vendor: false,
            cfg_target_thread_local: false,
            staged_api: false,
            stmt_expr_attributes: false,
            deprecated: false,
            question_mark: false,
            specialization: false,
            pub_restricted: false,
            structural_match: false,
            plugin: false,
            lang_items: false,
            linkage: false,
            thread_local: false,
            on_unimplemented: false,
            allocator: false,
            needs_allocator: false,
            fundamental: false,
            linked_from: false,
            naked: false,
            no_debug: false,
            omit_gdb_pretty_printer_section: false,
            dropck_parametricity: false,
            unwind_attributes: false,
            prelude_import: false,
            reflect: false,
            no_core: false,
            unsafe_no_drop_flag: false,
            custom_derive: false,
            custom_attribute: false,
            asm: false,
            log_syntax: false,
            trace_macros: false,
            concat_idents: false,
            box_syntax: false,
            placement_in_syntax: false,
            non_ascii_idents: false,
            macro_reexport: false,
            link_args: false,
            intrinsics: false,
            platform_intrinsics: false,
            abi_vectorcall: false,
            plugin_registrar: false,
            start: false,
            main: false,
            simd: false,
            repr_simd: false,
            optin_builtin_traits: false,
            link_llvm_intrinsics: false,
            type_ascription: false,
            inclusive_range_syntax: false,
            advanced_slice_patterns: false,
            slice_patterns: false,
            box_patterns: false,
            associated_consts: false,
            associated_type_defaults: false,
        }
    }
}

const EXPLAIN_BOX_SYNTAX: &'static str =
    "box expression syntax is experimental; you can call `Box::new` instead.";

const EXPLAIN_PLACEMENT_IN: &'static str =
    "placement-in expression syntax is experimental and subject to change.";

const EXPLAIN_PUSHPOP_UNSAFE: &'static str =
    "push/pop_unsafe macros are experimental and subject to change.";

const EXPLAIN_STMT_ATTR_SYNTAX: &'static str =
    "attributes on non-item statements and expressions are experimental.";

pub fn check_for_box_syntax(f: Option<&Features>, diag: &Handler, span: Span) {
    if let Some(&Features { allow_box: true, .. }) = f {
        return;
    }
    emit_feature_err(diag, "box_syntax", span, GateIssue::Language, EXPLAIN_BOX_SYNTAX);
}

pub fn check_for_placement_in(f: Option<&Features>, diag: &Handler, span: Span) {
    if let Some(&Features { allow_placement_in: true, .. }) = f {
        return;
    }
    emit_feature_err(diag, "placement_in_syntax", span, GateIssue::Language, EXPLAIN_PLACEMENT_IN);
}

pub fn check_for_pushpop_syntax(f: Option<&Features>, diag: &Handler, span: Span) {
    if let Some(&Features { allow_pushpop_unsafe: true, .. }) = f {
        return;
    }
    emit_feature_err(diag, "pushpop_unsafe", span, GateIssue::Language, EXPLAIN_PUSHPOP_UNSAFE);
}

struct Context<'a> {
    features: Features,
    span_handler: &'a Handler,
    cm: &'a CodeMap,
    plugin_attributes: &'a [(String, AttributeType)],
}

macro_rules! gate_feature_fn {
    ($cx: expr, $has_feature: expr, $span: expr, $name: expr, $explain: expr) => {{
        let (cx, has_feature, span, name, explain) = ($cx, $has_feature, $span, $name, $explain);
        let has_feature: bool = has_feature(&$cx.features);
        debug!("gate_feature(feature = {:?}, span = {:?}); has? {}", name, span, has_feature);
        if !has_feature && !cx.cm.span_allows_unstable(span) {
            emit_feature_err(cx.span_handler, name, span, GateIssue::Language, explain);
        }
    }}
}

macro_rules! gate_feature {
    ($cx: expr, $feature: ident, $span: expr, $explain: expr) => {
        gate_feature_fn!($cx, |x:&Features| x.$feature, $span, stringify!($feature), $explain)
    }
}

impl<'a> Context<'a> {
    fn check_attribute(&self, attr: &ast::Attribute, is_macro: bool) {
        debug!("check_attribute(attr = {:?})", attr);
        let name = &*attr.name();
        for &(n, ty, ref gateage) in KNOWN_ATTRIBUTES {
            if n == name {
                if let &Gated(ref name, ref desc, ref has_feature) = gateage {
                    gate_feature_fn!(self, has_feature, attr.span, name, desc);
                }
                debug!("check_attribute: {:?} is known, {:?}, {:?}", name, ty, gateage);
                return;
            }
        }
        for &(ref n, ref ty) in self.plugin_attributes {
            if n == name {
                // Plugins can't gate attributes, so we don't check for it
                // unlike the code above; we only use this loop to
                // short-circuit to avoid the checks below
                debug!("check_attribute: {:?} is registered by a plugin, {:?}", name, ty);
                return;
            }
        }
        if name.starts_with("rustc_") {
            gate_feature!(self, rustc_attrs, attr.span,
                          "unless otherwise specified, attributes \
                           with the prefix `rustc_` \
                           are reserved for internal compiler diagnostics");
        } else if name.starts_with("derive_") {
            gate_feature!(self, custom_derive, attr.span,
                          "attributes of the form `#[derive_*]` are reserved \
                           for the compiler");
        } else {
            // Only run the custom attribute lint during regular
            // feature gate checking. Macro gating runs
            // before the plugin attributes are registered
            // so we skip this then
            if !is_macro {
                gate_feature!(self, custom_attribute, attr.span,
                              &format!("The attribute `{}` is currently \
                                        unknown to the compiler and \
                                        may have meaning \
                                        added to it in the future",
                                       name));
            }
        }
    }
}

fn find_lang_feature_issue(feature: &str) -> Option<u32> {
    let info = KNOWN_FEATURES.iter()
                              .find(|t| t.0 == feature)
                              .unwrap();
    let issue = info.2;
    if let Active = info.3 {
        // FIXME (#28244): enforce that active features have issue numbers
        // assert!(issue.is_some())
    }
    issue
}

pub enum GateIssue {
    Language,
    Library(Option<u32>)
}

pub fn emit_feature_err(diag: &Handler, feature: &str, span: Span, issue: GateIssue,
                        explain: &str) {
    let issue = match issue {
        GateIssue::Language => find_lang_feature_issue(feature),
        GateIssue::Library(lib) => lib,
    };

    let mut err = if let Some(n) = issue {
        diag.struct_span_err(span, &format!("{} (see issue #{})", explain, n))
    } else {
        diag.struct_span_err(span, explain)
    };

    // #23973: do not suggest `#![feature(...)]` if we are in beta/stable
    if option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_some() {
        err.emit();
        return;
    }
    err.fileline_help(span, &format!("add #![feature({})] to the \
                                      crate attributes to enable",
                                     feature));
    err.emit();
}

pub const EXPLAIN_ASM: &'static str =
    "inline assembly is not stable enough for use and is subject to change";

pub const EXPLAIN_LOG_SYNTAX: &'static str =
    "`log_syntax!` is not stable enough for use and is subject to change";

pub const EXPLAIN_CONCAT_IDENTS: &'static str =
    "`concat_idents` is not stable enough for use and is subject to change";

pub const EXPLAIN_TRACE_MACROS: &'static str =
    "`trace_macros` is not stable enough for use and is subject to change";
pub const EXPLAIN_ALLOW_INTERNAL_UNSTABLE: &'static str =
    "allow_internal_unstable side-steps feature gating and stability checks";

pub const EXPLAIN_CUSTOM_DERIVE: &'static str =
    "`#[derive]` for custom traits is not stable enough for use and is subject to change";

struct MacroVisitor<'a> {
    context: &'a Context<'a>
}

impl<'a, 'v> Visitor<'v> for MacroVisitor<'a> {
    fn visit_mac(&mut self, mac: &ast::Mac) {
        let path = &mac.node.path;
        let name = path.segments.last().unwrap().identifier.name.as_str();

        // Issue 22234: If you add a new case here, make sure to also
        // add code to catch the macro during or after expansion.
        //
        // We still keep this MacroVisitor (rather than *solely*
        // relying on catching cases during or after expansion) to
        // catch uses of these macros within conditionally-compiled
        // code, e.g. `#[cfg]`-guarded functions.

        if name == "asm" {
            gate_feature!(self.context, asm, path.span, EXPLAIN_ASM);
        }

        else if name == "log_syntax" {
            gate_feature!(self.context, log_syntax, path.span, EXPLAIN_LOG_SYNTAX);
        }

        else if name == "trace_macros" {
            gate_feature!(self.context, trace_macros, path.span, EXPLAIN_TRACE_MACROS);
        }

        else if name == "concat_idents" {
            gate_feature!(self.context, concat_idents, path.span, EXPLAIN_CONCAT_IDENTS);
        }
    }

    fn visit_attribute(&mut self, attr: &'v ast::Attribute) {
        self.context.check_attribute(attr, true);
    }

    fn visit_expr(&mut self, e: &ast::Expr) {
        // Issue 22181: overloaded-`box` and placement-`in` are
        // implemented via a desugaring expansion, so their feature
        // gates go into MacroVisitor since that works pre-expansion.
        //
        // Issue 22234: we also check during expansion as well.
        // But we keep these checks as a pre-expansion check to catch
        // uses in e.g. conditionalized code.

        if let ast::ExprKind::Box(_) = e.node {
            gate_feature!(self.context, box_syntax, e.span, EXPLAIN_BOX_SYNTAX);
        }

        if let ast::ExprKind::InPlace(..) = e.node {
            gate_feature!(self.context, placement_in_syntax, e.span, EXPLAIN_PLACEMENT_IN);
        }

        visit::walk_expr(self, e);
    }
}

struct PostExpansionVisitor<'a> {
    context: &'a Context<'a>,
}

macro_rules! gate_feature_post {
    ($cx: expr, $feature: ident, $span: expr, $explain: expr) => {{
        let (cx, span) = ($cx, $span);
        if !cx.context.cm.span_allows_unstable(span) {
            gate_feature!(cx.context, $feature, span, $explain)
        }
    }}
}

impl<'a, 'v> Visitor<'v> for PostExpansionVisitor<'a> {
    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        if !self.context.cm.span_allows_unstable(attr.span) {
            self.context.check_attribute(attr, false);
        }
    }

    fn visit_name(&mut self, sp: Span, name: ast::Name) {
        if !name.as_str().is_ascii() {
            gate_feature_post!(&self, non_ascii_idents, sp,
                               "non-ascii idents are not fully supported.");
        }
    }

    fn visit_item(&mut self, i: &ast::Item) {
        match i.node {
            ast::ItemKind::ExternCrate(_) => {
                if attr::contains_name(&i.attrs[..], "macro_reexport") {
                    gate_feature_post!(&self, macro_reexport, i.span,
                                       "macros reexports are experimental \
                                        and possibly buggy");
                }
            }

            ast::ItemKind::ForeignMod(ref foreign_module) => {
                if attr::contains_name(&i.attrs[..], "link_args") {
                    gate_feature_post!(&self, link_args, i.span,
                                      "the `link_args` attribute is not portable \
                                       across platforms, it is recommended to \
                                       use `#[link(name = \"foo\")]` instead")
                }
                match foreign_module.abi {
                    Abi::RustIntrinsic =>
                        gate_feature_post!(&self, intrinsics, i.span,
                                           "intrinsics are subject to change"),
                    Abi::PlatformIntrinsic => {
                        gate_feature_post!(&self, platform_intrinsics, i.span,
                                           "platform intrinsics are experimental \
                                            and possibly buggy")
                    },
                    Abi::Vectorcall => {
                        gate_feature_post!(&self, abi_vectorcall, i.span,
                                           "vectorcall is experimental and subject to change")
                    }
                    _ => ()
                }
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
                if attr::contains_name(&i.attrs[..], "simd") {
                    gate_feature_post!(&self, simd, i.span,
                                       "SIMD types are experimental and possibly buggy");
                    self.context.span_handler.span_warn(i.span,
                                                        "the `#[simd]` attribute is deprecated, \
                                                         use `#[repr(simd)]` instead");
                }
                for attr in &i.attrs {
                    if attr.name() == "repr" {
                        for item in attr.meta_item_list().unwrap_or(&[]) {
                            if item.name() == "simd" {
                                gate_feature_post!(&self, repr_simd, i.span,
                                                   "SIMD types are experimental \
                                                    and possibly buggy");

                            }
                        }
                    }
                }
            }

            ast::ItemKind::DefaultImpl(..) => {
                gate_feature_post!(&self, optin_builtin_traits,
                                   i.span,
                                   "default trait implementations are experimental \
                                    and possibly buggy");
            }

            ast::ItemKind::Impl(_, polarity, _, _, _, _) => {
                match polarity {
                    ast::ImplPolarity::Negative => {
                        gate_feature_post!(&self, optin_builtin_traits,
                                           i.span,
                                           "negative trait bounds are not yet fully implemented; \
                                            use marker types for now");
                    },
                    _ => {}
                }
            }

            _ => {}
        }

        visit::walk_item(self, i);
    }

    fn visit_variant_data(&mut self, s: &'v ast::VariantData, _: ast::Ident,
                          _: &'v ast::Generics, _: ast::NodeId, span: Span) {
        if s.fields().is_empty() {
            if s.is_tuple() {
                self.context.span_handler.struct_span_err(span, "empty tuple structs and enum \
                                                                 variants are not allowed, use \
                                                                 unit structs and enum variants \
                                                                 instead")
                                         .span_help(span, "remove trailing `()` to make a unit \
                                                           struct or unit enum variant")
                                         .emit();
            }
        }
        visit::walk_struct_def(self, s)
    }

    fn visit_foreign_item(&mut self, i: &ast::ForeignItem) {
        let links_to_llvm = match attr::first_attr_value_str_by_name(&i.attrs,
                                                                     "link_name") {
            Some(val) => val.starts_with("llvm."),
            _ => false
        };
        if links_to_llvm {
            gate_feature_post!(&self, link_llvm_intrinsics, i.span,
                              "linking to LLVM intrinsics is experimental");
        }

        visit::walk_foreign_item(self, i)
    }

    fn visit_expr(&mut self, e: &ast::Expr) {
        match e.node {
            ast::ExprKind::Box(_) => {
                gate_feature_post!(&self, box_syntax, e.span, EXPLAIN_BOX_SYNTAX);
            }
            ast::ExprKind::Type(..) => {
                gate_feature_post!(&self, type_ascription, e.span,
                                  "type ascription is experimental");
            }
            ast::ExprKind::Range(_, _, ast::RangeLimits::Closed) => {
                gate_feature_post!(&self, inclusive_range_syntax,
                                  e.span,
                                  "inclusive range syntax is experimental");
            }
            ast::ExprKind::Try(..) => {
                gate_feature_post!(&self, question_mark, e.span, "the `?` operator is not stable");
            }
            _ => {}
        }
        visit::walk_expr(self, e);
    }

    fn visit_pat(&mut self, pattern: &ast::Pat) {
        match pattern.node {
            PatKind::Vec(_, Some(_), ref last) if !last.is_empty() => {
                gate_feature_post!(&self, advanced_slice_patterns,
                                  pattern.span,
                                  "multiple-element slice matches anywhere \
                                   but at the end of a slice (e.g. \
                                   `[0, ..xs, 0]`) are experimental")
            }
            PatKind::Vec(..) => {
                gate_feature_post!(&self, slice_patterns,
                                  pattern.span,
                                  "slice pattern syntax is experimental");
            }
            PatKind::Box(..) => {
                gate_feature_post!(&self, box_patterns,
                                  pattern.span,
                                  "box pattern syntax is experimental");
            }
            _ => {}
        }
        visit::walk_pat(self, pattern)
    }

    fn visit_fn(&mut self,
                fn_kind: FnKind<'v>,
                fn_decl: &'v ast::FnDecl,
                block: &'v ast::Block,
                span: Span,
                _node_id: NodeId) {
        // check for const fn declarations
        match fn_kind {
            FnKind::ItemFn(_, _, _, ast::Constness::Const, _, _) => {
                gate_feature_post!(&self, const_fn, span, "const fn is unstable");
            }
            _ => {
                // stability of const fn methods are covered in
                // visit_trait_item and visit_impl_item below; this is
                // because default methods don't pass through this
                // point.
            }
        }

        match fn_kind {
            FnKind::ItemFn(_, _, _, _, abi, _) if abi == Abi::RustIntrinsic => {
                gate_feature_post!(&self, intrinsics,
                                  span,
                                  "intrinsics are subject to change")
            }
            FnKind::ItemFn(_, _, _, _, abi, _) |
            FnKind::Method(_, &ast::MethodSig { abi, .. }, _) => match abi {
                Abi::RustCall => {
                    gate_feature_post!(&self, unboxed_closures, span,
                        "rust-call ABI is subject to change");
                },
                Abi::Vectorcall => {
                    gate_feature_post!(&self, abi_vectorcall, span,
                        "vectorcall is experimental and subject to change");
                },
                _ => {}
            },
            _ => {}
        }
        visit::walk_fn(self, fn_kind, fn_decl, block, span);
    }

    fn visit_trait_item(&mut self, ti: &'v ast::TraitItem) {
        match ti.node {
            ast::TraitItemKind::Const(..) => {
                gate_feature_post!(&self, associated_consts,
                                  ti.span,
                                  "associated constants are experimental")
            }
            ast::TraitItemKind::Method(ref sig, _) => {
                if sig.constness == ast::Constness::Const {
                    gate_feature_post!(&self, const_fn, ti.span, "const fn is unstable");
                }
            }
            ast::TraitItemKind::Type(_, Some(_)) => {
                gate_feature_post!(&self, associated_type_defaults, ti.span,
                                  "associated type defaults are unstable");
            }
            _ => {}
        }
        visit::walk_trait_item(self, ti);
    }

    fn visit_impl_item(&mut self, ii: &'v ast::ImplItem) {
        if ii.defaultness == ast::Defaultness::Default {
            gate_feature_post!(&self, specialization,
                              ii.span,
                              "specialization is unstable");
        }

        match ii.node {
            ast::ImplItemKind::Const(..) => {
                gate_feature_post!(&self, associated_consts,
                                  ii.span,
                                  "associated constants are experimental")
            }
            ast::ImplItemKind::Method(ref sig, _) => {
                if sig.constness == ast::Constness::Const {
                    gate_feature_post!(&self, const_fn, ii.span, "const fn is unstable");
                }
            }
            _ => {}
        }
        visit::walk_impl_item(self, ii);
    }

    fn visit_vis(&mut self, vis: &'v ast::Visibility) {
        let span = match *vis {
            ast::Visibility::Crate(span) => span,
            ast::Visibility::Restricted { ref path, .. } => path.span,
            _ => return,
        };
        gate_feature_post!(&self, pub_restricted, span, "`pub(restricted)` syntax is experimental");
    }
}

fn check_crate_inner<F>(cm: &CodeMap, span_handler: &Handler,
                        krate: &ast::Crate,
                        plugin_attributes: &[(String, AttributeType)],
                        check: F)
                       -> Features
    where F: FnOnce(&mut Context, &ast::Crate)
{
    let mut accepted_features = Vec::new();
    let mut unknown_features = Vec::new();
    let mut enabled_features = Vec::new();

    for attr in &krate.attrs {
        if !attr.check_name("feature") {
            continue
        }

        match attr.meta_item_list() {
            None => {
                span_handler.span_err(attr.span, "malformed feature attribute, \
                                                  expected #![feature(...)]");
            }
            Some(list) => {
                for mi in list {
                    let name = match mi.node {
                        ast::MetaItemKind::Word(ref word) => (*word).clone(),
                        _ => {
                            span_handler.span_err(mi.span,
                                                  "malformed feature, expected just \
                                                   one word");
                            continue
                        }
                    };
                    match KNOWN_FEATURES.iter()
                                        .find(|& &(n, _, _, _)| name == n) {
                        Some(&(name, _, _, Active)) => {
                            enabled_features.push(name);
                        }
                        Some(&(_, _, _, Removed)) => {
                            span_handler.span_err(mi.span, "feature has been removed");
                        }
                        Some(&(_, _, _, Accepted)) => {
                            accepted_features.push(mi.span);
                        }
                        None => {
                            unknown_features.push((name, mi.span));
                        }
                    }
                }
            }
        }
    }

    let has_feature = |feature: &str| -> bool {
        enabled_features.iter().any(|&n| n == feature)
    };

    // FIXME (pnkfelix): Before adding the 99th entry below, change it
    // to a single-pass (instead of N calls to `.has_feature`).

    let features = Features {
        unboxed_closures: has_feature("unboxed_closures"),
        rustc_diagnostic_macros: has_feature("rustc_diagnostic_macros"),
        allow_quote: has_feature("quote"),
        allow_asm: has_feature("asm"),
        allow_log_syntax: has_feature("log_syntax"),
        allow_concat_idents: has_feature("concat_idents"),
        allow_trace_macros: has_feature("trace_macros"),
        allow_internal_unstable: has_feature("allow_internal_unstable"),
        allow_custom_derive: has_feature("custom_derive"),
        allow_placement_in: has_feature("placement_in_syntax"),
        allow_box: has_feature("box_syntax"),
        allow_pushpop_unsafe: has_feature("pushpop_unsafe"),
        allow_inclusive_range: has_feature("inclusive_range_syntax"),
        simd_ffi: has_feature("simd_ffi"),
        unmarked_api: has_feature("unmarked_api"),
        declared_stable_lang_features: accepted_features,
        declared_lib_features: unknown_features,
        const_fn: has_feature("const_fn"),
        const_indexing: has_feature("const_indexing"),
        static_recursion: has_feature("static_recursion"),
        default_type_parameter_fallback: has_feature("default_type_parameter_fallback"),
        rustc_attrs: has_feature("rustc_attrs"),
        type_macros: has_feature("type_macros"),
        cfg_target_feature: has_feature("cfg_target_feature"),
        cfg_target_vendor: has_feature("cfg_target_vendor"),
        cfg_target_thread_local: has_feature("cfg_target_thread_local"),
        staged_api: has_feature("staged_api"),
        stmt_expr_attributes: has_feature("stmt_expr_attributes"),
        deprecated: has_feature("deprecated"),
        question_mark: has_feature("question_mark"),
        specialization: has_feature("specialization"),
        pub_restricted: has_feature("pub_restricted"),
        structural_match: has_feature("bool"),
        plugin: has_feature("plugin"),
        lang_items: has_feature("lang_items"),
        linkage: has_feature("linkage"),
        thread_local: has_feature("thread_local"),
        on_unimplemented: has_feature("on_unimplemented"),
        allocator: has_feature("allocator"),
        needs_allocator: has_feature("needs_allocator"),
        fundamental: has_feature("fundamental"),
        linked_from: has_feature("linked_from"),
        naked: has_feature("naked"),
        no_debug: has_feature("no_debug"),
        omit_gdb_pretty_printer_section: has_feature("omit_gdb_pretty_printer_section"),
        dropck_parametricity: has_feature("dropck_parametricity"),
        unwind_attributes: has_feature("unwind_attributes"),
        prelude_import: has_feature("prelude_import"),
        reflect: has_feature("reflect"),
        no_core: has_feature("no_core"),
        unsafe_no_drop_flag: has_feature("unsafe_no_drop_flag"),
        custom_derive: has_feature("custom_derive"),
        custom_attribute: has_feature("custom_attribute"),
        asm: has_feature("asm"),
        log_syntax: has_feature("log_syntax"),
        trace_macros: has_feature("trace_macros"),
        concat_idents: has_feature("concat_idents"),
        box_syntax: has_feature("box_syntax"),
        placement_in_syntax: has_feature("placement_in_syntax"),
        non_ascii_idents: has_feature("non_ascii_idents"),
        macro_reexport: has_feature("macro_reexport"),
        link_args: has_feature("link_args"),
        intrinsics: has_feature("intrinsics"),
        platform_intrinsics: has_feature("platform_intrinsics"),
        abi_vectorcall: has_feature("abi_vectorcall"),
        plugin_registrar: has_feature("plugin_registrar"),
        start: has_feature("start"),
        main: has_feature("main"),
        simd: has_feature("simd"),
        repr_simd: has_feature("repr_simd"),
        optin_builtin_traits: has_feature("optin_builtin_traits"),
        link_llvm_intrinsics: has_feature("link_llvm_intrinsics"),
        type_ascription: has_feature("type_ascription"),
        inclusive_range_syntax: has_feature("inclusive_range_syntax"),
        advanced_slice_patterns: has_feature("advanced_slice_patterns"),
        slice_patterns: has_feature("slice_patterns"),
        box_patterns: has_feature("box_patterns"),
        associated_consts: has_feature("associated_consts"),
        associated_type_defaults: has_feature("associated_type_defaults"),
    };

    let mut cx = Context {
        features: features,
        span_handler: span_handler,
        cm: cm,
        plugin_attributes: plugin_attributes,
    };

    check(&mut cx, krate);
    cx.features
}

pub fn check_crate_macros(cm: &CodeMap, span_handler: &Handler, krate: &ast::Crate)
-> Features {
    check_crate_inner(cm, span_handler, krate, &[] as &'static [_],
                      |ctx, krate| visit::walk_crate(&mut MacroVisitor { context: ctx }, krate))
}

pub fn check_crate(cm: &CodeMap, span_handler: &Handler, krate: &ast::Crate,
                   plugin_attributes: &[(String, AttributeType)],
                   unstable: UnstableFeatures) -> Features
{
    maybe_stage_features(span_handler, krate, unstable);

    check_crate_inner(cm, span_handler, krate, plugin_attributes,
                      |ctx, krate| visit::walk_crate(&mut PostExpansionVisitor { context: ctx },
                                                     krate))
}

#[derive(Clone, Copy)]
pub enum UnstableFeatures {
    /// Hard errors for unstable features are active, as on
    /// beta/stable channels.
    Disallow,
    /// Allow features to me activated, as on nightly.
    Allow,
    /// Errors are bypassed for bootstrapping. This is required any time
    /// during the build that feature-related lints are set to warn or above
    /// because the build turns on warnings-as-errors and uses lots of unstable
    /// features. As a result, this is always required for building Rust itself.
    Cheat
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
                let ref msg = format!("#[feature] may not be used on the {} release channel",
                                      release_channel);
                span_handler.span_err(attr.span, msg);
            }
        }
    }
}
