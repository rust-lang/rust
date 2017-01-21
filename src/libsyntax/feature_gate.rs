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

use abi::Abi;
use ast::{self, NodeId, PatKind};
use attr;
use codemap::{CodeMap, Spanned};
use syntax_pos::Span;
use errors::{DiagnosticBuilder, Handler, FatalError};
use visit::{self, FnKind, Visitor};
use parse::ParseSess;
use symbol::Symbol;

use std::ascii::AsciiExt;
use std::env;

macro_rules! setter {
    ($field: ident) => {{
        fn f(features: &mut Features) -> &mut bool {
            &mut features.$field
        }
        f as fn(&mut Features) -> &mut bool
    }}
}

macro_rules! declare_features {
    ($((active, $feature: ident, $ver: expr, $issue: expr),)+) => {
        /// Represents active features that are currently being implemented or
        /// currently being considered for addition/removal.
        const ACTIVE_FEATURES: &'static [(&'static str, &'static str,
                                          Option<u32>, fn(&mut Features) -> &mut bool)] = &[
            $((stringify!($feature), $ver, $issue, setter!($feature))),+
        ];

        /// A set of features to be used by later passes.
        pub struct Features {
            /// #![feature] attrs for stable language features, for error reporting
            pub declared_stable_lang_features: Vec<(Symbol, Span)>,
            /// #![feature] attrs for non-language (library) features
            pub declared_lib_features: Vec<(Symbol, Span)>,
            $(pub $feature: bool),+
        }

        impl Features {
            pub fn new() -> Features {
                Features {
                    declared_stable_lang_features: Vec::new(),
                    declared_lib_features: Vec::new(),
                    $($feature: false),+
                }
            }
        }
    };

    ($((removed, $feature: ident, $ver: expr, $issue: expr),)+) => {
        /// Represents features which has since been removed (it was once Active)
        const REMOVED_FEATURES: &'static [(&'static str, &'static str, Option<u32>)] = &[
            $((stringify!($feature), $ver, $issue)),+
        ];
    };

    ($((accepted, $feature: ident, $ver: expr, $issue: expr),)+) => {
        /// Those language feature has since been Accepted (it was once Active)
        const ACCEPTED_FEATURES: &'static [(&'static str, &'static str, Option<u32>)] = &[
            $((stringify!($feature), $ver, $issue)),+
        ];
    }
}

// If you change this list without updating src/doc/reference.md, @cmr will be sad
// Don't ever remove anything from this list; set them to 'Removed'.
// The version numbers here correspond to the version in which the current status
// was set. This is most important for knowing when a particular feature became
// stable (active).
// NB: The featureck.py script parses this information directly out of the source
// so take care when modifying it.

declare_features! (
    (active, asm, "1.0.0", Some(29722)),
    (active, concat_idents, "1.0.0", Some(29599)),
    (active, link_args, "1.0.0", Some(29596)),
    (active, log_syntax, "1.0.0", Some(29598)),
    (active, non_ascii_idents, "1.0.0", Some(28979)),
    (active, plugin_registrar, "1.0.0", Some(29597)),
    (active, thread_local, "1.0.0", Some(29594)),
    (active, trace_macros, "1.0.0", Some(29598)),

    // rustc internal, for now:
    (active, intrinsics, "1.0.0", None),
    (active, lang_items, "1.0.0", None),

    (active, link_llvm_intrinsics, "1.0.0", Some(29602)),
    (active, linkage, "1.0.0", Some(29603)),
    (active, quote, "1.0.0", Some(29601)),
    (active, simd, "1.0.0", Some(27731)),


    // rustc internal
    (active, rustc_diagnostic_macros, "1.0.0", None),
    (active, advanced_slice_patterns, "1.0.0", Some(23121)),
    (active, box_syntax, "1.0.0", Some(27779)),
    (active, placement_in_syntax, "1.0.0", Some(27779)),
    (active, reflect, "1.0.0", Some(27749)),
    (active, unboxed_closures, "1.0.0", Some(29625)),

    (active, allocator, "1.0.0", Some(27389)),
    (active, fundamental, "1.0.0", Some(29635)),
    (active, main, "1.0.0", Some(29634)),
    (active, needs_allocator, "1.4.0", Some(27389)),
    (active, on_unimplemented, "1.0.0", Some(29628)),
    (active, plugin, "1.0.0", Some(29597)),
    (active, simd_ffi, "1.0.0", Some(27731)),
    (active, start, "1.0.0", Some(29633)),
    (active, structural_match, "1.8.0", Some(31434)),
    (active, panic_runtime, "1.10.0", Some(32837)),
    (active, needs_panic_runtime, "1.10.0", Some(32837)),

    // OIBIT specific features
    (active, optin_builtin_traits, "1.0.0", Some(13231)),

    // macro reexport needs more discussion and stabilization
    (active, macro_reexport, "1.0.0", Some(29638)),

    // Allows use of #[staged_api]
    // rustc internal
    (active, staged_api, "1.0.0", None),

    // Allows using #![no_core]
    (active, no_core, "1.3.0", Some(29639)),

    // Allows using `box` in patterns; RFC 469
    (active, box_patterns, "1.0.0", Some(29641)),

    // Allows using the unsafe_destructor_blind_to_params attribute;
    // RFC 1238
    (active, dropck_parametricity, "1.3.0", Some(28498)),

    // Allows using the may_dangle attribute; RFC 1327
    (active, dropck_eyepatch, "1.10.0", Some(34761)),

    // Allows the use of custom attributes; RFC 572
    (active, custom_attribute, "1.0.0", Some(29642)),

    // Allows the use of #[derive(Anything)] as sugar for
    // #[derive_Anything].
    (active, custom_derive, "1.0.0", Some(29644)),

    // Allows the use of rustc_* attributes; RFC 572
    (active, rustc_attrs, "1.0.0", Some(29642)),

    // Allows the use of #[allow_internal_unstable]. This is an
    // attribute on macro_rules! and can't use the attribute handling
    // below (it has to be checked before expansion possibly makes
    // macros disappear).
    //
    // rustc internal
    (active, allow_internal_unstable, "1.0.0", None),

    // #23121. Array patterns have some hazards yet.
    (active, slice_patterns, "1.0.0", Some(23121)),

    // Allows the definition of associated constants in `trait` or `impl`
    // blocks.
    (active, associated_consts, "1.0.0", Some(29646)),

    // Allows the definition of `const fn` functions.
    (active, const_fn, "1.2.0", Some(24111)),

    // Allows indexing into constant arrays.
    (active, const_indexing, "1.4.0", Some(29947)),

    // Allows using #[prelude_import] on glob `use` items.
    //
    // rustc internal
    (active, prelude_import, "1.2.0", None),

    // Allows the definition recursive static items.
    (active, static_recursion, "1.3.0", Some(29719)),

    // Allows default type parameters to influence type inference.
    (active, default_type_parameter_fallback, "1.3.0", Some(27336)),

    // Allows associated type defaults
    (active, associated_type_defaults, "1.2.0", Some(29661)),

    // allow `repr(simd)`, and importing the various simd intrinsics
    (active, repr_simd, "1.4.0", Some(27731)),

    // Allows cfg(target_feature = "...").
    (active, cfg_target_feature, "1.4.0", Some(29717)),

    // allow `extern "platform-intrinsic" { ... }`
    (active, platform_intrinsics, "1.4.0", Some(27731)),

    // allow `#[unwind]`
    // rust runtime internal
    (active, unwind_attributes, "1.4.0", None),

    // allow the use of `#[naked]` on functions.
    (active, naked_functions, "1.9.0", Some(32408)),

    // allow `#[no_debug]`
    (active, no_debug, "1.5.0", Some(29721)),

    // allow `#[omit_gdb_pretty_printer_section]`
    // rustc internal.
    (active, omit_gdb_pretty_printer_section, "1.5.0", None),

    // Allows cfg(target_vendor = "...").
    (active, cfg_target_vendor, "1.5.0", Some(29718)),

    // Allow attributes on expressions and non-item statements
    (active, stmt_expr_attributes, "1.6.0", Some(15701)),

    // allow using type ascription in expressions
    (active, type_ascription, "1.6.0", Some(23416)),

    // Allows cfg(target_thread_local)
    (active, cfg_target_thread_local, "1.7.0", Some(29594)),

    // rustc internal
    (active, abi_vectorcall, "1.7.0", None),

    // a...b and ...b
    (active, inclusive_range_syntax, "1.7.0", Some(28237)),

    // impl specialization (RFC 1210)
    (active, specialization, "1.7.0", Some(31844)),

    // pub(restricted) visibilities (RFC 1422)
    (active, pub_restricted, "1.9.0", Some(32409)),

    // Allow Drop types in statics/const functions (RFC 1440)
    (active, drop_types_in_const, "1.9.0", Some(33156)),

    // Allows cfg(target_has_atomic = "...").
    (active, cfg_target_has_atomic, "1.9.0", Some(32976)),

    // Allows `impl Trait` in function return types.
    (active, conservative_impl_trait, "1.12.0", Some(34511)),

    // Permits numeric fields in struct expressions and patterns.
    (active, relaxed_adts, "1.12.0", Some(35626)),

    // The `!` type
    (active, never_type, "1.13.0", Some(35121)),

    // Allows all literals in attribute lists and values of key-value pairs.
    (active, attr_literals, "1.13.0", Some(34981)),

    // Allows the sysV64 ABI to be specified on all platforms
    // instead of just the platforms on which it is the C ABI
    (active, abi_sysv64, "1.13.0", Some(36167)),

    // Allows untagged unions `union U { ... }`
    (active, untagged_unions, "1.13.0", Some(32836)),

    // elide `'static` lifetimes in `static`s and `const`s
    (active, static_in_const, "1.13.0", Some(35897)),

    // Used to identify the `compiler_builtins` crate
    // rustc internal
    (active, compiler_builtins, "1.13.0", None),

    // Allows attributes on lifetime/type formal parameters in generics (RFC 1327)
    (active, generic_param_attrs, "1.11.0", Some(34761)),

    // Allows field shorthands (`x` meaning `x: x`) in struct literal expressions.
    (active, field_init_shorthand, "1.14.0", Some(37340)),

    // The #![windows_subsystem] attribute
    (active, windows_subsystem, "1.14.0", Some(37499)),

    // Allows using `Self` and associated types in struct expressions and patterns.
    (active, more_struct_aliases, "1.14.0", Some(37544)),


    // Allows #[link(..., cfg(..))]
    (active, link_cfg, "1.14.0", Some(37406)),

    (active, use_extern_macros, "1.15.0", Some(35896)),

    // Allows `break {expr}` with a value inside `loop`s.
    (active, loop_break_value, "1.14.0", Some(37339)),

    // Allows #[target_feature(...)]
    (active, target_feature, "1.15.0", None),

    // `extern "ptx-*" fn()`
    (active, abi_ptx, "1.15.0", None),

    // The `i128` type
    (active, i128_type, "1.16.0", Some(35118)),

    // The `unadjusted` ABI. Perma unstable.
    (active, abi_unadjusted, "1.16.0", None),

    // Macros 1.1
    (active, proc_macro, "1.16.0", Some(35900)),

    // Allows attributes on struct literal fields.
    (active, struct_field_attributes, "1.16.0", Some(38814)),

    // `extern "msp430-interrupt" fn()`
    (active, abi_msp430_interrupt, "1.16.0", Some(38487)),
);

declare_features! (
    (removed, import_shadowing, "1.0.0", None),
    (removed, managed_boxes, "1.0.0", None),
    // Allows use of unary negate on unsigned integers, e.g. -e for e: u8
    (removed, negate_unsigned, "1.0.0", Some(29645)),
    // A way to temporarily opt out of opt in copy. This will *never* be accepted.
    (removed, opt_out_copy, "1.0.0", None),
    (removed, quad_precision_float, "1.0.0", None),
    (removed, struct_inherit, "1.0.0", None),
    (removed, test_removed_feature, "1.0.0", None),
    (removed, visible_private_types, "1.0.0", None),
    (removed, unsafe_no_drop_flag, "1.0.0", None),
    // Allows using items which are missing stability attributes
    // rustc internal
    (removed, unmarked_api, "1.0.0", None),
    (removed, pushpop_unsafe, "1.2.0", None),
);

declare_features! (
    (accepted, associated_types, "1.0.0", None),
    // allow overloading augmented assignment operations like `a += b`
    (accepted, augmented_assignments, "1.8.0", Some(28235)),
    // allow empty structs and enum variants with braces
    (accepted, braced_empty_structs, "1.8.0", Some(29720)),
    (accepted, default_type_params, "1.0.0", None),
    (accepted, globs, "1.0.0", None),
    (accepted, if_let, "1.0.0", None),
    // A temporary feature gate used to enable parser extensions needed
    // to bootstrap fix for #5723.
    (accepted, issue_5723_bootstrap, "1.0.0", None),
    (accepted, macro_rules, "1.0.0", None),
    // Allows using #![no_std]
    (accepted, no_std, "1.6.0", None),
    (accepted, slicing_syntax, "1.0.0", None),
    (accepted, struct_variant, "1.0.0", None),
    // These are used to test this portion of the compiler, they don't actually
    // mean anything
    (accepted, test_accepted_feature, "1.0.0", None),
    (accepted, tuple_indexing, "1.0.0", None),
    // Allows macros to appear in the type position.
    (accepted, type_macros, "1.13.0", Some(27245)),
    (accepted, while_let, "1.0.0", None),
    // Allows `#[deprecated]` attribute
    (accepted, deprecated, "1.9.0", Some(29935)),
    // `expr?`
    (accepted, question_mark, "1.13.0", Some(31436)),
    // Allows `..` in tuple (struct) patterns
    (accepted, dotdot_in_tuple_patterns, "1.14.0", Some(33627)),
    (accepted, item_like_imports, "1.14.0", Some(35120)),
);
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

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Stability {
    Unstable,
    // Argument is tracking issue link.
    Deprecated(&'static str),
}

// fn() is not Debug
impl ::std::fmt::Debug for AttributeGate {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            Gated(ref stab, ref name, ref expl, _) =>
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

pub fn is_builtin_attr(attr: &ast::Attribute) -> bool {
    BUILTIN_ATTRIBUTES.iter().any(|&(builtin_name, _, _)| attr.check_name(builtin_name))
}

// Attributes that have a special meaning to rustc or rustdoc
pub const BUILTIN_ATTRIBUTES: &'static [(&'static str, AttributeType, AttributeGate)] = &[
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
    ("structural_match", Whitelisted, Gated(Stability::Unstable,
                                            "structural_match",
                                            "the semantics of constant patterns is \
                                             not yet settled",
                                            cfg_fn!(structural_match))),

    // Not used any more, but we can't feature gate it
    ("no_stack_check", Normal, Ungated),

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
                                         not currently handle destructors. There is no \
                                         corresponding `#[task_local]` mapping to the task \
                                         model",
                                        cfg_fn!(thread_local))),

    ("rustc_on_unimplemented", Normal, Gated(Stability::Unstable,
                                             "on_unimplemented",
                                             "the `#[rustc_on_unimplemented]` attribute \
                                              is an experimental feature",
                                             cfg_fn!(on_unimplemented))),
    ("allocator", Whitelisted, Gated(Stability::Unstable,
                                     "allocator",
                                     "the `#[allocator]` attribute is an experimental feature",
                                     cfg_fn!(allocator))),
    ("needs_allocator", Normal, Gated(Stability::Unstable,
                                      "needs_allocator",
                                      "the `#[needs_allocator]` \
                                       attribute is an experimental \
                                       feature",
                                      cfg_fn!(needs_allocator))),
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
    ("rustc_variance", Normal, Gated(Stability::Unstable,
                                     "rustc_attrs",
                                     "the `#[rustc_variance]` attribute \
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
    ("rustc_metadata_dirty", Whitelisted, Gated(Stability::Unstable,
                                                "rustc_attrs",
                                                "the `#[rustc_metadata_dirty]` attribute \
                                                 is just used for rustc unit tests \
                                                 and will never be stable",
                                                 cfg_fn!(rustc_attrs))),
    ("rustc_metadata_clean", Whitelisted, Gated(Stability::Unstable,
                                                "rustc_attrs",
                                                "the `#[rustc_metadata_clean]` attribute \
                                                 is just used for rustc unit tests \
                                                 and will never be stable",
                                                 cfg_fn!(rustc_attrs))),
    ("rustc_partition_reused", Whitelisted, Gated(Stability::Unstable,
                                                  "rustc_attrs",
                                                  "this attribute \
                                                   is just used for rustc unit tests \
                                                   and will never be stable",
                                                  cfg_fn!(rustc_attrs))),
    ("rustc_partition_translated", Whitelisted, Gated(Stability::Unstable,
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
    ("rustc_move_fragments", Normal, Gated(Stability::Unstable,
                                           "rustc_attrs",
                                           "the `#[rustc_move_fragments]` attribute \
                                            is just used for rustc unit tests \
                                            and will never be stable",
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
    ("compiler_builtins", Whitelisted, Gated(Stability::Unstable,
                                             "compiler_builtins",
                                             "the `#[compiler_builtins]` attribute is used to \
                                              identify the `compiler_builtins` crate which \
                                              contains compiler-rt intrinsics and will never be \
                                              stable",
                                          cfg_fn!(compiler_builtins))),

    ("allow_internal_unstable", Normal, Gated(Stability::Unstable,
                                              "allow_internal_unstable",
                                              EXPLAIN_ALLOW_INTERNAL_UNSTABLE,
                                              cfg_fn!(allow_internal_unstable))),

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

    // FIXME: #14406 these are processed in trans, which happens after the
    // lint pass
    ("cold", Whitelisted, Ungated),
    ("naked", Whitelisted, Gated(Stability::Unstable,
                                 "naked_functions",
                                 "the `#[naked]` attribute \
                                  is an experimental feature",
                                 cfg_fn!(naked_functions))),
    ("target_feature", Whitelisted, Gated(
        Stability::Unstable, "target_feature",
        "the `#[target_feature]` attribute is an experimental feature",
        cfg_fn!(target_feature))),
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
        "the `#[no_debug]` attribute is an experimental feature",
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
    ("rustc_reflect_like", Whitelisted, Gated(Stability::Unstable,
                                              "reflect",
                                              "defining reflective traits is still evolving",
                                              cfg_fn!(reflect))),

    ("windows_subsystem", Whitelisted, Gated(Stability::Unstable,
                                             "windows_subsystem",
                                             "the windows subsystem attribute \
                                              is currently unstable",
                                             cfg_fn!(windows_subsystem))),

    ("proc_macro_attribute", Normal, Gated(Stability::Unstable,
                                           "proc_macro",
                                           "attribute proc macros are currently unstable",
                                           cfg_fn!(proc_macro))),

    ("rustc_derive_registrar", Normal, Gated(Stability::Unstable,
                                             "rustc_derive_registrar",
                                             "used internally by rustc",
                                             cfg_fn!(rustc_attrs))),

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
const GATED_CFGS: &'static [(&'static str, &'static str, fn(&Features) -> bool)] = &[
    // (name in cfg, feature, function to check if the feature is enabled)
    ("target_feature", "cfg_target_feature", cfg_fn!(cfg_target_feature)),
    ("target_vendor", "cfg_target_vendor", cfg_fn!(cfg_target_vendor)),
    ("target_thread_local", "cfg_target_thread_local", cfg_fn!(cfg_target_thread_local)),
    ("target_has_atomic", "cfg_target_has_atomic", cfg_fn!(cfg_target_has_atomic)),
];

#[derive(Debug, Eq, PartialEq)]
pub struct GatedCfg {
    span: Span,
    index: usize,
}

impl GatedCfg {
    pub fn gate(cfg: &ast::MetaItem) -> Option<GatedCfg> {
        let name = &*cfg.name().as_str();
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
        if !has_feature(features) && !sess.codemap().span_allows_unstable(self.span) {
            let explain = format!("`cfg({})` is experimental and subject to change", cfg);
            emit_feature_err(sess, feature, self.span, GateIssue::Language, &explain);
        }
    }
}

struct Context<'a> {
    features: &'a Features,
    parse_sess: &'a ParseSess,
    cm: &'a CodeMap,
    plugin_attributes: &'a [(String, AttributeType)],
}

macro_rules! gate_feature_fn {
    ($cx: expr, $has_feature: expr, $span: expr, $name: expr, $explain: expr) => {{
        let (cx, has_feature, span, name, explain) = ($cx, $has_feature, $span, $name, $explain);
        let has_feature: bool = has_feature(&$cx.features);
        debug!("gate_feature(feature = {:?}, span = {:?}); has? {}", name, span, has_feature);
        if !has_feature && !cx.cm.span_allows_unstable(span) {
            emit_feature_err(cx.parse_sess, name, span, GateIssue::Language, explain);
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
        let name = &*attr.name().as_str();
        for &(n, ty, ref gateage) in BUILTIN_ATTRIBUTES {
            if n == name {
                if let &Gated(_, ref name, ref desc, ref has_feature) = gateage {
                    gate_feature_fn!(self, has_feature, attr.span, name, desc);
                }
                debug!("check_attribute: {:?} is builtin, {:?}, {:?}", name, ty, gateage);
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
            gate_feature!(self, custom_derive, attr.span, EXPLAIN_DERIVE_UNDERSCORE);
        } else if attr::is_known(attr) {
            debug!("check_attribute: {:?} is known", name);
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

pub fn check_attribute(attr: &ast::Attribute, parse_sess: &ParseSess,
                       cm: &CodeMap, features: &Features) {
    let cx = Context {
        features: features, parse_sess: parse_sess,
        cm: cm, plugin_attributes: &[]
    };
    cx.check_attribute(attr, true);
}

pub fn find_lang_feature_accepted_version(feature: &str) -> Option<&'static str> {
    ACCEPTED_FEATURES.iter().find(|t| t.0 == feature).map(|t| t.1)
}

fn find_lang_feature_issue(feature: &str) -> Option<u32> {
    if let Some(info) = ACTIVE_FEATURES.iter().find(|t| t.0 == feature) {
        let issue = info.2;
        // FIXME (#28244): enforce that active features have issue numbers
        // assert!(issue.is_some())
        issue
    } else {
        // search in Accepted or Removed features
        match ACCEPTED_FEATURES.iter().chain(REMOVED_FEATURES).find(|t| t.0 == feature) {
            Some(&(_, _, issue)) => issue,
            None => panic!("Feature `{}` is not declared anywhere", feature),
        }
    }
}

pub enum GateIssue {
    Language,
    Library(Option<u32>)
}

pub fn emit_feature_err(sess: &ParseSess, feature: &str, span: Span, issue: GateIssue,
                        explain: &str) {
    feature_err(sess, feature, span, issue, explain).emit();
}

pub fn feature_err<'a>(sess: &'a ParseSess, feature: &str, span: Span, issue: GateIssue,
                   explain: &str) -> DiagnosticBuilder<'a> {
    let diag = &sess.span_diagnostic;

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
    if sess.unstable_features.is_nightly_build() {
        err.help(&format!("add #![feature({})] to the \
                           crate attributes to enable",
                          feature));
    }

    err
}

const EXPLAIN_BOX_SYNTAX: &'static str =
    "box expression syntax is experimental; you can call `Box::new` instead.";

pub const EXPLAIN_STMT_ATTR_SYNTAX: &'static str =
    "attributes on non-item statements and expressions are experimental.";

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
    "`#[derive]` for custom traits is not stable enough for use. It is deprecated and will \
     be removed in v1.15";

pub const EXPLAIN_DEPR_CUSTOM_DERIVE: &'static str =
    "`#[derive]` for custom traits is deprecated and will be removed in v1.15. Prefer using \
     procedural macro custom derive";

pub const EXPLAIN_DERIVE_UNDERSCORE: &'static str =
    "attributes of the form `#[derive_*]` are reserved for the compiler";

pub const EXPLAIN_PLACEMENT_IN: &'static str =
    "placement-in expression syntax is experimental and subject to change.";

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
            Abi::RustCall => {
                gate_feature_post!(&self, unboxed_closures, span,
                                   "rust-call ABI is subject to change");
            },
            Abi::SysV64 => {
                gate_feature_post!(&self, abi_sysv64, span,
                                   "sysv64 ABI is experimental and subject to change");
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
            // Stable
            Abi::Cdecl |
            Abi::Stdcall |
            Abi::Fastcall |
            Abi::Aapcs |
            Abi::Win64 |
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
                MetaItem(ref mi) => contains_novel_literal(&mi),
                Literal(_) => true,
            }
        }),
    }
}

fn starts_with_digit(s: &str) -> bool {
    s.as_bytes().first().cloned().map_or(false, |b| b >= b'0' && b <= b'9')
}

impl<'a> Visitor<'a> for PostExpansionVisitor<'a> {
    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        if !self.context.cm.span_allows_unstable(attr.span) {
            // check for gated attributes
            self.context.check_attribute(attr, false);
        }

        if contains_novel_literal(&attr.value) {
            gate_feature_post!(&self, attr_literals, attr.span,
                               "non-string literals in attributes, or string \
                               literals in top-level positions, are experimental");
        }
    }

    fn visit_name(&mut self, sp: Span, name: ast::Name) {
        if !name.as_str().is_ascii() {
            gate_feature_post!(&self, non_ascii_idents, sp,
                               "non-ascii idents are not fully supported.");
        }
    }

    fn visit_item(&mut self, i: &'a ast::Item) {
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
                if attr::contains_name(&i.attrs[..], "simd") {
                    gate_feature_post!(&self, simd, i.span,
                                       "SIMD types are experimental and possibly buggy");
                    self.context.parse_sess.span_diagnostic.span_warn(i.span,
                                                                      "the `#[simd]` attribute \
                                                                       is deprecated, use \
                                                                       `#[repr(simd)]` instead");
                }
                for attr in &i.attrs {
                    if attr.name() == "repr" {
                        for item in attr.meta_item_list().unwrap_or(&[]) {
                            if item.check_name("simd") {
                                gate_feature_post!(&self, repr_simd, i.span,
                                                   "SIMD types are experimental \
                                                    and possibly buggy");

                            }
                        }
                    }
                }
            }

            ast::ItemKind::Union(..) => {
                gate_feature_post!(&self, untagged_unions,
                                   i.span,
                                   "unions are unstable and possibly buggy");
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

    fn visit_foreign_item(&mut self, i: &'a ast::ForeignItem) {
        let links_to_llvm = match attr::first_attr_value_str_by_name(&i.attrs, "link_name") {
            Some(val) => val.as_str().starts_with("llvm."),
            _ => false
        };
        if links_to_llvm {
            gate_feature_post!(&self, link_llvm_intrinsics, i.span,
                              "linking to LLVM intrinsics is experimental");
        }

        visit::walk_foreign_item(self, i)
    }

    fn visit_ty(&mut self, ty: &'a ast::Ty) {
        match ty.node {
            ast::TyKind::BareFn(ref bare_fn_ty) => {
                self.check_abi(bare_fn_ty.abi, ty.span);
            }
            ast::TyKind::ImplTrait(..) => {
                gate_feature_post!(&self, conservative_impl_trait, ty.span,
                                   "`impl Trait` is experimental");
            }
            ast::TyKind::Never => {
                gate_feature_post!(&self, never_type, ty.span,
                                   "The `!` type is experimental");
            },
            _ => {}
        }
        visit::walk_ty(self, ty)
    }

    fn visit_fn_ret_ty(&mut self, ret_ty: &'a ast::FunctionRetTy) {
        if let ast::FunctionRetTy::Ty(ref output_ty) = *ret_ty {
            match output_ty.node {
                ast::TyKind::Never => return,
                _ => (),
            };
            self.visit_ty(output_ty)
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
            ast::ExprKind::Range(_, _, ast::RangeLimits::Closed) => {
                gate_feature_post!(&self, inclusive_range_syntax,
                                  e.span,
                                  "inclusive range syntax is experimental");
            }
            ast::ExprKind::InPlace(..) => {
                gate_feature_post!(&self, placement_in_syntax, e.span, EXPLAIN_PLACEMENT_IN);
            }
            ast::ExprKind::Struct(_, ref fields, _) => {
                for field in fields {
                    if field.is_shorthand {
                        gate_feature_post!(&self, field_init_shorthand, field.span,
                                           "struct field shorthands are unstable");
                    }
                    if starts_with_digit(&field.ident.node.name.as_str()) {
                        gate_feature_post!(&self, relaxed_adts,
                                          field.span,
                                          "numeric fields in struct expressions are unstable");
                    }
                }
            }
            ast::ExprKind::Break(_, Some(_)) => {
                gate_feature_post!(&self, loop_break_value, e.span,
                                   "`break` with a value is experimental");
            }
            ast::ExprKind::Lit(ref lit) => {
                if let ast::LitKind::Int(_, ref ty) = lit.node {
                    match *ty {
                        ast::LitIntType::Signed(ast::IntTy::I128) |
                        ast::LitIntType::Unsigned(ast::UintTy::U128) => {
                            gate_feature_post!(&self, i128_type, e.span,
                                               "128-bit integers are not stable");
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        visit::walk_expr(self, e);
    }

    fn visit_pat(&mut self, pattern: &'a ast::Pat) {
        match pattern.node {
            PatKind::Slice(_, Some(_), ref last) if !last.is_empty() => {
                gate_feature_post!(&self, advanced_slice_patterns,
                                  pattern.span,
                                  "multiple-element slice matches anywhere \
                                   but at the end of a slice (e.g. \
                                   `[0, ..xs, 0]`) are experimental")
            }
            PatKind::Slice(..) => {
                gate_feature_post!(&self, slice_patterns,
                                  pattern.span,
                                  "slice pattern syntax is experimental");
            }
            PatKind::Box(..) => {
                gate_feature_post!(&self, box_patterns,
                                  pattern.span,
                                  "box pattern syntax is experimental");
            }
            PatKind::Struct(_, ref fields, _) => {
                for field in fields {
                    if starts_with_digit(&field.node.ident.name.as_str()) {
                        gate_feature_post!(&self, relaxed_adts,
                                          field.span,
                                          "numeric fields in struct patterns are unstable");
                    }
                }
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
        // check for const fn declarations
        match fn_kind {
            FnKind::ItemFn(_, _, _, Spanned { node: ast::Constness::Const, .. }, _, _, _) => {
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
            FnKind::ItemFn(_, _, _, _, abi, _, _) |
            FnKind::Method(_, &ast::MethodSig { abi, .. }, _, _) => {
                self.check_abi(abi, span);
            }
            _ => {}
        }
        visit::walk_fn(self, fn_kind, fn_decl, span);
    }

    fn visit_trait_item(&mut self, ti: &'a ast::TraitItem) {
        match ti.node {
            ast::TraitItemKind::Const(..) => {
                gate_feature_post!(&self, associated_consts,
                                  ti.span,
                                  "associated constants are experimental")
            }
            ast::TraitItemKind::Method(ref sig, ref block) => {
                if block.is_none() {
                    self.check_abi(sig.abi, ti.span);
                }
                if sig.constness.node == ast::Constness::Const {
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

    fn visit_impl_item(&mut self, ii: &'a ast::ImplItem) {
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
                if sig.constness.node == ast::Constness::Const {
                    gate_feature_post!(&self, const_fn, ii.span, "const fn is unstable");
                }
            }
            _ => {}
        }
        visit::walk_impl_item(self, ii);
    }

    fn visit_vis(&mut self, vis: &'a ast::Visibility) {
        let span = match *vis {
            ast::Visibility::Crate(span) => span,
            ast::Visibility::Restricted { ref path, .. } => path.span,
            _ => return,
        };
        gate_feature_post!(&self, pub_restricted, span, "`pub(restricted)` syntax is experimental");

        visit::walk_vis(self, vis)
    }

    fn visit_generics(&mut self, g: &'a ast::Generics) {
        for t in &g.ty_params {
            if !t.attrs.is_empty() {
                gate_feature_post!(&self, generic_param_attrs, t.attrs[0].span,
                                   "attributes on type parameter bindings are experimental");
            }
        }
        visit::walk_generics(self, g)
    }

    fn visit_lifetime_def(&mut self, lifetime_def: &'a ast::LifetimeDef) {
        if !lifetime_def.attrs.is_empty() {
            gate_feature_post!(&self, generic_param_attrs, lifetime_def.attrs[0].span,
                               "attributes on lifetime bindings are experimental");
        }
        visit::walk_lifetime_def(self, lifetime_def)
    }
}

pub fn get_features(span_handler: &Handler, krate_attrs: &[ast::Attribute]) -> Features {
    let mut features = Features::new();

    let mut feature_checker = MutexFeatureChecker::default();

    for attr in krate_attrs {
        if !attr.check_name("feature") {
            continue
        }

        match attr.meta_item_list() {
            None => {
                span_err!(span_handler, attr.span, E0555,
                          "malformed feature attribute, expected #![feature(...)]");
            }
            Some(list) => {
                for mi in list {
                    let name = if let Some(word) = mi.word() {
                        word.name()
                    } else {
                        span_err!(span_handler, mi.span, E0556,
                                  "malformed feature, expected just one word");
                        continue
                    };

                    if let Some(&(_, _, _, setter)) = ACTIVE_FEATURES.iter()
                        .find(|& &(n, _, _, _)| name == n) {
                        *(setter(&mut features)) = true;
                        feature_checker.collect(&features, mi.span);
                    }
                    else if let Some(&(_, _, _)) = REMOVED_FEATURES.iter()
                        .find(|& &(n, _, _)| name == n) {
                        span_err!(span_handler, mi.span, E0557, "feature has been removed");
                    }
                    else if let Some(&(_, _, _)) = ACCEPTED_FEATURES.iter()
                        .find(|& &(n, _, _)| name == n) {
                        features.declared_stable_lang_features.push((name, mi.span));
                    } else {
                        features.declared_lib_features.push((name, mi.span));
                    }
                }
            }
        }
    }

    feature_checker.check(span_handler);

    features
}

// A collector for mutually-exclusive features and their flag spans
#[derive(Default)]
struct MutexFeatureChecker {
    proc_macro: Option<Span>,
    custom_attribute: Option<Span>,
}

impl MutexFeatureChecker {
    // If this method turns out to be a hotspot due to branching,
    // the branching can be eliminated by modifying `setter!()` to set these spans
    // only for the features that need to be checked for mutual exclusion.
    fn collect(&mut self, features: &Features, span: Span) {
        if features.proc_macro {
            // If self.proc_macro is None, set to Some(span)
            self.proc_macro = self.proc_macro.or(Some(span));
        }

        if features.custom_attribute {
            self.custom_attribute = self.custom_attribute.or(Some(span));
        }
    }

    fn check(self, handler: &Handler) {
        if let (Some(pm_span), Some(ca_span)) = (self.proc_macro, self.custom_attribute) {
            handler.struct_span_err(pm_span, "Cannot use `#![feature(proc_macro)]` and \
                                              `#![feature(custom_attribute)] at the same time")
                .span_note(ca_span, "`#![feature(custom_attribute)]` declared here")
                .emit();

            panic!(FatalError);
        }
    }
}

pub fn check_crate(krate: &ast::Crate,
                   sess: &ParseSess,
                   features: &Features,
                   plugin_attributes: &[(String, AttributeType)],
                   unstable: UnstableFeatures) {
    maybe_stage_features(&sess.span_diagnostic, krate, unstable);
    let ctx = Context {
        features: features,
        parse_sess: sess,
        cm: sess.codemap(),
        plugin_attributes: plugin_attributes,
    };
    visit::walk_crate(&mut PostExpansionVisitor { context: &ctx }, krate);
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
                          "#[feature] may not be used on the {} release channel",
                          release_channel);
            }
        }
    }
}
