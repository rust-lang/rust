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
use ast::{self, NodeId, PatKind, RangeEnd, RangeSyntax};
use attr;
use codemap::Spanned;
use syntax_pos::Span;
use errors::{DiagnosticBuilder, Handler, FatalError};
use visit::{self, FnKind, Visitor};
use parse::ParseSess;
use symbol::{keywords, Symbol};

use std::{env, path};

macro_rules! set {
    (proc_macro) => {{
        fn f(features: &mut Features, span: Span) {
            features.declared_lib_features.push((Symbol::intern("proc_macro"), span));
            features.proc_macro = true;
        }
        f as fn(&mut Features, Span)
    }};
    ($field: ident) => {{
        fn f(features: &mut Features, _: Span) {
            features.$field = true;
        }
        f as fn(&mut Features, Span)
    }}
}

macro_rules! declare_features {
    ($((active, $feature: ident, $ver: expr, $issue: expr),)+) => {
        /// Represents active features that are currently being implemented or
        /// currently being considered for addition/removal.
        const ACTIVE_FEATURES:
                &'static [(&'static str, &'static str, Option<u32>, fn(&mut Features, Span))] =
            &[$((stringify!($feature), $ver, $issue, set!($feature))),+];

        /// A set of features to be used by later passes.
        pub struct Features {
            /// `#![feature]` attrs for stable language features, for error reporting
            pub declared_stable_lang_features: Vec<(Symbol, Span)>,
            /// `#![feature]` attrs for non-language (library) features
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
        /// Represents unstable features which have since been removed (it was once Active)
        const REMOVED_FEATURES: &'static [(&'static str, &'static str, Option<u32>)] = &[
            $((stringify!($feature), $ver, $issue)),+
        ];
    };

    ($((stable_removed, $feature: ident, $ver: expr, $issue: expr),)+) => {
        /// Represents stable features which have since been removed (it was once Accepted)
        const STABLE_REMOVED_FEATURES: &'static [(&'static str, &'static str, Option<u32>)] = &[
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

// If you change this, please modify src/doc/unstable-book as well.
//
// Don't ever remove anything from this list; set them to 'Removed'.
//
// The version numbers here correspond to the version in which the current status
// was set. This is most important for knowing when a particular feature became
// stable (active).
//
// NB: tools/tidy/src/features.rs parses this information directly out of the
// source, so take care when modifying it.

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


    // rustc internal
    (active, rustc_diagnostic_macros, "1.0.0", None),
    (active, rustc_const_unstable, "1.0.0", None),
    (active, advanced_slice_patterns, "1.0.0", Some(23121)),
    (active, box_syntax, "1.0.0", Some(27779)),
    (active, placement_in_syntax, "1.0.0", Some(27779)),
    (active, unboxed_closures, "1.0.0", Some(29625)),

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

    // macro re-export needs more discussion and stabilization
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

    // Allows the use of non lexical lifetimes; RFC 2094
    (active, nll, "1.0.0", Some(43234)),

    // Allows the use of #[allow_internal_unstable]. This is an
    // attribute on macro_rules! and can't use the attribute handling
    // below (it has to be checked before expansion possibly makes
    // macros disappear).
    //
    // rustc internal
    (active, allow_internal_unstable, "1.0.0", None),

    // Allows the use of #[allow_internal_unsafe]. This is an
    // attribute on macro_rules! and can't use the attribute handling
    // below (it has to be checked before expansion possibly makes
    // macros disappear).
    //
    // rustc internal
    (active, allow_internal_unsafe, "1.0.0", None),

    // #23121. Array patterns have some hazards yet.
    (active, slice_patterns, "1.0.0", Some(23121)),

    // Allows the definition of `const fn` functions.
    (active, const_fn, "1.2.0", Some(24111)),

    // Allows indexing into constant arrays.
    (active, const_indexing, "1.4.0", Some(29947)),

    // Allows using #[prelude_import] on glob `use` items.
    //
    // rustc internal
    (active, prelude_import, "1.2.0", None),

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

    // a..=b and ..=b
    (active, inclusive_range_syntax, "1.7.0", Some(28237)),

    // X..Y patterns
    (active, exclusive_range_pattern, "1.11.0", Some(37854)),

    // impl specialization (RFC 1210)
    (active, specialization, "1.7.0", Some(31844)),

    // Allows cfg(target_has_atomic = "...").
    (active, cfg_target_has_atomic, "1.9.0", Some(32976)),

    // Allows `impl Trait` in function return types.
    (active, conservative_impl_trait, "1.12.0", Some(34511)),

    // Allows `impl Trait` in function arguments.
    (active, universal_impl_trait, "1.23.0", Some(34511)),

    // The `!` type
    (active, never_type, "1.13.0", Some(35121)),

    // Allows all literals in attribute lists and values of key-value pairs.
    (active, attr_literals, "1.13.0", Some(34981)),

    // Allows untagged unions `union U { ... }`
    (active, untagged_unions, "1.13.0", Some(32836)),

    // Used to identify the `compiler_builtins` crate
    // rustc internal
    (active, compiler_builtins, "1.13.0", None),

    // Allows attributes on lifetime/type formal parameters in generics (RFC 1327)
    (active, generic_param_attrs, "1.11.0", Some(34761)),

    // Allows #[link(..., cfg(..))]
    (active, link_cfg, "1.14.0", Some(37406)),

    (active, use_extern_macros, "1.15.0", Some(35896)),

    // Allows #[target_feature(...)]
    (active, target_feature, "1.15.0", None),

    // `extern "ptx-*" fn()`
    (active, abi_ptx, "1.15.0", None),

    // The `i128` type
    (active, i128_type, "1.16.0", Some(35118)),

    // The `repr(i128)` annotation for enums
    (active, repr128, "1.16.0", Some(35118)),

    // The `unadjusted` ABI. Perma unstable.
    (active, abi_unadjusted, "1.16.0", None),

    // Procedural macros 2.0.
    (active, proc_macro, "1.16.0", Some(38356)),

    // Declarative macros 2.0 (`macro`).
    (active, decl_macro, "1.17.0", Some(39412)),

    // Allows #[link(kind="static-nobundle"...]
    (active, static_nobundle, "1.16.0", Some(37403)),

    // `extern "msp430-interrupt" fn()`
    (active, abi_msp430_interrupt, "1.16.0", Some(38487)),

    // Used to identify crates that contain sanitizer runtimes
    // rustc internal
    (active, sanitizer_runtime, "1.17.0", None),

    // Used to identify crates that contain the profiler runtime
    // rustc internal
    (active, profiler_runtime, "1.18.0", None),

    // `extern "x86-interrupt" fn()`
    (active, abi_x86_interrupt, "1.17.0", Some(40180)),


    // Allows the `catch {...}` expression
    (active, catch_expr, "1.17.0", Some(31436)),

    // Used to preserve symbols (see llvm.used)
    (active, used, "1.18.0", Some(40289)),

    // Allows module-level inline assembly by way of global_asm!()
    (active, global_asm, "1.18.0", Some(35119)),

    // Allows overlapping impls of marker traits
    (active, overlapping_marker_traits, "1.18.0", Some(29864)),

    // Allows use of the :vis macro fragment specifier
    (active, macro_vis_matcher, "1.18.0", Some(41022)),

    // rustc internal
    (active, abi_thiscall, "1.19.0", None),

    // Allows a test to fail without failing the whole suite
    (active, allow_fail, "1.19.0", Some(42219)),

    // Allows unsized tuple coercion.
    (active, unsized_tuple_coercion, "1.20.0", Some(42877)),

    // Generators
    (active, generators, "1.21.0", None),

    // Trait aliases
    (active, trait_alias, "1.24.0", Some(41517)),

    // global allocators and their internals
    (active, global_allocator, "1.20.0", None),
    (active, allocator_internals, "1.20.0", None),

    // #[doc(cfg(...))]
    (active, doc_cfg, "1.21.0", Some(43781)),
    // #[doc(masked)]
    (active, doc_masked, "1.21.0", Some(44027)),
    // #[doc(spotlight)]
    (active, doc_spotlight, "1.22.0", Some(45040)),
    // #[doc(include="some-file")]
    (active, external_doc, "1.22.0", Some(44732)),

    // allow `#[must_use]` on functions and comparison operators (RFC 1940)
    (active, fn_must_use, "1.21.0", Some(43302)),

    // Future-proofing enums/structs with #[non_exhaustive] attribute (RFC 2008)
    (active, non_exhaustive, "1.22.0", Some(44109)),

    // Copy/Clone closures (RFC 2132)
    (active, clone_closures, "1.22.0", Some(44490)),
    (active, copy_closures, "1.22.0", Some(44490)),

    // allow `'_` placeholder lifetimes
    (active, underscore_lifetimes, "1.22.0", Some(44524)),

    // allow `..=` in patterns (RFC 1192)
    (active, dotdoteq_in_patterns, "1.22.0", Some(28237)),

    // Default match binding modes (RFC 2005)
    (active, match_default_bindings, "1.22.0", Some(42640)),

    // Trait object syntax with `dyn` prefix
    (active, dyn_trait, "1.22.0", Some(44662)),

    // `crate` as visibility modifier, synonymous to `pub(crate)`
    (active, crate_visibility_modifier, "1.23.0", Some(45388)),

    // extern types
    (active, extern_types, "1.23.0", Some(43467)),

    // Allow trait methods with arbitrary self types
    (active, arbitrary_self_types, "1.23.0", Some(44874)),

    // #![wasm_import_memory] attribute
    (active, wasm_import_memory, "1.22.0", None),

    // `crate` in paths
    (active, crate_in_paths, "1.23.0", Some(45477)),

    // In-band lifetime bindings (e.g. `fn foo(x: &'a u8) -> &'a u8`)
    (active, in_band_lifetimes, "1.23.0", Some(44524)),

    // generic associated types (RFC 1598)
    (active, generic_associated_types, "1.23.0", Some(44265)),

    // Resolve absolute paths as paths from other crates
    (active, extern_absolute_paths, "1.24.0", Some(44660)),

    // `foo.rs` as an alternative to `foo/mod.rs`
    (active, non_modrs_mods, "1.24.0", Some(44660)),

    // Nested `impl Trait`
    (active, nested_impl_trait, "1.24.0", Some(34511)),

    // Termination trait in main (RFC 1937)
    (active, termination_trait, "1.24.0", Some(43301)),

    // Allows use of the :lifetime macro fragment specifier
    (active, macro_lifetime_matcher, "1.24.0", Some(46895)),

    // `extern` in paths
    (active, extern_in_paths, "1.23.0", Some(44660)),

    // Allows `#[repr(transparent)]` attribute on newtype structs
    (active, repr_transparent, "1.25.0", Some(43036)),

    // Use `?` as the Kleene "at most one" operator
    (active, macro_at_most_once_rep, "1.25.0", Some(48075)),
);

declare_features! (
    (removed, import_shadowing, "1.0.0", None),
    (removed, managed_boxes, "1.0.0", None),
    // Allows use of unary negate on unsigned integers, e.g. -e for e: u8
    (removed, negate_unsigned, "1.0.0", Some(29645)),
    (removed, reflect, "1.0.0", Some(27749)),
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
    (removed, allocator, "1.0.0", None),
    // Allows the `#[simd]` attribute -- removed in favor of `#[repr(simd)]`
    (removed, simd, "1.0.0", Some(27731)),
);

declare_features! (
    (stable_removed, no_stack_check, "1.0.0", None),
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
    (accepted, item_like_imports, "1.15.0", Some(35120)),
    // Allows using `Self` and associated types in struct expressions and patterns.
    (accepted, more_struct_aliases, "1.16.0", Some(37544)),
    // elide `'static` lifetimes in `static`s and `const`s
    (accepted, static_in_const, "1.17.0", Some(35897)),
    // Allows field shorthands (`x` meaning `x: x`) in struct literal expressions.
    (accepted, field_init_shorthand, "1.17.0", Some(37340)),
    // Allows the definition recursive static items.
    (accepted, static_recursion, "1.17.0", Some(29719)),
    // pub(restricted) visibilities (RFC 1422)
    (accepted, pub_restricted, "1.18.0", Some(32409)),
    // The #![windows_subsystem] attribute
    (accepted, windows_subsystem, "1.18.0", Some(37499)),
    // Allows `break {expr}` with a value inside `loop`s.
    (accepted, loop_break_value, "1.19.0", Some(37339)),
    // Permits numeric fields in struct expressions and patterns.
    (accepted, relaxed_adts, "1.19.0", Some(35626)),
    // Coerces non capturing closures to function pointers
    (accepted, closure_to_fn_coercion, "1.19.0", Some(39817)),
    // Allows attributes on struct literal fields.
    (accepted, struct_field_attributes, "1.20.0", Some(38814)),
    // Allows the definition of associated constants in `trait` or `impl`
    // blocks.
    (accepted, associated_consts, "1.20.0", Some(29646)),
    // Usage of the `compile_error!` macro
    (accepted, compile_error, "1.20.0", Some(40872)),
    // See rust-lang/rfcs#1414. Allows code like `let x: &'static u32 = &42` to work.
    (accepted, rvalue_static_promotion, "1.21.0", Some(38865)),
    // Allow Drop types in constants (RFC 1440)
    (accepted, drop_types_in_const, "1.22.0", Some(33156)),
    // Allows the sysV64 ABI to be specified on all platforms
    // instead of just the platforms on which it is the C ABI
    (accepted, abi_sysv64, "1.24.0", Some(36167)),
    // Allows `repr(align(16))` struct attribute (RFC 1358)
    (accepted, repr_align, "1.25.0", Some(33626)),
    // allow '|' at beginning of match arms (RFC 1925)
    (accepted, match_beginning_vert, "1.25.0", Some(44101)),
    // Nested groups in `use` (RFC 2128)
    (accepted, use_nested_groups, "1.25.0", Some(44494)),
);

// If you change this, please modify src/doc/unstable-book as well. You must
// move that documentation into the relevant place in the other docs, and
// remove the chapter on the flag.

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
    ("link_args", Normal, Gated(Stability::Unstable,
                                "link_args",
                                "the `link_args` attribute is experimental and not \
                                 portable across platforms, it is recommended to \
                                 use `#[link(name = \"foo\")] instead",
                                cfg_fn!(link_args))),
    ("macro_escape", Normal, Ungated),

    // RFC #1445.
    ("structural_match", Whitelisted, Gated(Stability::Unstable,
                                            "structural_match",
                                            "the semantics of constant patterns is \
                                             not yet settled",
                                            cfg_fn!(structural_match))),

    // RFC #2008
    ("non_exhaustive", Whitelisted, Gated(Stability::Unstable,
                                          "non_exhaustive",
                                          "non exhaustive is an experimental feature",
                                          cfg_fn!(non_exhaustive))),

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
                                         not currently handle destructors.",
                                        cfg_fn!(thread_local))),

    ("rustc_on_unimplemented", Normal, Gated(Stability::Unstable,
                                             "on_unimplemented",
                                             "the `#[rustc_on_unimplemented]` attribute \
                                              is an experimental feature",
                                             cfg_fn!(on_unimplemented))),
    ("rustc_const_unstable", Normal, Gated(Stability::Unstable,
                                             "rustc_const_unstable",
                                             "the `#[rustc_const_unstable]` attribute \
                                              is an internal feature",
                                             cfg_fn!(rustc_const_unstable))),
    ("global_allocator", Normal, Gated(Stability::Unstable,
                                       "global_allocator",
                                       "the `#[global_allocator]` attribute is \
                                        an experimental feature",
                                       cfg_fn!(global_allocator))),
    ("default_lib_allocator", Whitelisted, Gated(Stability::Unstable,
                                            "allocator_internals",
                                            "the `#[default_lib_allocator]` \
                                             attribute is an experimental feature",
                                            cfg_fn!(allocator_internals))),
    ("needs_allocator", Normal, Gated(Stability::Unstable,
                                      "allocator_internals",
                                      "the `#[needs_allocator]` \
                                       attribute is an experimental \
                                       feature",
                                      cfg_fn!(allocator_internals))),
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
    ("rustc_regions", Normal, Gated(Stability::Unstable,
                                    "rustc_attrs",
                                    "the `#[rustc_regions]` attribute \
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
    ("rustc_serialize_exclude_null", Normal, Gated(Stability::Unstable,
                                             "rustc_attrs",
                                             "the `#[rustc_serialize_exclude_null]` attribute \
                                              is an internal-only feature",
                                             cfg_fn!(rustc_attrs))),
    ("rustc_synthetic", Whitelisted, Gated(Stability::Unstable,
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

    // RFC #2094
    ("nll", Whitelisted, Gated(Stability::Unstable,
                               "nll",
                               "Non lexical lifetimes",
                               cfg_fn!(nll))),
    ("compiler_builtins", Whitelisted, Gated(Stability::Unstable,
                                             "compiler_builtins",
                                             "the `#[compiler_builtins]` attribute is used to \
                                              identify the `compiler_builtins` crate which \
                                              contains compiler-rt intrinsics and will never be \
                                              stable",
                                          cfg_fn!(compiler_builtins))),
    ("sanitizer_runtime", Whitelisted, Gated(Stability::Unstable,
                                             "sanitizer_runtime",
                                             "the `#[sanitizer_runtime]` attribute is used to \
                                              identify crates that contain the runtime of a \
                                              sanitizer and will never be stable",
                                             cfg_fn!(sanitizer_runtime))),
    ("profiler_runtime", Whitelisted, Gated(Stability::Unstable,
                                             "profiler_runtime",
                                             "the `#[profiler_runtime]` attribute is used to \
                                              identify the `profiler_builtins` crate which \
                                              contains the profiler runtime and will never be \
                                              stable",
                                             cfg_fn!(profiler_runtime))),

    ("allow_internal_unstable", Normal, Gated(Stability::Unstable,
                                              "allow_internal_unstable",
                                              EXPLAIN_ALLOW_INTERNAL_UNSTABLE,
                                              cfg_fn!(allow_internal_unstable))),

    ("allow_internal_unsafe", Normal, Gated(Stability::Unstable,
                                            "allow_internal_unsafe",
                                            EXPLAIN_ALLOW_INTERNAL_UNSAFE,
                                            cfg_fn!(allow_internal_unsafe))),

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
        "the `#[no_debug]` attribute was an experimental feature that has been \
         deprecated due to lack of demand",
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
    ("used", Whitelisted, Gated(
        Stability::Unstable, "used",
        "the `#[used]` attribute is an experimental feature",
        cfg_fn!(used))),

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

    ("windows_subsystem", Whitelisted, Ungated),

    ("proc_macro_attribute", Normal, Gated(Stability::Unstable,
                                           "proc_macro",
                                           "attribute proc macros are currently unstable",
                                           cfg_fn!(proc_macro))),

    ("proc_macro", Normal, Gated(Stability::Unstable,
                                 "proc_macro",
                                 "function-like proc macros are currently unstable",
                                 cfg_fn!(proc_macro))),

    ("rustc_derive_registrar", Normal, Gated(Stability::Unstable,
                                             "rustc_derive_registrar",
                                             "used internally by rustc",
                                             cfg_fn!(rustc_attrs))),

    ("allow_fail", Normal, Gated(Stability::Unstable,
                                 "allow_fail",
                                 "allow_fail attribute is currently unstable",
                                 cfg_fn!(allow_fail))),

    ("rustc_std_internal_symbol", Whitelisted, Gated(Stability::Unstable,
                                     "rustc_attrs",
                                     "this is an internal attribute that will \
                                      never be stable",
                                     cfg_fn!(rustc_attrs))),

    // whitelists "identity-like" conversion methods to suggest on type mismatch
    ("rustc_conversion_suggestion", Whitelisted, Gated(Stability::Unstable,
                                                       "rustc_attrs",
                                                       "this is an internal attribute that will \
                                                        never be stable",
                                                       cfg_fn!(rustc_attrs))),

    ("wasm_import_memory", Whitelisted, Gated(Stability::Unstable,
                                 "wasm_import_memory",
                                 "wasm_import_memory attribute is currently unstable",
                                 cfg_fn!(wasm_import_memory))),

    ("rustc_args_required_const", Whitelisted, Gated(Stability::Unstable,
                                 "rustc_attrs",
                                 "never will be stable",
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
const GATED_CFGS: &[(&str, &str, fn(&Features) -> bool)] = &[
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
        let name = cfg.name().as_str();
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
        if !has_feature(features) && !self.span.allows_unstable() {
            let explain = format!("`cfg({})` is experimental and subject to change", cfg);
            emit_feature_err(sess, feature, self.span, GateIssue::Language, &explain);
        }
    }
}

struct Context<'a> {
    features: &'a Features,
    parse_sess: &'a ParseSess,
    plugin_attributes: &'a [(String, AttributeType)],
}

macro_rules! gate_feature_fn {
    ($cx: expr, $has_feature: expr, $span: expr, $name: expr, $explain: expr, $level: expr) => {{
        let (cx, has_feature, span,
             name, explain, level) = ($cx, $has_feature, $span, $name, $explain, $level);
        let has_feature: bool = has_feature(&$cx.features);
        debug!("gate_feature(feature = {:?}, span = {:?}); has? {}", name, span, has_feature);
        if !has_feature && !span.allows_unstable() {
            leveled_feature_err(cx.parse_sess, name, span, GateIssue::Language, explain, level)
                .emit();
        }
    }}
}

macro_rules! gate_feature {
    ($cx: expr, $feature: ident, $span: expr, $explain: expr) => {
        gate_feature_fn!($cx, |x:&Features| x.$feature, $span,
                         stringify!($feature), $explain, GateStrength::Hard)
    };
    ($cx: expr, $feature: ident, $span: expr, $explain: expr, $level: expr) => {
        gate_feature_fn!($cx, |x:&Features| x.$feature, $span,
                         stringify!($feature), $explain, $level)
    };
}

impl<'a> Context<'a> {
    fn check_attribute(&self, attr: &ast::Attribute, is_macro: bool) {
        debug!("check_attribute(attr = {:?})", attr);
        let name = unwrap_or!(attr.name(), return).as_str();
        for &(n, ty, ref gateage) in BUILTIN_ATTRIBUTES {
            if name == n {
                if let Gated(_, name, desc, ref has_feature) = *gateage {
                    gate_feature_fn!(self, has_feature, attr.span, name, desc, GateStrength::Hard);
                } else if name == "doc" {
                    if let Some(content) = attr.meta_item_list() {
                        if content.iter().any(|c| c.check_name("include")) {
                            gate_feature!(self, external_doc, attr.span,
                                "#[doc(include = \"...\")] is experimental"
                            );
                        }
                    }
                }
                debug!("check_attribute: {:?} is builtin, {:?}, {:?}", attr.path, ty, gateage);
                return;
            }
        }
        for &(ref n, ref ty) in self.plugin_attributes {
            if attr.path == &**n {
                // Plugins can't gate attributes, so we don't check for it
                // unlike the code above; we only use this loop to
                // short-circuit to avoid the checks below
                debug!("check_attribute: {:?} is registered by a plugin, {:?}", attr.path, ty);
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
        } else if !attr::is_known(attr) {
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
                                       attr.path));
            }
        }
    }
}

pub fn check_attribute(attr: &ast::Attribute, parse_sess: &ParseSess, features: &Features) {
    let cx = Context { features: features, parse_sess: parse_sess, plugin_attributes: &[] };
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
        // search in Accepted, Removed, or Stable Removed features
        let found = ACCEPTED_FEATURES.iter().chain(REMOVED_FEATURES).chain(STABLE_REMOVED_FEATURES)
            .find(|t| t.0 == feature);
        match found {
            Some(&(_, _, issue)) => issue,
            None => panic!("Feature `{}` is not declared anywhere", feature),
        }
    }
}

pub enum GateIssue {
    Language,
    Library(Option<u32>)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GateStrength {
    /// A hard error. (Most feature gates should use this.)
    Hard,
    /// Only a warning. (Use this only as backwards-compatibility demands.)
    Soft,
}

pub fn emit_feature_err(sess: &ParseSess, feature: &str, span: Span, issue: GateIssue,
                        explain: &str) {
    feature_err(sess, feature, span, issue, explain).emit();
}

pub fn feature_err<'a>(sess: &'a ParseSess, feature: &str, span: Span, issue: GateIssue,
                       explain: &str) -> DiagnosticBuilder<'a> {
    leveled_feature_err(sess, feature, span, issue, explain, GateStrength::Hard)
}

fn leveled_feature_err<'a>(sess: &'a ParseSess, feature: &str, span: Span, issue: GateIssue,
                           explain: &str, level: GateStrength) -> DiagnosticBuilder<'a> {
    let diag = &sess.span_diagnostic;

    let issue = match issue {
        GateIssue::Language => find_lang_feature_issue(feature),
        GateIssue::Library(lib) => lib,
    };

    let explanation = if let Some(n) = issue {
        format!("{} (see issue #{})", explain, n)
    } else {
        explain.to_owned()
    };

    let mut err = match level {
        GateStrength::Hard => {
            diag.struct_span_err_with_code(span, &explanation, stringify_error_code!(E0658))
        }
        GateStrength::Soft => diag.struct_span_warn(span, &explanation),
    };

    // #23973: do not suggest `#![feature(...)]` if we are in beta/stable
    if sess.unstable_features.is_nightly_build() {
        err.help(&format!("add #![feature({})] to the \
                           crate attributes to enable",
                          feature));
    }

    // If we're on stable and only emitting a "soft" warning, add a note to
    // clarify that the feature isn't "on" (rather than being on but
    // warning-worthy).
    if !sess.unstable_features.is_nightly_build() && level == GateStrength::Soft {
        err.help("a nightly build of the compiler is required to enable this feature");
    }

    err

}

const EXPLAIN_BOX_SYNTAX: &'static str =
    "box expression syntax is experimental; you can call `Box::new` instead.";

pub const EXPLAIN_STMT_ATTR_SYNTAX: &'static str =
    "attributes on non-item statements and expressions are experimental.";

pub const EXPLAIN_ASM: &'static str =
    "inline assembly is not stable enough for use and is subject to change";

pub const EXPLAIN_GLOBAL_ASM: &'static str =
    "`global_asm!` is not stable enough for use and is subject to change";

pub const EXPLAIN_LOG_SYNTAX: &'static str =
    "`log_syntax!` is not stable enough for use and is subject to change";

pub const EXPLAIN_CONCAT_IDENTS: &'static str =
    "`concat_idents` is not stable enough for use and is subject to change";

pub const EXPLAIN_TRACE_MACROS: &'static str =
    "`trace_macros` is not stable enough for use and is subject to change";
pub const EXPLAIN_ALLOW_INTERNAL_UNSTABLE: &'static str =
    "allow_internal_unstable side-steps feature gating and stability checks";
pub const EXPLAIN_ALLOW_INTERNAL_UNSAFE: &'static str =
    "allow_internal_unsafe side-steps the unsafe_code lint";

pub const EXPLAIN_CUSTOM_DERIVE: &'static str =
    "`#[derive]` for custom traits is deprecated and will be removed in the future.";

pub const EXPLAIN_DEPR_CUSTOM_DERIVE: &'static str =
    "`#[derive]` for custom traits is deprecated and will be removed in the future. \
    Prefer using procedural macro custom derive.";

pub const EXPLAIN_DERIVE_UNDERSCORE: &'static str =
    "attributes of the form `#[derive_*]` are reserved for the compiler";

pub const EXPLAIN_VIS_MATCHER: &'static str =
    ":vis fragment specifier is experimental and subject to change";

pub const EXPLAIN_LIFETIME_MATCHER: &'static str =
    ":lifetime fragment specifier is experimental and subject to change";

pub const EXPLAIN_PLACEMENT_IN: &'static str =
    "placement-in expression syntax is experimental and subject to change.";

pub const EXPLAIN_UNSIZED_TUPLE_COERCION: &'static str =
    "Unsized tuple coercion is not stable enough for use and is subject to change";

pub const EXPLAIN_MACRO_AT_MOST_ONCE_REP: &'static str =
    "Using the `?` macro Kleene operator for \"at most one\" repetition is unstable";

struct PostExpansionVisitor<'a> {
    context: &'a Context<'a>,
}

macro_rules! gate_feature_post {
    ($cx: expr, $feature: ident, $span: expr, $explain: expr) => {{
        let (cx, span) = ($cx, $span);
        if !span.allows_unstable() {
            gate_feature!(cx.context, $feature, span, $explain)
        }
    }};
    ($cx: expr, $feature: ident, $span: expr, $explain: expr, $level: expr) => {{
        let (cx, span) = ($cx, $span);
        if !span.allows_unstable() {
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
}

fn contains_novel_literal(item: &ast::MetaItem) -> bool {
    use ast::MetaItemKind::*;
    use ast::NestedMetaItemKind::*;

    match item.node {
        Word => false,
        NameValue(ref lit) => !lit.node.is_str(),
        List(ref list) => list.iter().any(|li| {
            match li.node {
                MetaItem(ref mi) => contains_novel_literal(mi),
                Literal(_) => true,
            }
        }),
    }
}

// Bans nested `impl Trait`, e.g. `impl Into<impl Debug>`.
// Nested `impl Trait` _is_ allowed in associated type position,
// e.g `impl Iterator<Item=impl Debug>`
struct NestedImplTraitVisitor<'a> {
    context: &'a Context<'a>,
    is_in_impl_trait: bool,
}

impl<'a> NestedImplTraitVisitor<'a> {
    fn with_impl_trait<F>(&mut self, is_in_impl_trait: bool, f: F)
        where F: FnOnce(&mut NestedImplTraitVisitor<'a>)
    {
        let old_is_in_impl_trait = self.is_in_impl_trait;
        self.is_in_impl_trait = is_in_impl_trait;
        f(self);
        self.is_in_impl_trait = old_is_in_impl_trait;
    }
}


impl<'a> Visitor<'a> for NestedImplTraitVisitor<'a> {
    fn visit_ty(&mut self, t: &'a ast::Ty) {
        if let ast::TyKind::ImplTrait(_) = t.node {
            if self.is_in_impl_trait {
                gate_feature_post!(&self, nested_impl_trait, t.span,
                    "nested `impl Trait` is experimental"
                );
            }
            self.with_impl_trait(true, |this| visit::walk_ty(this, t));
        } else {
            visit::walk_ty(self, t);
        }
    }
    fn visit_path_parameters(&mut self, _: Span, path_parameters: &'a ast::PathParameters) {
        match *path_parameters {
            ast::PathParameters::AngleBracketed(ref params) => {
                for type_ in &params.types {
                    self.visit_ty(type_);
                }
                for type_binding in &params.bindings {
                    // Type bindings such as `Item=impl Debug` in `Iterator<Item=Debug>`
                    // are allowed to contain nested `impl Trait`.
                    self.with_impl_trait(false, |this| visit::walk_ty(this, &type_binding.ty));
                }
            }
            ast::PathParameters::Parenthesized(ref params) => {
                for type_ in &params.inputs {
                    self.visit_ty(type_);
                }
                if let Some(ref type_) = params.output {
                    // `-> Foo` syntax is essentially an associated type binding,
                    // so it is also allowed to contain nested `impl Trait`.
                    self.with_impl_trait(false, |this| visit::walk_ty(this, type_));
                }
            }
        }
    }
}

impl<'a> PostExpansionVisitor<'a> {
    fn whole_crate_feature_gates(&mut self, krate: &ast::Crate) {
        visit::walk_crate(
            &mut NestedImplTraitVisitor {
                context: self.context,
                is_in_impl_trait: false,
            }, krate);

        for &(ident, span) in &*self.context.parse_sess.non_modrs_mods.borrow() {
            if !span.allows_unstable() {
                let cx = &self.context;
                let level = GateStrength::Hard;
                let has_feature = cx.features.non_modrs_mods;
                let name = "non_modrs_mods";
                debug!("gate_feature(feature = {:?}, span = {:?}); has? {}",
                        name, span, has_feature);

                if !has_feature && !span.allows_unstable() {
                    leveled_feature_err(
                        cx.parse_sess, name, span, GateIssue::Language,
                        "mod statements in non-mod.rs files are unstable", level
                    )
                    .help(&format!("on stable builds, rename this file to {}{}mod.rs",
                                   ident, path::MAIN_SEPARATOR))
                    .emit();
                }
            }
        }
    }
}

impl<'a> Visitor<'a> for PostExpansionVisitor<'a> {
    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        if !attr.span.allows_unstable() {
            // check for gated attributes
            self.context.check_attribute(attr, false);
        }

        if attr.check_name("doc") {
            if let Some(content) = attr.meta_item_list() {
                if content.len() == 1 && content[0].check_name("cfg") {
                    gate_feature_post!(&self, doc_cfg, attr.span,
                        "#[doc(cfg(...))] is experimental"
                    );
                } else if content.iter().any(|c| c.check_name("masked")) {
                    gate_feature_post!(&self, doc_masked, attr.span,
                        "#[doc(masked)] is experimental"
                    );
                } else if content.iter().any(|c| c.check_name("spotlight")) {
                    gate_feature_post!(&self, doc_spotlight, attr.span,
                        "#[doc(spotlight)] is experimental"
                    );
                }
            }
        }

        // allow attr_literals in #[repr(align(x))]
        let mut is_repr_align = false;
        if attr.path == "repr" {
            if let Some(content) = attr.meta_item_list() {
                is_repr_align = content.iter().any(|c| c.check_name("align"));
            }
        }

        if self.context.features.proc_macro && attr::is_known(attr) {
            return
        }

        if !is_repr_align {
            let meta = panictry!(attr.parse_meta(self.context.parse_sess));
            if contains_novel_literal(&meta) {
                gate_feature_post!(&self, attr_literals, attr.span,
                                   "non-string literals in attributes, or string \
                                   literals in top-level positions, are experimental");
            }
        }
    }

    fn visit_name(&mut self, sp: Span, name: ast::Name) {
        if !name.as_str().is_ascii() {
            gate_feature_post!(&self,
                               non_ascii_idents,
                               self.context.parse_sess.codemap().def_span(sp),
                               "non-ascii idents are not fully supported.");
        }
    }

    fn visit_item(&mut self, i: &'a ast::Item) {
        match i.node {
            ast::ItemKind::ExternCrate(_) => {
                if let Some(attr) = attr::find_by_name(&i.attrs[..], "macro_reexport") {
                    gate_feature_post!(&self, macro_reexport, attr.span,
                                       "macros re-exports are experimental \
                                        and possibly buggy");
                }
            }

            ast::ItemKind::ForeignMod(ref foreign_module) => {
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
                if let Some(attr) = attr::find_by_name(&i.attrs[..], "must_use") {
                    gate_feature_post!(&self, fn_must_use, attr.span,
                                       "`#[must_use]` on functions is experimental",
                                       GateStrength::Soft);
                }
            }

            ast::ItemKind::Struct(..) => {
                if let Some(attr) = attr::find_by_name(&i.attrs[..], "repr") {
                    for item in attr.meta_item_list().unwrap_or_else(Vec::new) {
                        if item.check_name("simd") {
                            gate_feature_post!(&self, repr_simd, attr.span,
                                               "SIMD types are experimental and possibly buggy");
                        }
                        if item.check_name("transparent") {
                            gate_feature_post!(&self, repr_transparent, attr.span,
                                               "the `#[repr(transparent)]` attribute \
                                               is experimental");
                        }
                    }
                }
            }

            ast::ItemKind::TraitAlias(..) => {
                gate_feature_post!(&self, trait_alias,
                                   i.span,
                                   "trait aliases are not yet fully implemented");
            }

            ast::ItemKind::Impl(_, polarity, defaultness, _, _, _, ref impl_items) => {
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

                for impl_item in impl_items {
                    if let ast::ImplItemKind::Method(..) = impl_item.node {
                        if let Some(attr) = attr::find_by_name(&impl_item.attrs[..], "must_use") {
                            gate_feature_post!(&self, fn_must_use, attr.span,
                                               "`#[must_use]` on methods is experimental",
                                               GateStrength::Soft);
                        }
                    }
                }
            }

            ast::ItemKind::Trait(ast::IsAuto::Yes, ..) => {
                gate_feature_post!(&self, optin_builtin_traits,
                                   i.span,
                                   "auto traits are experimental and possibly buggy");
            }

            ast::ItemKind::MacroDef(ast::MacroDef { legacy: false, .. }) => {
                let msg = "`macro` is experimental";
                gate_feature_post!(&self, decl_macro, i.span, msg);
            }

            _ => {}
        }

        visit::walk_item(self, i);
    }

    fn visit_foreign_item(&mut self, i: &'a ast::ForeignItem) {
        match i.node {
            ast::ForeignItemKind::Fn(..) |
            ast::ForeignItemKind::Static(..) => {
                let link_name = attr::first_attr_value_str_by_name(&i.attrs, "link_name");
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
            },
            ast::TyKind::TraitObject(_, ast::TraitObjectSyntax::Dyn) => {
                gate_feature_post!(&self, dyn_trait, ty.span,
                                   "`dyn Trait` syntax is unstable");
            }
            _ => {}
        }
        visit::walk_ty(self, ty)
    }

    fn visit_fn_ret_ty(&mut self, ret_ty: &'a ast::FunctionRetTy) {
        if let ast::FunctionRetTy::Ty(ref output_ty) = *ret_ty {
            if output_ty.node != ast::TyKind::Never {
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
            ast::ExprKind::Yield(..) => {
                gate_feature_post!(&self, generators,
                                  e.span,
                                  "yield syntax is experimental");
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
            ast::ExprKind::Catch(_) => {
                gate_feature_post!(&self, catch_expr, e.span, "`catch` expression is experimental");
            }
            _ => {}
        }
        visit::walk_expr(self, e);
    }

    fn visit_arm(&mut self, arm: &'a ast::Arm) {
        visit::walk_arm(self, arm)
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
            PatKind::Range(_, _, RangeEnd::Excluded) => {
                gate_feature_post!(&self, exclusive_range_pattern, pattern.span,
                                   "exclusive range pattern syntax is experimental");
            }
            PatKind::Range(_, _, RangeEnd::Included(RangeSyntax::DotDotEq)) => {
                gate_feature_post!(&self, dotdoteq_in_patterns, pattern.span,
                                   "`..=` syntax in patterns is experimental");
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
        if let FnKind::ItemFn(_, _, Spanned { node: ast::Constness::Const, .. }, _, _, _) =
            fn_kind {
            gate_feature_post!(&self, const_fn, span, "const fn is unstable");
        }
        // stability of const fn methods are covered in
        // visit_trait_item and visit_impl_item below; this is
        // because default methods don't pass through this
        // point.

        match fn_kind {
            FnKind::ItemFn(_, _, _, abi, _, _) |
            FnKind::Method(_, &ast::MethodSig { abi, .. }, _, _) => {
                self.check_abi(abi, span);
            }
            _ => {}
        }
        visit::walk_fn(self, fn_kind, fn_decl, span);
    }

    fn visit_trait_item(&mut self, ti: &'a ast::TraitItem) {
        match ti.node {
            ast::TraitItemKind::Method(ref sig, ref block) => {
                if block.is_none() {
                    self.check_abi(sig.abi, ti.span);
                }
                if sig.constness.node == ast::Constness::Const {
                    gate_feature_post!(&self, const_fn, ti.span, "const fn is unstable");
                }
            }
            ast::TraitItemKind::Type(_, ref default) => {
                // We use two if statements instead of something like match guards so that both
                // of these errors can be emitted if both cases apply.
                if default.is_some() {
                    gate_feature_post!(&self, associated_type_defaults, ti.span,
                                       "associated type defaults are unstable");
                }
                if ti.generics.is_parameterized() {
                    gate_feature_post!(&self, generic_associated_types, ti.span,
                                       "generic associated types are unstable");
                }
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
            ast::ImplItemKind::Method(ref sig, _) => {
                if sig.constness.node == ast::Constness::Const {
                    gate_feature_post!(&self, const_fn, ii.span, "const fn is unstable");
                }
            }
            ast::ImplItemKind::Type(_) if ii.generics.is_parameterized() => {
                gate_feature_post!(&self, generic_associated_types, ii.span,
                                   "generic associated types are unstable");
            }
            _ => {}
        }
        visit::walk_impl_item(self, ii);
    }

    fn visit_path(&mut self, path: &'a ast::Path, _id: NodeId) {
        for segment in &path.segments {
            if segment.identifier.name == keywords::Crate.name() {
                gate_feature_post!(&self, crate_in_paths, segment.span,
                                   "`crate` in paths is experimental");
            } else if segment.identifier.name == keywords::Extern.name() {
                gate_feature_post!(&self, extern_in_paths, segment.span,
                                   "`extern` in paths is experimental");
            }
        }

        visit::walk_path(self, path);
    }

    fn visit_vis(&mut self, vis: &'a ast::Visibility) {
        if let ast::Visibility::Crate(span, ast::CrateSugar::JustCrate) = *vis {
            gate_feature_post!(&self, crate_visibility_modifier, span,
                               "`crate` visibility modifier is experimental");
        }
        visit::walk_vis(self, vis);
    }

    fn visit_generic_param(&mut self, param: &'a ast::GenericParam) {
        let (attrs, explain) = match *param {
            ast::GenericParam::Lifetime(ref ld) =>
                (&ld.attrs, "attributes on lifetime bindings are experimental"),
            ast::GenericParam::Type(ref t) =>
                (&t.attrs, "attributes on type parameter bindings are experimental"),
        };

        if !attrs.is_empty() {
            gate_feature_post!(&self, generic_param_attrs, attrs[0].span, explain);
        }

        visit::walk_generic_param(self, param)
    }

    fn visit_lifetime(&mut self, lt: &'a ast::Lifetime) {
        if lt.ident.name == "'_" {
            gate_feature_post!(&self, underscore_lifetimes, lt.span,
                               "underscore lifetimes are unstable");
        }
        visit::walk_lifetime(self, lt)
    }
}

pub fn get_features(span_handler: &Handler, krate_attrs: &[ast::Attribute]) -> Features {
    let mut features = Features::new();

    let mut feature_checker = FeatureChecker::default();

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

                    if let Some(&(_, _, _, set)) = ACTIVE_FEATURES.iter()
                        .find(|& &(n, _, _, _)| name == n) {
                        set(&mut features, mi.span);
                        feature_checker.collect(&features, mi.span);
                    }
                    else if let Some(&(_, _, _)) = REMOVED_FEATURES.iter()
                            .find(|& &(n, _, _)| name == n)
                        .or_else(|| STABLE_REMOVED_FEATURES.iter()
                            .find(|& &(n, _, _)| name == n)) {
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

/// A collector for mutually exclusive and interdependent features and their flag spans.
#[derive(Default)]
struct FeatureChecker {
    proc_macro: Option<Span>,
    custom_attribute: Option<Span>,
    copy_closures: Option<Span>,
    clone_closures: Option<Span>,
}

impl FeatureChecker {
    // If this method turns out to be a hotspot due to branching,
    // the branching can be eliminated by modifying `set!()` to set these spans
    // only for the features that need to be checked for mutual exclusion.
    fn collect(&mut self, features: &Features, span: Span) {
        if features.proc_macro {
            // If self.proc_macro is None, set to Some(span)
            self.proc_macro = self.proc_macro.or(Some(span));
        }

        if features.custom_attribute {
            self.custom_attribute = self.custom_attribute.or(Some(span));
        }

        if features.copy_closures {
            self.copy_closures = self.copy_closures.or(Some(span));
        }

        if features.clone_closures {
            self.clone_closures = self.clone_closures.or(Some(span));
        }
    }

    fn check(self, handler: &Handler) {
        if let (Some(pm_span), Some(ca_span)) = (self.proc_macro, self.custom_attribute) {
            handler.struct_span_err(pm_span, "Cannot use `#![feature(proc_macro)]` and \
                                              `#![feature(custom_attribute)] at the same time")
                .span_note(ca_span, "`#![feature(custom_attribute)]` declared here")
                .emit();

            FatalError.raise();
        }

        if let (Some(span), None) = (self.copy_closures, self.clone_closures) {
            handler.struct_span_err(span, "`#![feature(copy_closures)]` can only be used with \
                                           `#![feature(clone_closures)]`")
                  .span_note(span, "`#![feature(copy_closures)]` declared here")
                  .emit();

            FatalError.raise();
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
        features,
        parse_sess: sess,
        plugin_attributes,
    };
    let visitor = &mut PostExpansionVisitor { context: &ctx };
    visitor.whole_crate_feature_gates(krate);
    visit::walk_crate(visitor, krate);
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
                          "#![feature] may not be used on the {} release channel",
                          release_channel);
            }
        }
    }
}
