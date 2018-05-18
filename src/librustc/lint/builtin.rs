// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Some lints that are built in to the compiler.
//!
//! These are the built-in lints that are emitted direct in the main
//! compiler code, rather than using their own custom pass. Those
//! lints are all available in `rustc_lint::builtin`.

use errors::{Applicability, DiagnosticBuilder};
use lint::{LintPass, LateLintPass, LintArray};
use session::Session;
use syntax::codemap::Span;

declare_lint! {
    pub EXCEEDING_BITSHIFTS,
    Deny,
    "shift exceeds the type's number of bits"
}

declare_lint! {
    pub CONST_ERR,
    Warn,
    "constant evaluation detected erroneous expression"
}

declare_lint! {
    pub UNUSED_IMPORTS,
    Warn,
    "imports that are never used"
}

declare_lint! {
    pub UNUSED_EXTERN_CRATES,
    Allow,
    "extern crates that are never used"
}

declare_lint! {
    pub UNUSED_QUALIFICATIONS,
    Allow,
    "detects unnecessarily qualified names"
}

declare_lint! {
    pub UNKNOWN_LINTS,
    Warn,
    "unrecognized lint attribute"
}

declare_lint! {
    pub UNUSED_VARIABLES,
    Warn,
    "detect variables which are not used in any way"
}

declare_lint! {
    pub UNUSED_ASSIGNMENTS,
    Warn,
    "detect assignments that will never be read"
}

declare_lint! {
    pub DEAD_CODE,
    Warn,
    "detect unused, unexported items"
}

declare_lint! {
    pub UNREACHABLE_CODE,
    Warn,
    "detects unreachable code paths"
}

declare_lint! {
    pub UNREACHABLE_PATTERNS,
    Warn,
    "detects unreachable patterns"
}

declare_lint! {
    pub UNUSED_MACROS,
    Warn,
    "detects macros that were not used"
}

declare_lint! {
    pub WARNINGS,
    Warn,
    "mass-change the level for lints which produce warnings"
}

declare_lint! {
    pub UNUSED_FEATURES,
    Warn,
    "unused or unknown features found in crate-level #[feature] directives"
}

declare_lint! {
    pub STABLE_FEATURES,
    Warn,
    "stable features found in #[feature] directive"
}

declare_lint! {
    pub UNKNOWN_CRATE_TYPES,
    Deny,
    "unknown crate type found in #[crate_type] directive"
}

declare_lint! {
    pub TRIVIAL_CASTS,
    Allow,
    "detects trivial casts which could be removed"
}

declare_lint! {
    pub TRIVIAL_NUMERIC_CASTS,
    Allow,
    "detects trivial casts of numeric types which could be removed"
}

declare_lint! {
    pub PRIVATE_IN_PUBLIC,
    Warn,
    "detect private items in public interfaces not caught by the old implementation"
}

declare_lint! {
    pub PUB_USE_OF_PRIVATE_EXTERN_CRATE,
    Deny,
    "detect public re-exports of private extern crates"
}

declare_lint! {
    pub INVALID_TYPE_PARAM_DEFAULT,
    Deny,
    "type parameter default erroneously allowed in invalid location"
}

declare_lint! {
    pub RENAMED_AND_REMOVED_LINTS,
    Warn,
    "lints that have been renamed or removed"
}

declare_lint! {
    pub SAFE_EXTERN_STATICS,
    Deny,
    "safe access to extern statics was erroneously allowed"
}

declare_lint! {
    pub SAFE_PACKED_BORROWS,
    Warn,
    "safe borrows of fields of packed structs were was erroneously allowed"
}

declare_lint! {
    pub PATTERNS_IN_FNS_WITHOUT_BODY,
    Warn,
    "patterns in functions without body were erroneously allowed"
}

declare_lint! {
    pub LEGACY_DIRECTORY_OWNERSHIP,
    Deny,
    "non-inline, non-`#[path]` modules (e.g. `mod foo;`) were erroneously allowed in some files \
     not named `mod.rs`"
}

declare_lint! {
    pub LEGACY_IMPORTS,
    Deny,
    "detects names that resolve to ambiguous glob imports with RFC 1560"
}

declare_lint! {
    pub LEGACY_CONSTRUCTOR_VISIBILITY,
    Deny,
    "detects use of struct constructors that would be invisible with new visibility rules"
}

declare_lint! {
    pub MISSING_FRAGMENT_SPECIFIER,
    Deny,
    "detects missing fragment specifiers in unused `macro_rules!` patterns"
}

declare_lint! {
    pub PARENTHESIZED_PARAMS_IN_TYPES_AND_MODULES,
    Deny,
    "detects parenthesized generic parameters in type and module names"
}

declare_lint! {
    pub LATE_BOUND_LIFETIME_ARGUMENTS,
    Warn,
    "detects generic lifetime arguments in path segments with late bound lifetime parameters"
}

declare_lint! {
    pub INCOHERENT_FUNDAMENTAL_IMPLS,
    Warn,
    "potentially-conflicting impls were erroneously allowed"
}

declare_lint! {
    pub DEPRECATED,
    Warn,
    "detects use of deprecated items"
}

declare_lint! {
    pub UNUSED_UNSAFE,
    Warn,
    "unnecessary use of an `unsafe` block"
}

declare_lint! {
    pub UNUSED_MUT,
    Warn,
    "detect mut variables which don't need to be mutable"
}

declare_lint! {
    pub SINGLE_USE_LIFETIMES,
    Allow,
   "detects single use lifetimes"
}

declare_lint! {
    pub TYVAR_BEHIND_RAW_POINTER,
    Warn,
    "raw pointer to an inference variable"
}

declare_lint! {
    pub ELIDED_LIFETIMES_IN_PATHS,
    Allow,
    "hidden lifetime parameters are deprecated, try `Foo<'_>`"
}

declare_lint! {
    pub BARE_TRAIT_OBJECTS,
    Allow,
    "suggest using `dyn Trait` for trait objects"
}

declare_lint! {
    pub ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,
    Allow,
    "fully qualified paths that start with a module name \
     instead of `crate`, `self`, or an extern crate name"
}

declare_lint! {
    pub ILLEGAL_FLOATING_POINT_LITERAL_PATTERN,
    Warn,
    "floating-point literals cannot be used in patterns"
}

declare_lint! {
    pub UNSTABLE_NAME_COLLISIONS,
    Warn,
    "detects name collision with an existing but unstable method"
}

/// Does nothing as a lint pass, but registers some `Lint`s
/// which are used by other parts of the compiler.
#[derive(Copy, Clone)]
pub struct HardwiredLints;

impl LintPass for HardwiredLints {
    fn get_lints(&self) -> LintArray {
        lint_array!(
            ILLEGAL_FLOATING_POINT_LITERAL_PATTERN,
            EXCEEDING_BITSHIFTS,
            UNUSED_IMPORTS,
            UNUSED_EXTERN_CRATES,
            UNUSED_QUALIFICATIONS,
            UNKNOWN_LINTS,
            UNUSED_VARIABLES,
            UNUSED_ASSIGNMENTS,
            DEAD_CODE,
            UNREACHABLE_CODE,
            UNREACHABLE_PATTERNS,
            UNUSED_MACROS,
            WARNINGS,
            UNUSED_FEATURES,
            STABLE_FEATURES,
            UNKNOWN_CRATE_TYPES,
            TRIVIAL_CASTS,
            TRIVIAL_NUMERIC_CASTS,
            PRIVATE_IN_PUBLIC,
            PUB_USE_OF_PRIVATE_EXTERN_CRATE,
            INVALID_TYPE_PARAM_DEFAULT,
            CONST_ERR,
            RENAMED_AND_REMOVED_LINTS,
            SAFE_EXTERN_STATICS,
            SAFE_PACKED_BORROWS,
            PATTERNS_IN_FNS_WITHOUT_BODY,
            LEGACY_DIRECTORY_OWNERSHIP,
            LEGACY_IMPORTS,
            LEGACY_CONSTRUCTOR_VISIBILITY,
            MISSING_FRAGMENT_SPECIFIER,
            PARENTHESIZED_PARAMS_IN_TYPES_AND_MODULES,
            LATE_BOUND_LIFETIME_ARGUMENTS,
            INCOHERENT_FUNDAMENTAL_IMPLS,
            DEPRECATED,
            UNUSED_UNSAFE,
            UNUSED_MUT,
            SINGLE_USE_LIFETIMES,
            TYVAR_BEHIND_RAW_POINTER,
            ELIDED_LIFETIMES_IN_PATHS,
            BARE_TRAIT_OBJECTS,
            ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,
            UNSTABLE_NAME_COLLISIONS,
        )
    }
}

// this could be a closure, but then implementing derive traits
// becomes hacky (and it gets allocated)
#[derive(PartialEq, RustcEncodable, RustcDecodable, Debug)]
pub enum BuiltinLintDiagnostics {
    Normal,
    BareTraitObject(Span, /* is_global */ bool),
    AbsPathWithModule(Span),
}

impl BuiltinLintDiagnostics {
    pub fn run(self, sess: &Session, db: &mut DiagnosticBuilder) {
        match self {
            BuiltinLintDiagnostics::Normal => (),
            BuiltinLintDiagnostics::BareTraitObject(span, is_global) => {
                let (sugg, app) = match sess.codemap().span_to_snippet(span) {
                    Ok(ref s) if is_global => (format!("dyn ({})", s),
                                               Applicability::MachineApplicable),
                    Ok(s) => (format!("dyn {}", s), Applicability::MachineApplicable),
                    Err(_) => (format!("dyn <type>"), Applicability::HasPlaceholders)
                };
                db.span_suggestion_with_applicability(span, "use `dyn`", sugg, app);
            }
            BuiltinLintDiagnostics::AbsPathWithModule(span) => {
                let (sugg, app) = match sess.codemap().span_to_snippet(span) {
                    Ok(ref s) => {
                        // FIXME(Manishearth) ideally the emitting code
                        // can tell us whether or not this is global
                        let opt_colon = if s.trim_left().starts_with("::") {
                            ""
                        } else {
                            "::"
                        };

                        (format!("crate{}{}", opt_colon, s), Applicability::MachineApplicable)
                    }
                    Err(_) => (format!("crate::<path>"), Applicability::HasPlaceholders)
                };
                db.span_suggestion_with_applicability(span, "use `crate`", sugg, app);
            }
        }
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for HardwiredLints {}
