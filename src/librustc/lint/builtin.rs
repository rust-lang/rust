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
use syntax::ast;
use syntax::source_map::Span;

declare_lint! {
    pub EXCEEDING_BITSHIFTS,
    Deny,
    "shift exceeds the type's number of bits"
}

declare_lint! {
    pub CONST_ERR,
    Deny,
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
    "detects unreachable code paths",
    report_in_external_macro: true
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
    "unused features found in crate-level #[feature] directives"
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
    Deny,
    "potentially-conflicting impls were erroneously allowed"
}

declare_lint! {
    pub BAD_REPR,
    Warn,
    "detects incorrect use of `repr` attribute"
}

declare_lint! {
    pub DEPRECATED,
    Warn,
    "detects use of deprecated items",
    report_in_external_macro: true
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
    pub UNCONDITIONAL_RECURSION,
    Warn,
    "functions that cannot return without calling themselves"
}

declare_lint! {
    pub SINGLE_USE_LIFETIMES,
    Allow,
    "detects lifetime parameters that are only used once"
}

declare_lint! {
    pub UNUSED_LIFETIMES,
    Allow,
    "detects lifetime parameters that are never used"
}

declare_lint! {
    pub TYVAR_BEHIND_RAW_POINTER,
    Warn,
    "raw pointer to an inference variable"
}

declare_lint! {
    pub ELIDED_LIFETIMES_IN_PATHS,
    Allow,
    "hidden lifetime parameters in types are deprecated"
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

declare_lint! {
    pub IRREFUTABLE_LET_PATTERNS,
    Deny,
    "detects irrefutable patterns in if-let and while-let statements"
}

declare_lint! {
    pub UNUSED_LABELS,
    Allow,
    "detects labels that are never used"
}

declare_lint! {
    pub DUPLICATE_MACRO_EXPORTS,
    Deny,
    "detects duplicate macro exports"
}

declare_lint! {
    pub INTRA_DOC_LINK_RESOLUTION_FAILURE,
    Warn,
    "warn about documentation intra links resolution failure"
}

declare_lint! {
    pub MISSING_DOC_CODE_EXAMPLES,
    Allow,
    "warn about missing code example in an item's documentation"
}

declare_lint! {
    pub PRIVATE_DOC_TESTS,
    Allow,
    "warn about doc test in private item"
}

declare_lint! {
    pub WHERE_CLAUSES_OBJECT_SAFETY,
    Warn,
    "checks the object safety of where clauses"
}

declare_lint! {
    pub PROC_MACRO_DERIVE_RESOLUTION_FALLBACK,
    Warn,
    "detects proc macro derives using inaccessible names from parent modules"
}

declare_lint! {
    pub MACRO_USE_EXTERN_CRATE,
    Allow,
    "the `#[macro_use]` attribute is now deprecated in favor of using macros \
     via the module system"
}

declare_lint! {
    pub MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
    Deny,
    "macro-expanded `macro_export` macros from the current crate \
     cannot be referred to by absolute paths"
}

declare_lint! {
    pub EXPLICIT_OUTLIVES_REQUIREMENTS,
    Allow,
    "outlives requirements can be inferred"
}

/// Some lints that are buffered from `libsyntax`. See `syntax::early_buffered_lints`.
pub mod parser {
    declare_lint! {
        pub QUESTION_MARK_MACRO_SEP,
        Allow,
        "detects the use of `?` as a macro separator"
    }

    declare_lint! {
        pub INCORRECT_MACRO_FRAGMENT_REPETITION,
        Warn,
        "detects incorrect macro fragment follow due to repetition"
    }
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
            LEGACY_CONSTRUCTOR_VISIBILITY,
            MISSING_FRAGMENT_SPECIFIER,
            PARENTHESIZED_PARAMS_IN_TYPES_AND_MODULES,
            LATE_BOUND_LIFETIME_ARGUMENTS,
            INCOHERENT_FUNDAMENTAL_IMPLS,
            DEPRECATED,
            UNUSED_UNSAFE,
            UNUSED_MUT,
            UNCONDITIONAL_RECURSION,
            SINGLE_USE_LIFETIMES,
            UNUSED_LIFETIMES,
            UNUSED_LABELS,
            TYVAR_BEHIND_RAW_POINTER,
            ELIDED_LIFETIMES_IN_PATHS,
            BARE_TRAIT_OBJECTS,
            ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,
            UNSTABLE_NAME_COLLISIONS,
            IRREFUTABLE_LET_PATTERNS,
            DUPLICATE_MACRO_EXPORTS,
            INTRA_DOC_LINK_RESOLUTION_FAILURE,
            MISSING_DOC_CODE_EXAMPLES,
            PRIVATE_DOC_TESTS,
            WHERE_CLAUSES_OBJECT_SAFETY,
            PROC_MACRO_DERIVE_RESOLUTION_FALLBACK,
            MACRO_USE_EXTERN_CRATE,
            MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
            parser::QUESTION_MARK_MACRO_SEP,
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
    DuplicatedMacroExports(ast::Ident, Span, Span),
    ProcMacroDeriveResolutionFallback(Span),
    MacroExpandedMacroExportsAccessedByAbsolutePaths(Span),
    ElidedLifetimesInPaths(usize, Span, bool, Span, String),
    UnknownCrateTypes(Span, String, String),
    IncorrectMacroFragmentRepetition {
        span: Span,
        token_span: Span,
        sugg_span: Span,
        frag: String,
        possible: Vec<String>,
    }
}

impl BuiltinLintDiagnostics {
    pub fn run(self, sess: &Session, db: &mut DiagnosticBuilder<'_>) {
        match self {
            BuiltinLintDiagnostics::Normal => (),
            BuiltinLintDiagnostics::BareTraitObject(span, is_global) => {
                let (sugg, app) = match sess.source_map().span_to_snippet(span) {
                    Ok(ref s) if is_global => (format!("dyn ({})", s),
                                               Applicability::MachineApplicable),
                    Ok(s) => (format!("dyn {}", s), Applicability::MachineApplicable),
                    Err(_) => ("dyn <type>".to_string(), Applicability::HasPlaceholders)
                };
                db.span_suggestion_with_applicability(span, "use `dyn`", sugg, app);
            }
            BuiltinLintDiagnostics::AbsPathWithModule(span) => {
                let (sugg, app) = match sess.source_map().span_to_snippet(span) {
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
                    Err(_) => ("crate::<path>".to_string(), Applicability::HasPlaceholders)
                };
                db.span_suggestion_with_applicability(span, "use `crate`", sugg, app);
            }
            BuiltinLintDiagnostics::DuplicatedMacroExports(ident, earlier_span, later_span) => {
                db.span_label(later_span, format!("`{}` already exported", ident));
                db.span_note(earlier_span, "previous macro export is now shadowed");
            }
            BuiltinLintDiagnostics::ProcMacroDeriveResolutionFallback(span) => {
                db.span_label(span, "names from parent modules are not \
                                     accessible without an explicit import");
            }
            BuiltinLintDiagnostics::MacroExpandedMacroExportsAccessedByAbsolutePaths(span_def) => {
                db.span_note(span_def, "the macro is defined here");
            }
            BuiltinLintDiagnostics::ElidedLifetimesInPaths(
                n, path_span, incl_angl_brckt, insertion_span, anon_lts
            ) => {
                let (replace_span, suggestion) = if incl_angl_brckt {
                    (insertion_span, anon_lts)
                } else {
                    // When possible, prefer a suggestion that replaces the whole
                    // `Path<T>` expression with `Path<'_, T>`, rather than inserting `'_, `
                    // at a point (which makes for an ugly/confusing label)
                    if let Ok(snippet) = sess.source_map().span_to_snippet(path_span) {
                        // But our spans can get out of whack due to macros; if the place we think
                        // we want to insert `'_` isn't even within the path expression's span, we
                        // should bail out of making any suggestion rather than panicking on a
                        // subtract-with-overflow or string-slice-out-out-bounds (!)
                        // FIXME: can we do better?
                        if insertion_span.lo().0 < path_span.lo().0 {
                            return;
                        }
                        let insertion_index = (insertion_span.lo().0 - path_span.lo().0) as usize;
                        if insertion_index > snippet.len() {
                            return;
                        }
                        let (before, after) = snippet.split_at(insertion_index);
                        (path_span, format!("{}{}{}", before, anon_lts, after))
                    } else {
                        (insertion_span, anon_lts)
                    }
                };
                db.span_suggestion_with_applicability(
                    replace_span,
                    &format!("indicate the anonymous lifetime{}", if n >= 2 { "s" } else { "" }),
                    suggestion,
                    Applicability::MachineApplicable
                );
            }
            BuiltinLintDiagnostics::UnknownCrateTypes(span, note, sugg) => {
                db.span_suggestion_with_applicability(
                    span,
                    &note,
                    sugg,
                    Applicability::MaybeIncorrect
                );
            }
            BuiltinLintDiagnostics::IncorrectMacroFragmentRepetition {
                span,
                token_span,
                sugg_span,
                frag,
                possible,
            } => {
                if span == token_span {
                    db.span_label(
                        span,
                        "this fragment is followed by itself without a valid separator",
                    );
                } else {
                    db.span_label(
                        span,
                        "this fragment is followed by the first fragment in this repetition \
                         without a valid separator",
                    );
                    db.span_label(
                        token_span,
                        "this is the first fragment in the evaluated repetition",
                    );
                }
                let msg = "allowed there are: ";
                let sugg_msg = "add a valid separator for the repetition to be unambiguous";
                match &possible[..] {
                    &[] => {}
                    &[ref t] => {
                        db.note(&format!("only {} is allowed after `{}` fragments", t, frag));
                        if t.starts_with('`') && t.ends_with('`') {
                            db.span_suggestion_with_applicability(
                                sugg_span,
                                &format!("{}, for example", sugg_msg),
                                (&t[1..t.len()-1]).to_owned(),
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            db.note(sugg_msg);
                        }
                    }
                    _ => {
                        db.note(&format!(
                            "{}{} or {}",
                            msg,
                            possible[..possible.len() - 1].iter().map(|s| s.to_owned())
                                .collect::<Vec<_>>().join(", "),
                            possible[possible.len() - 1],
                        ));
                        let mut note = true;
                        for t in &possible {
                            if t.starts_with('`') && t.ends_with('`') {
                                db.span_suggestion_with_applicability(
                                    sugg_span,
                                    &format!("{}, for example", sugg_msg),
                                    (&t[1..t.len()-1]).to_owned(),
                                    Applicability::MaybeIncorrect,
                                );
                                note = false;
                                break;
                            }
                        }
                        if note {
                            db.note(sugg_msg);
                        }
                    }
                }
            }
        }
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for HardwiredLints {}
