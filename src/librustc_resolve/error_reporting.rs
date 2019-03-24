use std::cmp::Reverse;

use errors::{Applicability, DiagnosticBuilder, DiagnosticId};
use log::debug;
use rustc::hir::def::{Def, CtorKind, Namespace::*};
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::session::config::nightly_options;
use syntax::ast::{ExprKind};
use syntax::symbol::keywords;
use syntax_pos::Span;

use crate::macros::ParentScope;
use crate::resolve_imports::ImportResolver;
use crate::{import_candidate_to_enum_paths, is_self_type, is_self_value, path_names_to_string};
use crate::{AssocSuggestion, CrateLint, ImportSuggestion, ModuleOrUniformRoot, PathResult,
            PathSource, Resolver, Segment};

impl<'a> Resolver<'a> {
    /// Handles error reporting for `smart_resolve_path_fragment` function.
    /// Creates base error and amends it with one short label and possibly some longer helps/notes.
    pub(crate) fn smart_resolve_report_errors(
        &mut self,
        path: &[Segment],
        span: Span,
        source: PathSource<'_>,
        def: Option<Def>,
    ) -> (DiagnosticBuilder<'a>, Vec<ImportSuggestion>) {
        let ident_span = path.last().map_or(span, |ident| ident.ident.span);
        let ns = source.namespace();
        let is_expected = &|def| source.is_expected(def);
        let is_enum_variant = &|def| if let Def::Variant(..) = def { true } else { false };

        // Make the base error.
        let expected = source.descr_expected();
        let path_str = Segment::names_to_string(path);
        let item_str = path.last().unwrap().ident;
        let code = source.error_code(def.is_some());
        let (base_msg, fallback_label, base_span) = if let Some(def) = def {
            (format!("expected {}, found {} `{}`", expected, def.kind_name(), path_str),
                format!("not a {}", expected),
                span)
        } else {
            let item_span = path.last().unwrap().ident.span;
            let (mod_prefix, mod_str) = if path.len() == 1 {
                (String::new(), "this scope".to_string())
            } else if path.len() == 2 && path[0].ident.name == keywords::PathRoot.name() {
                (String::new(), "the crate root".to_string())
            } else {
                let mod_path = &path[..path.len() - 1];
                let mod_prefix = match self.resolve_path_without_parent_scope(
                    mod_path, Some(TypeNS), false, span, CrateLint::No
                ) {
                    PathResult::Module(ModuleOrUniformRoot::Module(module)) =>
                        module.def(),
                    _ => None,
                }.map_or(String::new(), |def| format!("{} ", def.kind_name()));
                (mod_prefix, format!("`{}`", Segment::names_to_string(mod_path)))
            };
            (format!("cannot find {} `{}` in {}{}", expected, item_str, mod_prefix, mod_str),
                format!("not found in {}", mod_str),
                item_span)
        };

        let code = DiagnosticId::Error(code.into());
        let mut err = self.session.struct_span_err_with_code(base_span, &base_msg, code);

        // Emit help message for fake-self from other languages (e.g., `this` in Javascript).
        if ["this", "my"].contains(&&*item_str.as_str())
            && self.self_value_is_available(path[0].ident.span, span) {
            err.span_suggestion(
                span,
                "did you mean",
                "self".to_string(),
                Applicability::MaybeIncorrect,
            );
        }

        // Emit special messages for unresolved `Self` and `self`.
        if is_self_type(path, ns) {
            __diagnostic_used!(E0411);
            err.code(DiagnosticId::Error("E0411".into()));
            err.span_label(span, format!("`Self` is only available in impls, traits, \
                                          and type definitions"));
            return (err, Vec::new());
        }
        if is_self_value(path, ns) {
            debug!("smart_resolve_path_fragment: E0424, source={:?}", source);

            __diagnostic_used!(E0424);
            err.code(DiagnosticId::Error("E0424".into()));
            err.span_label(span, match source {
                PathSource::Pat => {
                    format!("`self` value is a keyword \
                             and may not be bound to \
                             variables or shadowed")
                }
                _ => {
                    format!("`self` value is a keyword \
                             only available in methods \
                             with `self` parameter")
                }
            });
            return (err, Vec::new());
        }

        // Try to lookup name in more relaxed fashion for better error reporting.
        let ident = path.last().unwrap().ident;
        let candidates = self.lookup_import_candidates(ident, ns, is_expected)
            .drain(..)
            .filter(|ImportSuggestion { did, .. }| {
                match (did, def.and_then(|def| def.opt_def_id())) {
                    (Some(suggestion_did), Some(actual_did)) => *suggestion_did != actual_did,
                    _ => true,
                }
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() && is_expected(Def::Enum(DefId::local(CRATE_DEF_INDEX))) {
            let enum_candidates =
                self.lookup_import_candidates(ident, ns, is_enum_variant);
            let mut enum_candidates = enum_candidates.iter()
                .map(|suggestion| {
                    import_candidate_to_enum_paths(&suggestion)
                }).collect::<Vec<_>>();
            enum_candidates.sort();

            if !enum_candidates.is_empty() {
                // Contextualize for E0412 "cannot find type", but don't belabor the point
                // (that it's a variant) for E0573 "expected type, found variant".
                let preamble = if def.is_none() {
                    let others = match enum_candidates.len() {
                        1 => String::new(),
                        2 => " and 1 other".to_owned(),
                        n => format!(" and {} others", n)
                    };
                    format!("there is an enum variant `{}`{}; ",
                            enum_candidates[0].0, others)
                } else {
                    String::new()
                };
                let msg = format!("{}try using the variant's enum", preamble);

                err.span_suggestions(
                    span,
                    &msg,
                    enum_candidates.into_iter()
                        .map(|(_variant_path, enum_ty_path)| enum_ty_path)
                        // Variants re-exported in prelude doesn't mean `prelude::v1` is the
                        // type name!
                        // FIXME: is there a more principled way to do this that
                        // would work for other re-exports?
                        .filter(|enum_ty_path| enum_ty_path != "std::prelude::v1")
                        // Also write `Option` rather than `std::prelude::v1::Option`.
                        .map(|enum_ty_path| {
                            // FIXME #56861: DRY-er prelude filtering.
                            enum_ty_path.trim_start_matches("std::prelude::v1::").to_owned()
                        }),
                    Applicability::MachineApplicable,
                );
            }
        }
        if path.len() == 1 && self.self_type_is_available(span) {
            if let Some(candidate) = self.lookup_assoc_candidate(ident, ns, is_expected) {
                let self_is_available = self.self_value_is_available(path[0].ident.span, span);
                match candidate {
                    AssocSuggestion::Field => {
                        err.span_suggestion(
                            span,
                            "try",
                            format!("self.{}", path_str),
                            Applicability::MachineApplicable,
                        );
                        if !self_is_available {
                            err.span_label(span, format!("`self` value is a keyword \
                                                         only available in \
                                                         methods with `self` parameter"));
                        }
                    }
                    AssocSuggestion::MethodWithSelf if self_is_available => {
                        err.span_suggestion(
                            span,
                            "try",
                            format!("self.{}", path_str),
                            Applicability::MachineApplicable,
                        );
                    }
                    AssocSuggestion::MethodWithSelf | AssocSuggestion::AssocItem => {
                        err.span_suggestion(
                            span,
                            "try",
                            format!("Self::{}", path_str),
                            Applicability::MachineApplicable,
                        );
                    }
                }
                return (err, candidates);
            }
        }

        let mut levenshtein_worked = false;

        // Try Levenshtein algorithm.
        let suggestion = self.lookup_typo_candidate(path, ns, is_expected, span);
        if let Some(suggestion) = suggestion {
            let msg = format!(
                "{} {} with a similar name exists",
                suggestion.article, suggestion.kind
            );
            err.span_suggestion(
                ident_span,
                &msg,
                suggestion.candidate.to_string(),
                Applicability::MaybeIncorrect,
            );

            levenshtein_worked = true;
        }

        // Try context-dependent help if relaxed lookup didn't work.
        if let Some(def) = def {
            if self.smart_resolve_context_dependent_help(&mut err,
                                                         span,
                                                         source,
                                                         def,
                                                         &path_str,
                                                         &fallback_label) {
                return (err, candidates);
            }
        }

        // Fallback label.
        if !levenshtein_worked {
            err.span_label(base_span, fallback_label);
            self.type_ascription_suggestion(&mut err, base_span);
        }
        (err, candidates)
    }

    /// Provides context-dependent help for errors reported by the `smart_resolve_path_fragment`
    /// function.
    /// Returns `true` if able to provide context-dependent help.
    fn smart_resolve_context_dependent_help(
        &mut self,
        err: &mut DiagnosticBuilder<'a>,
        span: Span,
        source: PathSource<'_>,
        def: Def,
        path_str: &str,
        fallback_label: &str,
    ) -> bool {
        let ns = source.namespace();
        let is_expected = &|def| source.is_expected(def);

        match (def, source) {
            (Def::Macro(..), _) => {
                err.span_suggestion(
                    span,
                    "use `!` to invoke the macro",
                    format!("{}!", path_str),
                    Applicability::MaybeIncorrect,
                );
                if path_str == "try" && span.rust_2015() {
                    err.note("if you want the `try` keyword, \
                        you need to be in the 2018 edition");
                }
            }
            (Def::TyAlias(..), PathSource::Trait(_)) => {
                err.span_label(span, "type aliases cannot be used as traits");
                if nightly_options::is_nightly_build() {
                    err.note("did you mean to use a trait alias?");
                }
            }
            (Def::Mod(..), PathSource::Expr(Some(parent))) => match parent.node {
                ExprKind::Field(_, ident) => {
                    err.span_suggestion(
                        parent.span,
                        "use the path separator to refer to an item",
                        format!("{}::{}", path_str, ident),
                        Applicability::MaybeIncorrect,
                    );
                }
                ExprKind::MethodCall(ref segment, ..) => {
                    let span = parent.span.with_hi(segment.ident.span.hi());
                    err.span_suggestion(
                        span,
                        "use the path separator to refer to an item",
                        format!("{}::{}", path_str, segment.ident),
                        Applicability::MaybeIncorrect,
                    );
                }
                _ => return false,
            },
            (Def::Enum(..), PathSource::TupleStruct)
                | (Def::Enum(..), PathSource::Expr(..))  => {
                if let Some(variants) = self.collect_enum_variants(def) {
                    if !variants.is_empty() {
                        let msg = if variants.len() == 1 {
                            "try using the enum's variant"
                        } else {
                            "try using one of the enum's variants"
                        };

                        err.span_suggestions(
                            span,
                            msg,
                            variants.iter().map(path_names_to_string),
                            Applicability::MaybeIncorrect,
                        );
                    }
                } else {
                    err.note("did you mean to use one of the enum's variants?");
                }
            },
            (Def::Struct(def_id), _) if ns == ValueNS => {
                if let Some((ctor_def, ctor_vis))
                        = self.struct_constructors.get(&def_id).cloned() {
                    let accessible_ctor = self.is_accessible(ctor_vis);
                    if is_expected(ctor_def) && !accessible_ctor {
                        err.span_label(span, format!("constructor is not visible \
                                                      here due to private fields"));
                    }
                } else {
                    // HACK(estebank): find a better way to figure out that this was a
                    // parser issue where a struct literal is being used on an expression
                    // where a brace being opened means a block is being started. Look
                    // ahead for the next text to see if `span` is followed by a `{`.
                    let sm = self.session.source_map();
                    let mut sp = span;
                    loop {
                        sp = sm.next_point(sp);
                        match sm.span_to_snippet(sp) {
                            Ok(ref snippet) => {
                                if snippet.chars().any(|c| { !c.is_whitespace() }) {
                                    break;
                                }
                            }
                            _ => break,
                        }
                    }
                    let followed_by_brace = match sm.span_to_snippet(sp) {
                        Ok(ref snippet) if snippet == "{" => true,
                        _ => false,
                    };
                    // In case this could be a struct literal that needs to be surrounded
                    // by parenthesis, find the appropriate span.
                    let mut i = 0;
                    let mut closing_brace = None;
                    loop {
                        sp = sm.next_point(sp);
                        match sm.span_to_snippet(sp) {
                            Ok(ref snippet) => {
                                if snippet == "}" {
                                    let sp = span.to(sp);
                                    if let Ok(snippet) = sm.span_to_snippet(sp) {
                                        closing_brace = Some((sp, snippet));
                                    }
                                    break;
                                }
                            }
                            _ => break,
                        }
                        i += 1;
                        // The bigger the span, the more likely we're incorrect --
                        // bound it to 100 chars long.
                        if i > 100 {
                            break;
                        }
                    }
                    match source {
                        PathSource::Expr(Some(parent)) => {
                            match parent.node {
                                ExprKind::MethodCall(ref path_assignment, _)  => {
                                    err.span_suggestion(
                                        sm.start_point(parent.span)
                                            .to(path_assignment.ident.span),
                                        "use `::` to access an associated function",
                                        format!("{}::{}",
                                                path_str,
                                                path_assignment.ident),
                                        Applicability::MaybeIncorrect
                                    );
                                },
                                _ => {
                                    err.span_label(
                                        span,
                                        format!("did you mean `{} {{ /* fields */ }}`?",
                                                path_str),
                                    );
                                },
                            }
                        },
                        PathSource::Expr(None) if followed_by_brace == true => {
                            if let Some((sp, snippet)) = closing_brace {
                                err.span_suggestion(
                                    sp,
                                    "surround the struct literal with parenthesis",
                                    format!("({})", snippet),
                                    Applicability::MaybeIncorrect,
                                );
                            } else {
                                err.span_label(
                                    span,
                                    format!("did you mean `({} {{ /* fields */ }})`?",
                                            path_str),
                                );
                            }
                        },
                        _ => {
                            err.span_label(
                                span,
                                format!("did you mean `{} {{ /* fields */ }}`?",
                                        path_str),
                            );
                        },
                    }
                }
            }
            (Def::Union(..), _) |
            (Def::Variant(..), _) |
            (Def::Ctor(_, _, CtorKind::Fictive), _) if ns == ValueNS => {
                err.span_label(span, format!("did you mean `{} {{ /* fields */ }}`?",
                                             path_str));
            }
            (Def::SelfTy(..), _) if ns == ValueNS => {
                err.span_label(span, fallback_label);
                err.note("can't use `Self` as a constructor, you must use the \
                          implemented struct");
            }
            (Def::TyAlias(_), _) | (Def::AssociatedTy(..), _) if ns == ValueNS => {
                err.note("can't use a type alias as a constructor");
            }
            _ => return false,
        }
        true
    }
}

impl<'a, 'b:'a> ImportResolver<'a, 'b> {
    /// Adds suggestions for a path that cannot be resolved.
    pub(crate) fn make_path_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        debug!("make_path_suggestion: span={:?} path={:?}", span, path);

        match (path.get(0), path.get(1)) {
            // `{{root}}::ident::...` on both editions.
            // On 2015 `{{root}}` is usually added implicitly.
            (Some(fst), Some(snd)) if fst.ident.name == keywords::PathRoot.name() &&
                                      !snd.ident.is_path_segment_keyword() => {}
            // `ident::...` on 2018.
            (Some(fst), _) if fst.ident.span.rust_2018() &&
                              !fst.ident.is_path_segment_keyword() => {
                // Insert a placeholder that's later replaced by `self`/`super`/etc.
                path.insert(0, Segment::from_ident(keywords::Invalid.ident()));
            }
            _ => return None,
        }

        self.make_missing_self_suggestion(span, path.clone(), parent_scope)
            .or_else(|| self.make_missing_crate_suggestion(span, path.clone(), parent_scope))
            .or_else(|| self.make_missing_super_suggestion(span, path.clone(), parent_scope))
            .or_else(|| self.make_external_crate_suggestion(span, path, parent_scope))
    }

    /// Suggest a missing `self::` if that resolves to an correct module.
    ///
    /// ```
    ///    |
    /// LL | use foo::Bar;
    ///    |     ^^^ did you mean `self::foo`?
    /// ```
    fn make_missing_self_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        // Replace first ident with `self` and check if that is valid.
        path[0].ident.name = keywords::SelfLower.name();
        let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_self_suggestion: path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((path, None))
        } else {
            None
        }
    }

    /// Suggests a missing `crate::` if that resolves to an correct module.
    ///
    /// ```
    ///    |
    /// LL | use foo::Bar;
    ///    |     ^^^ did you mean `crate::foo`?
    /// ```
    fn make_missing_crate_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        // Replace first ident with `crate` and check if that is valid.
        path[0].ident.name = keywords::Crate.name();
        let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_crate_suggestion:  path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((
                path,
                Some(
                    "`use` statements changed in Rust 2018; read more at \
                     <https://doc.rust-lang.org/edition-guide/rust-2018/module-system/path-\
                     clarity.html>".to_string()
                ),
            ))
        } else {
            None
        }
    }

    /// Suggests a missing `super::` if that resolves to an correct module.
    ///
    /// ```
    ///    |
    /// LL | use foo::Bar;
    ///    |     ^^^ did you mean `super::foo`?
    /// ```
    fn make_missing_super_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        // Replace first ident with `crate` and check if that is valid.
        path[0].ident.name = keywords::Super.name();
        let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_super_suggestion:  path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((path, None))
        } else {
            None
        }
    }

    /// Suggests a missing external crate name if that resolves to an correct module.
    ///
    /// ```
    ///    |
    /// LL | use foobar::Baz;
    ///    |     ^^^^^^ did you mean `baz::foobar`?
    /// ```
    ///
    /// Used when importing a submodule of an external crate but missing that crate's
    /// name as the first part of path.
    fn make_external_crate_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        if path[1].ident.span.rust_2015() {
            return None;
        }

        // Sort extern crate names in reverse order to get
        // 1) some consistent ordering for emitted dignostics, and
        // 2) `std` suggestions before `core` suggestions.
        let mut extern_crate_names =
            self.resolver.extern_prelude.iter().map(|(ident, _)| ident.name).collect::<Vec<_>>();
        extern_crate_names.sort_by_key(|name| Reverse(name.as_str()));

        for name in extern_crate_names.into_iter() {
            // Replace first ident with a crate name and check if that is valid.
            path[0].ident.name = name;
            let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
            debug!("make_external_crate_suggestion: name={:?} path={:?} result={:?}",
                    name, path, result);
            if let PathResult::Module(..) = result {
                return Some((path, None));
            }
        }

        None
    }
}
