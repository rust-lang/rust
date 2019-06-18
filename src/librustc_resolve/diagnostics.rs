use std::cmp::Reverse;

use errors::{Applicability, DiagnosticBuilder, DiagnosticId};
use log::debug;
use rustc::hir::def::{self, DefKind, CtorKind, Namespace::*};
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::session::{Session, config::nightly_options};
use syntax::ast::{self, Expr, ExprKind, Ident};
use syntax::ext::base::MacroKind;
use syntax::symbol::{Symbol, kw};
use syntax_pos::{BytePos, Span};

type Res = def::Res<ast::NodeId>;

use crate::macros::ParentScope;
use crate::resolve_imports::{ImportDirective, ImportDirectiveSubclass, ImportResolver};
use crate::{import_candidate_to_enum_paths, is_self_type, is_self_value, path_names_to_string};
use crate::{AssocSuggestion, CrateLint, ImportSuggestion, ModuleOrUniformRoot, PathResult,
            PathSource, Resolver, Segment, Suggestion};

impl<'a> Resolver<'a> {
    /// Handles error reporting for `smart_resolve_path_fragment` function.
    /// Creates base error and amends it with one short label and possibly some longer helps/notes.
    pub(crate) fn smart_resolve_report_errors(
        &mut self,
        path: &[Segment],
        span: Span,
        source: PathSource<'_>,
        res: Option<Res>,
    ) -> (DiagnosticBuilder<'a>, Vec<ImportSuggestion>) {
        let ident_span = path.last().map_or(span, |ident| ident.ident.span);
        let ns = source.namespace();
        let is_expected = &|res| source.is_expected(res);
        let is_enum_variant = &|res| {
            if let Res::Def(DefKind::Variant, _) = res { true } else { false }
        };

        // Make the base error.
        let expected = source.descr_expected();
        let path_str = Segment::names_to_string(path);
        let item_str = path.last().unwrap().ident;
        let code = source.error_code(res.is_some());
        let (base_msg, fallback_label, base_span) = if let Some(res) = res {
            (format!("expected {}, found {} `{}`", expected, res.descr(), path_str),
                format!("not a {}", expected),
                span)
        } else {
            let item_span = path.last().unwrap().ident.span;
            let (mod_prefix, mod_str) = if path.len() == 1 {
                (String::new(), "this scope".to_string())
            } else if path.len() == 2 && path[0].ident.name == kw::PathRoot {
                (String::new(), "the crate root".to_string())
            } else {
                let mod_path = &path[..path.len() - 1];
                let mod_prefix = match self.resolve_path_without_parent_scope(
                    mod_path, Some(TypeNS), false, span, CrateLint::No
                ) {
                    PathResult::Module(ModuleOrUniformRoot::Module(module)) =>
                        module.def_kind(),
                    _ => None,
                }.map_or(String::new(), |kind| format!("{} ", kind.descr()));
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
                match (did, res.and_then(|res| res.opt_def_id())) {
                    (Some(suggestion_did), Some(actual_did)) => *suggestion_did != actual_did,
                    _ => true,
                }
            })
            .collect::<Vec<_>>();
        let crate_def_id = DefId::local(CRATE_DEF_INDEX);
        if candidates.is_empty() && is_expected(Res::Def(DefKind::Enum, crate_def_id)) {
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
                let preamble = if res.is_none() {
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
                        if self_is_available {
                            err.span_suggestion(
                                span,
                                "you might have meant to use the available field",
                                format!("self.{}", path_str),
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.span_label(
                                span,
                                "a field by this name exists in `Self`",
                            );
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
        if let Some(res) = res {
            if self.smart_resolve_context_dependent_help(&mut err,
                                                         span,
                                                         source,
                                                         res,
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

    fn followed_by_brace(&self, span: Span) -> (bool, Option<(Span, String)>) {
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
        return (followed_by_brace, closing_brace)
    }

    /// Provides context-dependent help for errors reported by the `smart_resolve_path_fragment`
    /// function.
    /// Returns `true` if able to provide context-dependent help.
    fn smart_resolve_context_dependent_help(
        &mut self,
        err: &mut DiagnosticBuilder<'a>,
        span: Span,
        source: PathSource<'_>,
        res: Res,
        path_str: &str,
        fallback_label: &str,
    ) -> bool {
        let ns = source.namespace();
        let is_expected = &|res| source.is_expected(res);

        let path_sep = |err: &mut DiagnosticBuilder<'_>, expr: &Expr| match expr.node {
            ExprKind::Field(_, ident) => {
                err.span_suggestion(
                    expr.span,
                    "use the path separator to refer to an item",
                    format!("{}::{}", path_str, ident),
                    Applicability::MaybeIncorrect,
                );
                true
            }
            ExprKind::MethodCall(ref segment, ..) => {
                let span = expr.span.with_hi(segment.ident.span.hi());
                err.span_suggestion(
                    span,
                    "use the path separator to refer to an item",
                    format!("{}::{}", path_str, segment.ident),
                    Applicability::MaybeIncorrect,
                );
                true
            }
            _ => false,
        };

        let mut bad_struct_syntax_suggestion = || {
            let (followed_by_brace, closing_brace) = self.followed_by_brace(span);
            let mut suggested = false;
            match source {
                PathSource::Expr(Some(parent)) => {
                    suggested = path_sep(err, &parent);
                }
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
                            span,  // Note the parenthesis surrounding the suggestion below
                            format!("did you mean `({} {{ /* fields */ }})`?", path_str),
                        );
                    }
                    suggested = true;
                },
                _ => {}
            }
            if !suggested {
                err.span_label(
                    span,
                    format!("did you mean `{} {{ /* fields */ }}`?", path_str),
                );
            }
        };

        match (res, source) {
            (Res::Def(DefKind::Macro(..), _), _) => {
                err.span_suggestion(
                    span,
                    "use `!` to invoke the macro",
                    format!("{}!", path_str),
                    Applicability::MaybeIncorrect,
                );
                if path_str == "try" && span.rust_2015() {
                    err.note("if you want the `try` keyword, you need to be in the 2018 edition");
                }
            }
            (Res::Def(DefKind::TyAlias, _), PathSource::Trait(_)) => {
                err.span_label(span, "type aliases cannot be used as traits");
                if nightly_options::is_nightly_build() {
                    err.note("did you mean to use a trait alias?");
                }
            }
            (Res::Def(DefKind::Mod, _), PathSource::Expr(Some(parent))) => {
                if !path_sep(err, &parent) {
                    return false;
                }
            }
            (Res::Def(DefKind::Enum, def_id), PathSource::TupleStruct)
                | (Res::Def(DefKind::Enum, def_id), PathSource::Expr(..))  => {
                if let Some(variants) = self.collect_enum_variants(def_id) {
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
            (Res::Def(DefKind::Struct, def_id), _) if ns == ValueNS => {
                if let Some((ctor_def, ctor_vis))
                        = self.struct_constructors.get(&def_id).cloned() {
                    let accessible_ctor = self.is_accessible(ctor_vis);
                    if is_expected(ctor_def) && !accessible_ctor {
                        err.span_label(
                            span,
                            format!("constructor is not visible here due to private fields"),
                        );
                    }
                } else {
                    bad_struct_syntax_suggestion();
                }
            }
            (Res::Def(DefKind::Union, _), _) |
            (Res::Def(DefKind::Variant, _), _) |
            (Res::Def(DefKind::Ctor(_, CtorKind::Fictive), _), _) if ns == ValueNS => {
                bad_struct_syntax_suggestion();
            }
            (Res::SelfTy(..), _) if ns == ValueNS => {
                err.span_label(span, fallback_label);
                err.note("can't use `Self` as a constructor, you must use the implemented struct");
            }
            (Res::Def(DefKind::TyAlias, _), _)
            | (Res::Def(DefKind::AssocTy, _), _) if ns == ValueNS => {
                err.note("can't use a type alias as a constructor");
            }
            _ => return false,
        }
        true
    }
}

impl<'a, 'b> ImportResolver<'a, 'b> {
    /// Adds suggestions for a path that cannot be resolved.
    pub(crate) fn make_path_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Vec<String>)> {
        debug!("make_path_suggestion: span={:?} path={:?}", span, path);

        match (path.get(0), path.get(1)) {
            // `{{root}}::ident::...` on both editions.
            // On 2015 `{{root}}` is usually added implicitly.
            (Some(fst), Some(snd)) if fst.ident.name == kw::PathRoot &&
                                      !snd.ident.is_path_segment_keyword() => {}
            // `ident::...` on 2018.
            (Some(fst), _) if fst.ident.span.rust_2018() &&
                              !fst.ident.is_path_segment_keyword() => {
                // Insert a placeholder that's later replaced by `self`/`super`/etc.
                path.insert(0, Segment::from_ident(Ident::invalid()));
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
    ) -> Option<(Vec<Segment>, Vec<String>)> {
        // Replace first ident with `self` and check if that is valid.
        path[0].ident.name = kw::SelfLower;
        let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_self_suggestion: path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((path, Vec::new()))
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
    ) -> Option<(Vec<Segment>, Vec<String>)> {
        // Replace first ident with `crate` and check if that is valid.
        path[0].ident.name = kw::Crate;
        let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_crate_suggestion:  path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((
                path,
                vec![
                    "`use` statements changed in Rust 2018; read more at \
                     <https://doc.rust-lang.org/edition-guide/rust-2018/module-system/path-\
                     clarity.html>".to_string()
                ],
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
    ) -> Option<(Vec<Segment>, Vec<String>)> {
        // Replace first ident with `crate` and check if that is valid.
        path[0].ident.name = kw::Super;
        let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_super_suggestion:  path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((path, Vec::new()))
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
    ) -> Option<(Vec<Segment>, Vec<String>)> {
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
                return Some((path, Vec::new()));
            }
        }

        None
    }

    /// Suggests importing a macro from the root of the crate rather than a module within
    /// the crate.
    ///
    /// ```
    /// help: a macro with this name exists at the root of the crate
    ///    |
    /// LL | use issue_59764::makro;
    ///    |     ^^^^^^^^^^^^^^^^^^
    ///    |
    ///    = note: this could be because a macro annotated with `#[macro_export]` will be exported
    ///            at the root of the crate instead of the module where it is defined
    /// ```
    pub(crate) fn check_for_module_export_macro(
        &self,
        directive: &'b ImportDirective<'b>,
        module: ModuleOrUniformRoot<'b>,
        ident: Ident,
    ) -> Option<(Option<Suggestion>, Vec<String>)> {
        let mut crate_module = if let ModuleOrUniformRoot::Module(module) = module {
            module
        } else {
            return None;
        };

        while let Some(parent) = crate_module.parent {
            crate_module = parent;
        }

        if ModuleOrUniformRoot::same_def(ModuleOrUniformRoot::Module(crate_module), module) {
            // Don't make a suggestion if the import was already from the root of the
            // crate.
            return None;
        }

        let resolutions = crate_module.resolutions.borrow();
        let resolution = resolutions.get(&(ident, MacroNS))?;
        let binding = resolution.borrow().binding()?;
        if let Res::Def(DefKind::Macro(MacroKind::Bang), _) = binding.res() {
            let module_name = crate_module.kind.name().unwrap();
            let import = match directive.subclass {
                ImportDirectiveSubclass::SingleImport { source, target, .. } if source != target =>
                    format!("{} as {}", source, target),
                _ => format!("{}", ident),
            };

            let mut corrections: Vec<(Span, String)> = Vec::new();
            if !directive.is_nested() {
                // Assume this is the easy case of `use issue_59764::foo::makro;` and just remove
                // intermediate segments.
                corrections.push((directive.span, format!("{}::{}", module_name, import)));
            } else {
                // Find the binding span (and any trailing commas and spaces).
                //   ie. `use a::b::{c, d, e};`
                //                      ^^^
                let (found_closing_brace, binding_span) = find_span_of_binding_until_next_binding(
                    self.resolver.session, directive.span, directive.use_span,
                );
                debug!("check_for_module_export_macro: found_closing_brace={:?} binding_span={:?}",
                       found_closing_brace, binding_span);

                let mut removal_span = binding_span;
                if found_closing_brace {
                    // If the binding span ended with a closing brace, as in the below example:
                    //   ie. `use a::b::{c, d};`
                    //                      ^
                    // Then expand the span of characters to remove to include the previous
                    // binding's trailing comma.
                    //   ie. `use a::b::{c, d};`
                    //                    ^^^
                    if let Some(previous_span) = extend_span_to_previous_binding(
                        self.resolver.session, binding_span,
                    ) {
                        debug!("check_for_module_export_macro: previous_span={:?}", previous_span);
                        removal_span = removal_span.with_lo(previous_span.lo());
                    }
                }
                debug!("check_for_module_export_macro: removal_span={:?}", removal_span);

                // Remove the `removal_span`.
                corrections.push((removal_span, "".to_string()));

                // Find the span after the crate name and if it has nested imports immediatately
                // after the crate name already.
                //   ie. `use a::b::{c, d};`
                //               ^^^^^^^^^
                //   or  `use a::{b, c, d}};`
                //               ^^^^^^^^^^^
                let (has_nested, after_crate_name) = find_span_immediately_after_crate_name(
                    self.resolver.session, module_name, directive.use_span,
                );
                debug!("check_for_module_export_macro: has_nested={:?} after_crate_name={:?}",
                       has_nested, after_crate_name);

                let source_map = self.resolver.session.source_map();

                // Add the import to the start, with a `{` if required.
                let start_point = source_map.start_point(after_crate_name);
                if let Ok(start_snippet) = source_map.span_to_snippet(start_point) {
                    corrections.push((
                        start_point,
                        if has_nested {
                            // In this case, `start_snippet` must equal '{'.
                            format!("{}{}, ", start_snippet, import)
                        } else {
                            // In this case, add a `{`, then the moved import, then whatever
                            // was there before.
                            format!("{{{}, {}", import, start_snippet)
                        }
                    ));
                }

                // Add a `};` to the end if nested, matching the `{` added at the start.
                if !has_nested {
                    corrections.push((source_map.end_point(after_crate_name),
                                     "};".to_string()));
                }
            }

            let suggestion = Some((
                corrections,
                String::from("a macro with this name exists at the root of the crate"),
                Applicability::MaybeIncorrect,
            ));
            let note = vec![
                "this could be because a macro annotated with `#[macro_export]` will be exported \
                 at the root of the crate instead of the module where it is defined".to_string(),
            ];
            Some((suggestion, note))
        } else {
            None
        }
    }
}

/// Given a `binding_span` of a binding within a use statement:
///
/// ```
/// use foo::{a, b, c};
///              ^
/// ```
///
/// then return the span until the next binding or the end of the statement:
///
/// ```
/// use foo::{a, b, c};
///              ^^^
/// ```
pub(crate) fn find_span_of_binding_until_next_binding(
    sess: &Session,
    binding_span: Span,
    use_span: Span,
) -> (bool, Span) {
    let source_map = sess.source_map();

    // Find the span of everything after the binding.
    //   ie. `a, e};` or `a};`
    let binding_until_end = binding_span.with_hi(use_span.hi());

    // Find everything after the binding but not including the binding.
    //   ie. `, e};` or `};`
    let after_binding_until_end = binding_until_end.with_lo(binding_span.hi());

    // Keep characters in the span until we encounter something that isn't a comma or
    // whitespace.
    //   ie. `, ` or ``.
    //
    // Also note whether a closing brace character was encountered. If there
    // was, then later go backwards to remove any trailing commas that are left.
    let mut found_closing_brace = false;
    let after_binding_until_next_binding = source_map.span_take_while(
        after_binding_until_end,
        |&ch| {
            if ch == '}' { found_closing_brace = true; }
            ch == ' ' || ch == ','
        }
    );

    // Combine the two spans.
    //   ie. `a, ` or `a`.
    //
    // Removing these would leave `issue_52891::{d, e};` or `issue_52891::{d, e, };`
    let span = binding_span.with_hi(after_binding_until_next_binding.hi());

    (found_closing_brace, span)
}

/// Given a `binding_span`, return the span through to the comma or opening brace of the previous
/// binding.
///
/// ```
/// use foo::a::{a, b, c};
///               ^^--- binding span
///               |
///               returned span
///
/// use foo::{a, b, c};
///           --- binding span
/// ```
pub(crate) fn extend_span_to_previous_binding(
    sess: &Session,
    binding_span: Span,
) -> Option<Span> {
    let source_map = sess.source_map();

    // `prev_source` will contain all of the source that came before the span.
    // Then split based on a command and take the first (ie. closest to our span)
    // snippet. In the example, this is a space.
    let prev_source = source_map.span_to_prev_source(binding_span).ok()?;

    let prev_comma = prev_source.rsplit(',').collect::<Vec<_>>();
    let prev_starting_brace = prev_source.rsplit('{').collect::<Vec<_>>();
    if prev_comma.len() <= 1 || prev_starting_brace.len() <= 1 {
        return None;
    }

    let prev_comma = prev_comma.first().unwrap();
    let prev_starting_brace = prev_starting_brace.first().unwrap();

    // If the amount of source code before the comma is greater than
    // the amount of source code before the starting brace then we've only
    // got one item in the nested item (eg. `issue_52891::{self}`).
    if prev_comma.len() > prev_starting_brace.len() {
        return None;
    }

    Some(binding_span.with_lo(BytePos(
        // Take away the number of bytes for the characters we've found and an
        // extra for the comma.
        binding_span.lo().0 - (prev_comma.as_bytes().len() as u32) - 1
    )))
}

/// Given a `use_span` of a binding within a use statement, returns the highlighted span and if
/// it is a nested use tree.
///
/// ```
/// use foo::a::{b, c};
///          ^^^^^^^^^^ // false
///
/// use foo::{a, b, c};
///          ^^^^^^^^^^ // true
///
/// use foo::{a, b::{c, d}};
///          ^^^^^^^^^^^^^^^ // true
/// ```
fn find_span_immediately_after_crate_name(
    sess: &Session,
    module_name: Symbol,
    use_span: Span,
) -> (bool, Span) {
    debug!("find_span_immediately_after_crate_name: module_name={:?} use_span={:?}",
           module_name, use_span);
    let source_map = sess.source_map();

    // Using `use issue_59764::foo::{baz, makro};` as an example throughout..
    let mut num_colons = 0;
    // Find second colon.. `use issue_59764:`
    let until_second_colon = source_map.span_take_while(use_span, |c| {
        if *c == ':' { num_colons += 1; }
        match c {
            ':' if num_colons == 2 => false,
            _ => true,
        }
    });
    // Find everything after the second colon.. `foo::{baz, makro};`
    let from_second_colon = use_span.with_lo(until_second_colon.hi() + BytePos(1));

    let mut found_a_non_whitespace_character = false;
    // Find the first non-whitespace character in `from_second_colon`.. `f`
    let after_second_colon = source_map.span_take_while(from_second_colon, |c| {
        if found_a_non_whitespace_character { return false; }
        if !c.is_whitespace() { found_a_non_whitespace_character = true; }
        true
    });

    // Find the first `{` in from_second_colon.. `foo::{`
    let next_left_bracket = source_map.span_through_char(from_second_colon, '{');

    (next_left_bracket == after_second_colon, from_second_colon)
}
