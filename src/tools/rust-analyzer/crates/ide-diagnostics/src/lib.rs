//! Diagnostics rendering and fixits.
//!
//! Most of the diagnostics originate from the dark depth of the compiler, and
//! are originally expressed in term of IR. When we emit the diagnostic, we are
//! usually not in the position to decide how to best "render" it in terms of
//! user-authored source code. We are especially not in the position to offer
//! fixits, as the compiler completely lacks the infrastructure to edit the
//! source code.
//!
//! Instead, we "bubble up" raw, structured diagnostics until the `hir` crate,
//! where we "cook" them so that each diagnostic is formulated in terms of `hir`
//! types. Well, at least that's the aspiration, the "cooking" is somewhat
//! ad-hoc at the moment. Anyways, we get a bunch of ide-friendly diagnostic
//! structs from hir, and we want to render them to unified serializable
//! representation (span, level, message) here. If we can, we also provide
//! fixits. By the way, that's why we want to keep diagnostics structured
//! internally -- so that we have all the info to make fixes.
//!
//! We have one "handler" module per diagnostic code. Such a module contains
//! rendering, optional fixes and tests. It's OK if some low-level compiler
//! functionality ends up being tested via a diagnostic.
//!
//! There are also a couple of ad-hoc diagnostics implemented directly here, we
//! don't yet have a great pattern for how to do them properly.

mod handlers {
    pub(crate) mod await_outside_of_async;
    pub(crate) mod bad_rtn;
    pub(crate) mod break_outside_of_loop;
    pub(crate) mod elided_lifetimes_in_path;
    pub(crate) mod expected_function;
    pub(crate) mod generic_args_prohibited;
    pub(crate) mod inactive_code;
    pub(crate) mod incoherent_impl;
    pub(crate) mod incorrect_case;
    pub(crate) mod incorrect_generics_len;
    pub(crate) mod incorrect_generics_order;
    pub(crate) mod invalid_cast;
    pub(crate) mod invalid_derive_target;
    pub(crate) mod macro_error;
    pub(crate) mod malformed_derive;
    pub(crate) mod mismatched_arg_count;
    pub(crate) mod missing_fields;
    pub(crate) mod missing_lifetime;
    pub(crate) mod missing_match_arms;
    pub(crate) mod missing_unsafe;
    pub(crate) mod moved_out_of_ref;
    pub(crate) mod mutability_errors;
    pub(crate) mod no_such_field;
    pub(crate) mod non_exhaustive_let;
    pub(crate) mod parenthesized_generic_args_without_fn_trait;
    pub(crate) mod private_assoc_item;
    pub(crate) mod private_field;
    pub(crate) mod remove_trailing_return;
    pub(crate) mod remove_unnecessary_else;
    pub(crate) mod replace_filter_map_next_with_find_map;
    pub(crate) mod trait_impl_incorrect_safety;
    pub(crate) mod trait_impl_missing_assoc_item;
    pub(crate) mod trait_impl_orphan;
    pub(crate) mod trait_impl_redundant_assoc_item;
    pub(crate) mod type_mismatch;
    pub(crate) mod typed_hole;
    pub(crate) mod undeclared_label;
    pub(crate) mod unimplemented_builtin_macro;
    pub(crate) mod unreachable_label;
    pub(crate) mod unresolved_assoc_item;
    pub(crate) mod unresolved_extern_crate;
    pub(crate) mod unresolved_field;
    pub(crate) mod unresolved_ident;
    pub(crate) mod unresolved_import;
    pub(crate) mod unresolved_macro_call;
    pub(crate) mod unresolved_method;
    pub(crate) mod unresolved_module;
    pub(crate) mod unused_variables;

    // The handlers below are unusual, the implement the diagnostics as well.
    pub(crate) mod field_shorthand;
    pub(crate) mod json_is_not_rust;
    pub(crate) mod unlinked_file;
    pub(crate) mod useless_braces;
}

#[cfg(test)]
mod tests;

use std::{iter, sync::LazyLock};

use either::Either;
use hir::{
    Crate, DisplayTarget, InFile, Semantics, db::ExpandDatabase, diagnostics::AnyDiagnostic,
};
use ide_db::{
    EditionedFileId, FileId, FileRange, FxHashMap, FxHashSet, RootDatabase, Severity, SnippetCap,
    assists::{Assist, AssistId, AssistResolveStrategy, ExprFillDefaultMode},
    base_db::{ReleaseChannel, RootQueryDb as _},
    generated::lints::{CLIPPY_LINT_GROUPS, DEFAULT_LINT_GROUPS, DEFAULT_LINTS, Lint, LintGroup},
    imports::insert_use::InsertUseConfig,
    label::Label,
    source_change::SourceChange,
    syntax_helpers::node_ext::parse_tt_as_comma_sep_paths,
};
use itertools::Itertools;
use syntax::{
    AstPtr, Edition, NodeOrToken, SmolStr, SyntaxKind, SyntaxNode, SyntaxNodePtr, T, TextRange,
    ast::{self, AstNode, HasAttrs},
};

// FIXME: Make this an enum
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DiagnosticCode {
    RustcHardError(&'static str),
    SyntaxError,
    RustcLint(&'static str),
    Clippy(&'static str),
    Ra(&'static str, Severity),
}

impl DiagnosticCode {
    pub fn url(&self) -> String {
        match self {
            DiagnosticCode::RustcHardError(e) => {
                format!("https://doc.rust-lang.org/stable/error_codes/{e}.html")
            }
            DiagnosticCode::SyntaxError => {
                String::from("https://doc.rust-lang.org/stable/reference/")
            }
            DiagnosticCode::RustcLint(e) => {
                format!("https://doc.rust-lang.org/rustc/?search={e}")
            }
            DiagnosticCode::Clippy(e) => {
                format!("https://rust-lang.github.io/rust-clippy/master/#/{e}")
            }
            DiagnosticCode::Ra(e, _) => {
                format!("https://rust-analyzer.github.io/book/diagnostics.html#{e}")
            }
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DiagnosticCode::RustcHardError(r)
            | DiagnosticCode::RustcLint(r)
            | DiagnosticCode::Clippy(r)
            | DiagnosticCode::Ra(r, _) => r,
            DiagnosticCode::SyntaxError => "syntax-error",
        }
    }
}

#[derive(Debug)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub message: String,
    pub range: FileRange,
    pub severity: Severity,
    pub unused: bool,
    pub experimental: bool,
    pub fixes: Option<Vec<Assist>>,
    // The node that will be affected by `#[allow]` and similar attributes.
    pub main_node: Option<InFile<SyntaxNodePtr>>,
}

impl Diagnostic {
    fn new(
        code: DiagnosticCode,
        message: impl Into<String>,
        range: impl Into<FileRange>,
    ) -> Diagnostic {
        let message = message.into();
        Diagnostic {
            code,
            message,
            range: range.into(),
            severity: match code {
                DiagnosticCode::RustcHardError(_) | DiagnosticCode::SyntaxError => Severity::Error,
                // FIXME: Rustc lints are not always warning, but the ones that are currently implemented are all warnings.
                DiagnosticCode::RustcLint(_) => Severity::Warning,
                // FIXME: We can make this configurable, and if the user uses `cargo clippy` on flycheck, we can
                // make it normal warning.
                DiagnosticCode::Clippy(_) => Severity::WeakWarning,
                DiagnosticCode::Ra(_, s) => s,
            },
            unused: false,
            experimental: true,
            fixes: None,
            main_node: None,
        }
    }

    fn new_with_syntax_node_ptr(
        ctx: &DiagnosticsContext<'_>,
        code: DiagnosticCode,
        message: impl Into<String>,
        node: InFile<SyntaxNodePtr>,
    ) -> Diagnostic {
        Diagnostic::new(code, message, ctx.sema.diagnostics_display_range(node))
            .with_main_node(node)
    }

    fn stable(mut self) -> Diagnostic {
        self.experimental = false;
        self
    }

    fn with_main_node(mut self, main_node: InFile<SyntaxNodePtr>) -> Diagnostic {
        self.main_node = Some(main_node);
        self
    }

    fn with_fixes(mut self, fixes: Option<Vec<Assist>>) -> Diagnostic {
        self.fixes = fixes;
        self
    }

    fn with_unused(mut self, unused: bool) -> Diagnostic {
        self.unused = unused;
        self
    }
}

#[derive(Debug, Clone)]
pub struct DiagnosticsConfig {
    /// Whether native diagnostics are enabled.
    pub enabled: bool,
    pub proc_macros_enabled: bool,
    pub proc_attr_macros_enabled: bool,
    pub disable_experimental: bool,
    pub disabled: FxHashSet<String>,
    pub expr_fill_default: ExprFillDefaultMode,
    pub style_lints: bool,
    // FIXME: We may want to include a whole `AssistConfig` here
    pub snippet_cap: Option<SnippetCap>,
    pub insert_use: InsertUseConfig,
    pub prefer_no_std: bool,
    pub prefer_prelude: bool,
    pub prefer_absolute: bool,
    pub term_search_fuel: u64,
    pub term_search_borrowck: bool,
}

impl DiagnosticsConfig {
    pub fn test_sample() -> Self {
        use hir::PrefixKind;
        use ide_db::imports::insert_use::ImportGranularity;

        Self {
            enabled: true,
            proc_macros_enabled: Default::default(),
            proc_attr_macros_enabled: Default::default(),
            disable_experimental: Default::default(),
            disabled: Default::default(),
            expr_fill_default: Default::default(),
            style_lints: true,
            snippet_cap: SnippetCap::new(true),
            insert_use: InsertUseConfig {
                granularity: ImportGranularity::Preserve,
                enforce_granularity: false,
                prefix_kind: PrefixKind::Plain,
                group: false,
                skip_glob_imports: false,
            },
            prefer_no_std: false,
            prefer_prelude: true,
            prefer_absolute: false,
            term_search_fuel: 400,
            term_search_borrowck: true,
        }
    }
}

struct DiagnosticsContext<'a> {
    config: &'a DiagnosticsConfig,
    sema: Semantics<'a, RootDatabase>,
    resolve: &'a AssistResolveStrategy,
    edition: Edition,
    display_target: DisplayTarget,
    is_nightly: bool,
}

impl DiagnosticsContext<'_> {
    fn resolve_precise_location(
        &self,
        node: &InFile<SyntaxNodePtr>,
        precise_location: Option<TextRange>,
    ) -> FileRange {
        let sema = &self.sema;
        (|| {
            let precise_location = precise_location?;
            let root = sema.parse_or_expand(node.file_id);
            match root.covering_element(precise_location) {
                syntax::NodeOrToken::Node(it) => Some(sema.original_range(&it)),
                syntax::NodeOrToken::Token(it) => {
                    node.with_value(it).original_file_range_opt(sema.db)
                }
            }
        })()
        .map(|frange| ide_db::FileRange {
            file_id: frange.file_id.file_id(self.sema.db),
            range: frange.range,
        })
        .unwrap_or_else(|| sema.diagnostics_display_range(*node))
    }
}

/// Request parser level diagnostics for the given [`FileId`].
pub fn syntax_diagnostics(
    db: &RootDatabase,
    config: &DiagnosticsConfig,
    file_id: FileId,
) -> Vec<Diagnostic> {
    let _p = tracing::info_span!("syntax_diagnostics").entered();

    if config.disabled.contains("syntax-error") {
        return Vec::new();
    }

    let sema = Semantics::new(db);
    let editioned_file_id = sema
        .attach_first_edition(file_id)
        .unwrap_or_else(|| EditionedFileId::current_edition(db, file_id));

    let (file_id, _) = editioned_file_id.unpack(db);

    // [#3434] Only take first 128 errors to prevent slowing down editor/ide, the number 128 is chosen arbitrarily.
    db.parse_errors(editioned_file_id)
        .into_iter()
        .flatten()
        .take(128)
        .map(|err| {
            Diagnostic::new(
                DiagnosticCode::SyntaxError,
                format!("Syntax Error: {err}"),
                FileRange { file_id, range: err.range() },
            )
        })
        .collect()
}

/// Request semantic diagnostics for the given [`FileId`]. The produced diagnostics may point to other files
/// due to macros.
pub fn semantic_diagnostics(
    db: &RootDatabase,
    config: &DiagnosticsConfig,
    resolve: &AssistResolveStrategy,
    file_id: FileId,
) -> Vec<Diagnostic> {
    let _p = tracing::info_span!("semantic_diagnostics").entered();
    let sema = Semantics::new(db);
    let editioned_file_id = sema
        .attach_first_edition(file_id)
        .unwrap_or_else(|| EditionedFileId::current_edition(db, file_id));

    let (file_id, edition) = editioned_file_id.unpack(db);
    let mut res = Vec::new();

    let parse = sema.parse(editioned_file_id);

    // FIXME: This iterates the entire file which is a rather expensive operation.
    // We should implement these differently in some form?
    // Salsa caching + incremental re-parse would be better here
    for node in parse.syntax().descendants() {
        handlers::useless_braces::useless_braces(db, &mut res, editioned_file_id, &node);
        handlers::field_shorthand::field_shorthand(db, &mut res, editioned_file_id, &node);
        handlers::json_is_not_rust::json_in_items(
            &sema,
            &mut res,
            editioned_file_id,
            &node,
            config,
            edition,
        );
    }

    let module = sema.file_to_module_def(file_id);

    let is_nightly = matches!(
        module.and_then(|m| db.toolchain_channel(m.krate().into())),
        Some(ReleaseChannel::Nightly) | None
    );

    let krate = match module {
        Some(module) => module.krate(),
        None => {
            match db.all_crates().last() {
                Some(last) => (*last).into(),
                // short-circuit, return an empty vec of diagnostics
                None => return vec![],
            }
        }
    };
    let display_target = krate.to_display_target(db);
    let ctx = DiagnosticsContext { config, sema, resolve, edition, is_nightly, display_target };

    let mut diags = Vec::new();
    match module {
        // A bunch of parse errors in a file indicate some bigger structural parse changes in the
        // file, so we skip semantic diagnostics so we can show these faster.
        Some(m) => {
            if db.parse_errors(editioned_file_id).is_none_or(|es| es.len() < 16) {
                m.diagnostics(db, &mut diags, config.style_lints);
            }
        }
        None => {
            handlers::unlinked_file::unlinked_file(&ctx, &mut res, editioned_file_id.file_id(db))
        }
    }

    for diag in diags {
        let d = match diag {
            AnyDiagnostic::AwaitOutsideOfAsync(d) => handlers::await_outside_of_async::await_outside_of_async(&ctx, &d),
            AnyDiagnostic::CastToUnsized(d) => handlers::invalid_cast::cast_to_unsized(&ctx, &d),
            AnyDiagnostic::ExpectedFunction(d) => handlers::expected_function::expected_function(&ctx, &d),
            AnyDiagnostic::InactiveCode(d) => match handlers::inactive_code::inactive_code(&ctx, &d) {
                Some(it) => it,
                None => continue,
            }
            AnyDiagnostic::IncoherentImpl(d) => handlers::incoherent_impl::incoherent_impl(&ctx, &d),
            AnyDiagnostic::IncorrectCase(d) => handlers::incorrect_case::incorrect_case(&ctx, &d),
            AnyDiagnostic::InvalidCast(d) => handlers::invalid_cast::invalid_cast(&ctx, &d),
            AnyDiagnostic::InvalidDeriveTarget(d) => handlers::invalid_derive_target::invalid_derive_target(&ctx, &d),
            AnyDiagnostic::MacroDefError(d) => handlers::macro_error::macro_def_error(&ctx, &d),
            AnyDiagnostic::MacroError(d) => handlers::macro_error::macro_error(&ctx, &d),
            AnyDiagnostic::MacroExpansionParseError(d) => {
                // FIXME: Point to the correct error span here, not just the macro-call name
                res.extend(d.errors.iter().take(16).map(|err| {
                        Diagnostic::new(
                            DiagnosticCode::SyntaxError,
                            format!("Syntax Error in Expansion: {err}"),
                            ctx.resolve_precise_location(&d.node.clone(), d.precise_location),
                        )
                }));
                continue;
            },
            AnyDiagnostic::MalformedDerive(d) => handlers::malformed_derive::malformed_derive(&ctx, &d),
            AnyDiagnostic::MismatchedArgCount(d) => handlers::mismatched_arg_count::mismatched_arg_count(&ctx, &d),
            AnyDiagnostic::MissingFields(d) => handlers::missing_fields::missing_fields(&ctx, &d),
            AnyDiagnostic::MissingMatchArms(d) => handlers::missing_match_arms::missing_match_arms(&ctx, &d),
            AnyDiagnostic::MissingUnsafe(d) => handlers::missing_unsafe::missing_unsafe(&ctx, &d),
            AnyDiagnostic::MovedOutOfRef(d) => handlers::moved_out_of_ref::moved_out_of_ref(&ctx, &d),
            AnyDiagnostic::NeedMut(d) => match handlers::mutability_errors::need_mut(&ctx, &d) {
                Some(it) => it,
                None => continue,
            },
            AnyDiagnostic::NonExhaustiveLet(d) => handlers::non_exhaustive_let::non_exhaustive_let(&ctx, &d),
            AnyDiagnostic::NoSuchField(d) => handlers::no_such_field::no_such_field(&ctx, &d),
            AnyDiagnostic::PrivateAssocItem(d) => handlers::private_assoc_item::private_assoc_item(&ctx, &d),
            AnyDiagnostic::PrivateField(d) => handlers::private_field::private_field(&ctx, &d),
            AnyDiagnostic::ReplaceFilterMapNextWithFindMap(d) => handlers::replace_filter_map_next_with_find_map::replace_filter_map_next_with_find_map(&ctx, &d),
            AnyDiagnostic::TraitImplIncorrectSafety(d) => handlers::trait_impl_incorrect_safety::trait_impl_incorrect_safety(&ctx, &d),
            AnyDiagnostic::TraitImplMissingAssocItems(d) => handlers::trait_impl_missing_assoc_item::trait_impl_missing_assoc_item(&ctx, &d),
            AnyDiagnostic::TraitImplRedundantAssocItems(d) => handlers::trait_impl_redundant_assoc_item::trait_impl_redundant_assoc_item(&ctx, &d),
            AnyDiagnostic::TraitImplOrphan(d) => handlers::trait_impl_orphan::trait_impl_orphan(&ctx, &d),
            AnyDiagnostic::TypedHole(d) => handlers::typed_hole::typed_hole(&ctx, &d),
            AnyDiagnostic::TypeMismatch(d) => handlers::type_mismatch::type_mismatch(&ctx, &d),
            AnyDiagnostic::UndeclaredLabel(d) => handlers::undeclared_label::undeclared_label(&ctx, &d),
            AnyDiagnostic::UnimplementedBuiltinMacro(d) => handlers::unimplemented_builtin_macro::unimplemented_builtin_macro(&ctx, &d),
            AnyDiagnostic::UnreachableLabel(d) => handlers::unreachable_label::unreachable_label(&ctx, &d),
            AnyDiagnostic::UnresolvedAssocItem(d) => handlers::unresolved_assoc_item::unresolved_assoc_item(&ctx, &d),
            AnyDiagnostic::UnresolvedExternCrate(d) => handlers::unresolved_extern_crate::unresolved_extern_crate(&ctx, &d),
            AnyDiagnostic::UnresolvedField(d) => handlers::unresolved_field::unresolved_field(&ctx, &d),
            AnyDiagnostic::UnresolvedIdent(d) => handlers::unresolved_ident::unresolved_ident(&ctx, &d),
            AnyDiagnostic::UnresolvedImport(d) => handlers::unresolved_import::unresolved_import(&ctx, &d),
            AnyDiagnostic::UnresolvedMacroCall(d) => handlers::unresolved_macro_call::unresolved_macro_call(&ctx, &d),
            AnyDiagnostic::UnresolvedMethodCall(d) => handlers::unresolved_method::unresolved_method(&ctx, &d),
            AnyDiagnostic::UnresolvedModule(d) => handlers::unresolved_module::unresolved_module(&ctx, &d),
            AnyDiagnostic::UnusedMut(d) => match handlers::mutability_errors::unused_mut(&ctx, &d) {
                Some(it) => it,
                None => continue,
            },
            AnyDiagnostic::UnusedVariable(d) => match handlers::unused_variables::unused_variables(&ctx, &d) {
                Some(it) => it,
                None => continue,
            },
            AnyDiagnostic::BreakOutsideOfLoop(d) => handlers::break_outside_of_loop::break_outside_of_loop(&ctx, &d),
            AnyDiagnostic::MismatchedTupleStructPatArgCount(d) => handlers::mismatched_arg_count::mismatched_tuple_struct_pat_arg_count(&ctx, &d),
            AnyDiagnostic::RemoveTrailingReturn(d) => match handlers::remove_trailing_return::remove_trailing_return(&ctx, &d) {
                Some(it) => it,
                None => continue,
            },
            AnyDiagnostic::RemoveUnnecessaryElse(d) => match handlers::remove_unnecessary_else::remove_unnecessary_else(&ctx, &d) {
                Some(it) => it,
                None => continue,
            },
            AnyDiagnostic::GenericArgsProhibited(d) => handlers::generic_args_prohibited::generic_args_prohibited(&ctx, &d),
            AnyDiagnostic::ParenthesizedGenericArgsWithoutFnTrait(d) => handlers::parenthesized_generic_args_without_fn_trait::parenthesized_generic_args_without_fn_trait(&ctx, &d),
            AnyDiagnostic::BadRtn(d) => handlers::bad_rtn::bad_rtn(&ctx, &d),
            AnyDiagnostic::IncorrectGenericsLen(d) => handlers::incorrect_generics_len::incorrect_generics_len(&ctx, &d),
            AnyDiagnostic::IncorrectGenericsOrder(d) => handlers::incorrect_generics_order::incorrect_generics_order(&ctx, &d),
            AnyDiagnostic::MissingLifetime(d) => handlers::missing_lifetime::missing_lifetime(&ctx, &d),
            AnyDiagnostic::ElidedLifetimesInPath(d) => handlers::elided_lifetimes_in_path::elided_lifetimes_in_path(&ctx, &d),
        };
        res.push(d)
    }

    res.retain(|d| {
        !(ctx.config.disabled.contains(d.code.as_str())
            || ctx.config.disable_experimental && d.experimental)
    });

    let mut lints = res
        .iter_mut()
        .filter(|it| matches!(it.code, DiagnosticCode::Clippy(_) | DiagnosticCode::RustcLint(_)))
        .filter_map(|it| {
            Some((
                it.main_node.map(|ptr| {
                    ptr.map(|node| node.to_node(&ctx.sema.parse_or_expand(ptr.file_id)))
                })?,
                it,
            ))
        })
        .collect::<Vec<_>>();

    // The edition isn't accurate (each diagnostics may have its own edition due to macros),
    // but it's okay as it's only being used for error recovery.
    handle_lints(&ctx.sema, &mut lints, editioned_file_id.edition(db));

    res.retain(|d| d.severity != Severity::Allow);

    res.retain_mut(|diag| {
        if let Some(node) = diag
            .main_node
            .map(|ptr| ptr.map(|node| node.to_node(&ctx.sema.parse_or_expand(ptr.file_id))))
        {
            handle_diag_from_macros(&ctx.sema, diag, &node)
        } else {
            true
        }
    });

    res
}

/// Request both syntax and semantic diagnostics for the given [`FileId`].
pub fn full_diagnostics(
    db: &RootDatabase,
    config: &DiagnosticsConfig,
    resolve: &AssistResolveStrategy,
    file_id: FileId,
) -> Vec<Diagnostic> {
    let mut res = syntax_diagnostics(db, config, file_id);
    let sema = semantic_diagnostics(db, config, resolve, file_id);
    res.extend(sema);
    res
}

/// Returns whether to keep this diagnostic (or remove it).
fn handle_diag_from_macros(
    sema: &Semantics<'_, RootDatabase>,
    diag: &mut Diagnostic,
    node: &InFile<SyntaxNode>,
) -> bool {
    let Some(macro_file) = node.file_id.macro_file() else { return true };
    let span_map = sema.db.expansion_span_map(macro_file);
    let mut spans = span_map.spans_for_range(node.text_range());
    if spans.any(|span| {
        span.ctx.outer_expn(sema.db).is_some_and(|expansion| {
            let macro_call = sema.db.lookup_intern_macro_call(expansion.into());
            // We don't want to show diagnostics for non-local macros at all, but proc macros authors
            // seem to rely on being able to emit non-warning-free code, so we don't want to show warnings
            // for them even when the proc macro comes from the same workspace (in rustc that's not a
            // problem because it doesn't have the concept of workspaces, and proc macros always reside
            // in a different crate).
            !Crate::from(macro_call.def.krate).origin(sema.db).is_local()
                || !macro_call.def.kind.is_declarative()
        })
    }) {
        // Disable suggestions for external macros, they'll change library code and it's just bad.
        diag.fixes = None;

        // All Clippy lints report in macros, see https://github.com/rust-lang/rust-clippy/blob/903293b199364/declare_clippy_lint/src/lib.rs#L172.
        if let DiagnosticCode::RustcLint(lint) = diag.code
            && !LINTS_TO_REPORT_IN_EXTERNAL_MACROS.contains(lint)
        {
            return false;
        };
    }
    true
}

struct BuiltLint {
    lint: &'static Lint,
    groups: Vec<&'static str>,
}

static RUSTC_LINTS: LazyLock<FxHashMap<&str, BuiltLint>> =
    LazyLock::new(|| build_lints_map(DEFAULT_LINTS, DEFAULT_LINT_GROUPS, ""));

static CLIPPY_LINTS: LazyLock<FxHashMap<&str, BuiltLint>> = LazyLock::new(|| {
    build_lints_map(ide_db::generated::lints::CLIPPY_LINTS, CLIPPY_LINT_GROUPS, "clippy::")
});

// FIXME: Autogenerate this instead of enumerating by hand.
static LINTS_TO_REPORT_IN_EXTERNAL_MACROS: LazyLock<FxHashSet<&str>> =
    LazyLock::new(|| FxHashSet::from_iter([]));

fn build_lints_map(
    lints: &'static [Lint],
    lint_group: &'static [LintGroup],
    prefix: &'static str,
) -> FxHashMap<&'static str, BuiltLint> {
    let mut map_with_prefixes: FxHashMap<_, _> = lints
        .iter()
        .map(|lint| (lint.label, BuiltLint { lint, groups: vec![lint.label, "__RA_EVERY_LINT"] }))
        .collect();
    for g in lint_group {
        let mut add_children = |label: &'static str| {
            for child in g.children {
                map_with_prefixes.get_mut(child).unwrap().groups.push(label);
            }
        };
        add_children(g.lint.label);

        if g.lint.label == "nonstandard_style" {
            // Also add `bad_style`, which for some reason isn't listed in the groups.
            add_children("bad_style");
        }
    }
    map_with_prefixes.into_iter().map(|(k, v)| (k.strip_prefix(prefix).unwrap(), v)).collect()
}

fn handle_lints(
    sema: &Semantics<'_, RootDatabase>,
    diagnostics: &mut [(InFile<SyntaxNode>, &mut Diagnostic)],
    edition: Edition,
) {
    for (node, diag) in diagnostics {
        let lint = match diag.code {
            DiagnosticCode::RustcLint(lint) => RUSTC_LINTS[lint].lint,
            DiagnosticCode::Clippy(lint) => CLIPPY_LINTS[lint].lint,
            _ => panic!("non-lint passed to `handle_lints()`"),
        };
        let default_severity = default_lint_severity(lint, edition);
        if !(default_severity == Severity::Allow && diag.severity == Severity::WeakWarning) {
            diag.severity = default_severity;
        }

        let mut diag_severity =
            lint_severity_at(sema, node, &lint_groups(&diag.code, edition), edition);

        if let outline_diag_severity @ Some(_) =
            find_outline_mod_lint_severity(sema, node, diag, edition)
        {
            diag_severity = outline_diag_severity;
        }

        if let Some(diag_severity) = diag_severity {
            diag.severity = diag_severity;
        }
    }
}

fn default_lint_severity(lint: &Lint, edition: Edition) -> Severity {
    if lint.deny_since.is_some_and(|e| edition >= e) {
        Severity::Error
    } else if lint.warn_since.is_some_and(|e| edition >= e) {
        Severity::Warning
    } else {
        lint.default_severity
    }
}

fn find_outline_mod_lint_severity(
    sema: &Semantics<'_, RootDatabase>,
    node: &InFile<SyntaxNode>,
    diag: &Diagnostic,
    edition: Edition,
) -> Option<Severity> {
    let mod_node = node.value.ancestors().find_map(ast::Module::cast)?;
    if mod_node.item_list().is_some() {
        // Inline modules will be handled by `fill_lint_attrs()`.
        return None;
    }

    let mod_def = sema.to_module_def(&mod_node)?;
    let module_source_file = sema.module_definition_node(mod_def);
    let mut result = None;
    let lint_groups = lint_groups(&diag.code, edition);
    lint_attrs(
        sema,
        ast::AnyHasAttrs::cast(module_source_file.value).expect("SourceFile always has attrs"),
        edition,
    )
    .for_each(|(lint, severity)| {
        if lint_groups.contains(&lint) {
            result = Some(severity);
        }
    });
    result
}

fn lint_severity_at(
    sema: &Semantics<'_, RootDatabase>,
    node: &InFile<SyntaxNode>,
    lint_groups: &LintGroups,
    edition: Edition,
) -> Option<Severity> {
    node.value
        .ancestors()
        .filter_map(ast::AnyHasAttrs::cast)
        .find_map(|ancestor| {
            lint_attrs(sema, ancestor, edition)
                .find_map(|(lint, severity)| lint_groups.contains(&lint).then_some(severity))
        })
        .or_else(|| {
            lint_severity_at(sema, &sema.find_parent_file(node.file_id)?, lint_groups, edition)
        })
}

fn lint_attrs<'a>(
    sema: &'a Semantics<'a, RootDatabase>,
    ancestor: ast::AnyHasAttrs,
    edition: Edition,
) -> impl Iterator<Item = (SmolStr, Severity)> + 'a {
    ancestor
        .attrs_including_inner()
        .filter_map(|attr| {
            attr.as_simple_call().and_then(|(name, value)| match &*name {
                "allow" | "expect" => Some(Either::Left(iter::once((Severity::Allow, value)))),
                "warn" => Some(Either::Left(iter::once((Severity::Warning, value)))),
                "forbid" | "deny" => Some(Either::Left(iter::once((Severity::Error, value)))),
                "cfg_attr" => {
                    let mut lint_attrs = Vec::new();
                    cfg_attr_lint_attrs(sema, &value, &mut lint_attrs);
                    Some(Either::Right(lint_attrs.into_iter()))
                }
                _ => None,
            })
        })
        .flatten()
        .flat_map(move |(severity, lints)| {
            parse_tt_as_comma_sep_paths(lints, edition).into_iter().flat_map(move |lints| {
                // Rejoin the idents with `::`, so we have no spaces in between.
                lints.into_iter().map(move |lint| {
                    (
                        lint.segments().filter_map(|segment| segment.name_ref()).join("::").into(),
                        severity,
                    )
                })
            })
        })
}

fn cfg_attr_lint_attrs(
    sema: &Semantics<'_, RootDatabase>,
    value: &ast::TokenTree,
    lint_attrs: &mut Vec<(Severity, ast::TokenTree)>,
) {
    let prev_len = lint_attrs.len();

    let mut iter = value.token_trees_and_tokens().filter(|it| match it {
        NodeOrToken::Node(_) => true,
        NodeOrToken::Token(it) => !it.kind().is_trivia(),
    });

    // Skip the condition.
    for value in &mut iter {
        if value.as_token().is_some_and(|it| it.kind() == T![,]) {
            break;
        }
    }

    while let Some(value) = iter.next() {
        if let Some(token) = value.as_token()
            && token.kind() == SyntaxKind::IDENT
        {
            let severity = match token.text() {
                "allow" | "expect" => Some(Severity::Allow),
                "warn" => Some(Severity::Warning),
                "forbid" | "deny" => Some(Severity::Error),
                "cfg_attr" => {
                    if let Some(NodeOrToken::Node(value)) = iter.next() {
                        cfg_attr_lint_attrs(sema, &value, lint_attrs);
                    }
                    None
                }
                _ => None,
            };
            if let Some(severity) = severity {
                let lints = iter.next();
                if let Some(NodeOrToken::Node(lints)) = lints {
                    lint_attrs.push((severity, lints));
                }
            }
        }
    }

    if prev_len != lint_attrs.len()
        && let Some(false) | None = sema.check_cfg_attr(value)
    {
        // Discard the attributes when the condition is false.
        lint_attrs.truncate(prev_len);
    }
}

#[derive(Debug)]
struct LintGroups {
    groups: &'static [&'static str],
    inside_warnings: bool,
}

impl LintGroups {
    fn contains(&self, group: &str) -> bool {
        self.groups.contains(&group) || (self.inside_warnings && group == "warnings")
    }
}

fn lint_groups(lint: &DiagnosticCode, edition: Edition) -> LintGroups {
    let (groups, inside_warnings) = match lint {
        DiagnosticCode::RustcLint(name) => {
            let lint = &RUSTC_LINTS[name];
            let inside_warnings = default_lint_severity(lint.lint, edition) == Severity::Warning;
            (&lint.groups, inside_warnings)
        }
        DiagnosticCode::Clippy(name) => {
            let lint = &CLIPPY_LINTS[name];
            let inside_warnings = default_lint_severity(lint.lint, edition) == Severity::Warning;
            (&lint.groups, inside_warnings)
        }
        _ => panic!("non-lint passed to `handle_lints()`"),
    };
    LintGroups { groups, inside_warnings }
}

fn fix(id: &'static str, label: &str, source_change: SourceChange, target: TextRange) -> Assist {
    let mut res = unresolved_fix(id, label, target);
    res.source_change = Some(source_change);
    res
}

fn unresolved_fix(id: &'static str, label: &str, target: TextRange) -> Assist {
    assert!(!id.contains(' '));
    Assist {
        id: AssistId::quick_fix(id),
        label: Label::new(label.to_owned()),
        group: None,
        target,
        source_change: None,
        command: None,
    }
}

fn adjusted_display_range<N: AstNode>(
    ctx: &DiagnosticsContext<'_>,
    diag_ptr: InFile<AstPtr<N>>,
    adj: &dyn Fn(N) -> Option<TextRange>,
) -> FileRange {
    let source_file = ctx.sema.parse_or_expand(diag_ptr.file_id);
    let node = diag_ptr.value.to_node(&source_file);
    let hir::FileRange { file_id, range } = diag_ptr
        .with_value(adj(node).unwrap_or_else(|| diag_ptr.value.text_range()))
        .original_node_file_range_rooted(ctx.sema.db);
    ide_db::FileRange { file_id: file_id.file_id(ctx.sema.db), range }
}
