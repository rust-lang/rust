//! Basic types for managing and implementing lints.
//!
//! See <https://rustc-dev-guide.rust-lang.org/diagnostics.html> for an
//! overview of how lints are implemented.

use std::cell::Cell;
use std::slice;

use rustc_ast::BindingMode;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::sync;
use rustc_data_structures::unord::UnordMap;
use rustc_errors::{Diag, LintDiagnostic, MultiSpan};
use rustc_feature::Features;
use rustc_hir::def::Res;
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_hir::definitions::{DefPathData, DisambiguatedDefPathData};
use rustc_hir::{Pat, PatKind};
use rustc_middle::bug;
use rustc_middle::lint::LevelAndSource;
use rustc_middle::middle::privacy::EffectiveVisibilities;
use rustc_middle::ty::layout::{LayoutError, LayoutOfHelpers, TyAndLayout};
use rustc_middle::ty::print::{PrintError, PrintTraitRefExt as _, Printer, with_no_trimmed_paths};
use rustc_middle::ty::{self, GenericArg, RegisteredTools, Ty, TyCtxt, TypingEnv, TypingMode};
use rustc_session::lint::{FutureIncompatibleInfo, Lint, LintBuffer, LintExpectationId, LintId};
use rustc_session::{LintStoreMarker, Session};
use rustc_span::edit_distance::find_best_match_for_names;
use rustc_span::{Ident, Span, Symbol, sym};
use tracing::debug;
use {rustc_abi as abi, rustc_hir as hir};

use self::TargetLint::*;
use crate::levels::LintLevelsBuilder;
use crate::passes::{EarlyLintPassObject, LateLintPassObject};

type EarlyLintPassFactory = dyn Fn() -> EarlyLintPassObject + sync::DynSend + sync::DynSync;
type LateLintPassFactory =
    dyn for<'tcx> Fn(TyCtxt<'tcx>) -> LateLintPassObject<'tcx> + sync::DynSend + sync::DynSync;

/// Information about the registered lints.
pub struct LintStore {
    /// Registered lints.
    lints: Vec<&'static Lint>,

    /// Constructor functions for each variety of lint pass.
    ///
    /// These should only be called once, but since we want to avoid locks or
    /// interior mutability, we don't enforce this (and lints should, in theory,
    /// be compatible with being constructed more than once, though not
    /// necessarily in a sane manner. This is safe though.)
    pub pre_expansion_passes: Vec<Box<EarlyLintPassFactory>>,
    pub early_passes: Vec<Box<EarlyLintPassFactory>>,
    pub late_passes: Vec<Box<LateLintPassFactory>>,
    /// This is unique in that we construct them per-module, so not once.
    pub late_module_passes: Vec<Box<LateLintPassFactory>>,

    /// Lints indexed by name.
    by_name: UnordMap<String, TargetLint>,

    /// Map of registered lint groups to what lints they expand to.
    lint_groups: FxIndexMap<&'static str, LintGroup>,
}

impl LintStoreMarker for LintStore {}

/// The target of the `by_name` map, which accounts for renaming/deprecation.
#[derive(Debug)]
enum TargetLint {
    /// A direct lint target
    Id(LintId),

    /// Temporary renaming, used for easing migration pain; see #16545
    Renamed(String, LintId),

    /// Lint with this name existed previously, but has been removed/deprecated.
    /// The string argument is the reason for removal.
    Removed(String),

    /// A lint name that should give no warnings and have no effect.
    ///
    /// This is used by rustc to avoid warning about old rustdoc lints before rustdoc registers
    /// them as tool lints.
    Ignored,
}

pub enum FindLintError {
    NotFound,
    Removed,
}

struct LintAlias {
    name: &'static str,
    /// Whether deprecation warnings should be suppressed for this alias.
    silent: bool,
}

struct LintGroup {
    lint_ids: Vec<LintId>,
    is_externally_loaded: bool,
    depr: Option<LintAlias>,
}

#[derive(Debug)]
pub enum CheckLintNameResult<'a> {
    Ok(&'a [LintId]),
    /// Lint doesn't exist. Potentially contains a suggestion for a correct lint name.
    NoLint(Option<(Symbol, bool)>),
    /// The lint refers to a tool that has not been registered.
    NoTool,
    /// The lint has been renamed to a new name.
    Renamed(String),
    /// The lint has been removed due to the given reason.
    Removed(String),

    /// The lint is from a tool. The `LintId` will be returned as if it were a
    /// rustc lint. The `Option<String>` indicates if the lint has been
    /// renamed.
    Tool(&'a [LintId], Option<String>),

    /// The lint is from a tool. Either the lint does not exist in the tool or
    /// the code was not compiled with the tool and therefore the lint was
    /// never added to the `LintStore`.
    MissingTool,
}

impl LintStore {
    pub fn new() -> LintStore {
        LintStore {
            lints: vec![],
            pre_expansion_passes: vec![],
            early_passes: vec![],
            late_passes: vec![],
            late_module_passes: vec![],
            by_name: Default::default(),
            lint_groups: Default::default(),
        }
    }

    pub fn get_lints<'t>(&'t self) -> &'t [&'static Lint] {
        &self.lints
    }

    pub fn get_lint_groups(&self) -> impl Iterator<Item = (&'static str, Vec<LintId>, bool)> {
        self.lint_groups
            .iter()
            .filter(|(_, LintGroup { depr, .. })| {
                // Don't display deprecated lint groups.
                depr.is_none()
            })
            .map(|(k, LintGroup { lint_ids, is_externally_loaded, .. })| {
                (*k, lint_ids.clone(), *is_externally_loaded)
            })
    }

    pub fn register_early_pass(
        &mut self,
        pass: impl Fn() -> EarlyLintPassObject + 'static + sync::DynSend + sync::DynSync,
    ) {
        self.early_passes.push(Box::new(pass));
    }

    /// This lint pass is softly deprecated. It misses expanded code and has caused a few
    /// errors in the past. Currently, it is only used in Clippy. New implementations
    /// should avoid using this interface, as it might be removed in the future.
    ///
    /// * See [rust#69838](https://github.com/rust-lang/rust/pull/69838)
    /// * See [rust-clippy#5518](https://github.com/rust-lang/rust-clippy/pull/5518)
    pub fn register_pre_expansion_pass(
        &mut self,
        pass: impl Fn() -> EarlyLintPassObject + 'static + sync::DynSend + sync::DynSync,
    ) {
        self.pre_expansion_passes.push(Box::new(pass));
    }

    pub fn register_late_pass(
        &mut self,
        pass: impl for<'tcx> Fn(TyCtxt<'tcx>) -> LateLintPassObject<'tcx>
        + 'static
        + sync::DynSend
        + sync::DynSync,
    ) {
        self.late_passes.push(Box::new(pass));
    }

    pub fn register_late_mod_pass(
        &mut self,
        pass: impl for<'tcx> Fn(TyCtxt<'tcx>) -> LateLintPassObject<'tcx>
        + 'static
        + sync::DynSend
        + sync::DynSync,
    ) {
        self.late_module_passes.push(Box::new(pass));
    }

    /// Helper method for register_early/late_pass
    pub fn register_lints(&mut self, lints: &[&'static Lint]) {
        for lint in lints {
            self.lints.push(lint);

            let id = LintId::of(lint);
            if self.by_name.insert(lint.name_lower(), Id(id)).is_some() {
                bug!("duplicate specification of lint {}", lint.name_lower())
            }

            if let Some(FutureIncompatibleInfo { reason, .. }) = lint.future_incompatible {
                if let Some(edition) = reason.edition() {
                    self.lint_groups
                        .entry(edition.lint_name())
                        .or_insert(LintGroup {
                            lint_ids: vec![],
                            is_externally_loaded: lint.is_externally_loaded,
                            depr: None,
                        })
                        .lint_ids
                        .push(id);
                } else {
                    // Lints belonging to the `future_incompatible` lint group are lints where a
                    // future version of rustc will cause existing code to stop compiling.
                    // Lints tied to an edition don't count because they are opt-in.
                    self.lint_groups
                        .entry("future_incompatible")
                        .or_insert(LintGroup {
                            lint_ids: vec![],
                            is_externally_loaded: lint.is_externally_loaded,
                            depr: None,
                        })
                        .lint_ids
                        .push(id);
                }
            }
        }
    }

    pub fn register_group_alias(&mut self, lint_name: &'static str, alias: &'static str) {
        self.lint_groups.insert(
            alias,
            LintGroup {
                lint_ids: vec![],
                is_externally_loaded: false,
                depr: Some(LintAlias { name: lint_name, silent: true }),
            },
        );
    }

    pub fn register_group(
        &mut self,
        is_externally_loaded: bool,
        name: &'static str,
        deprecated_name: Option<&'static str>,
        to: Vec<LintId>,
    ) {
        let new = self
            .lint_groups
            .insert(name, LintGroup { lint_ids: to, is_externally_loaded, depr: None })
            .is_none();
        if let Some(deprecated) = deprecated_name {
            self.lint_groups.insert(
                deprecated,
                LintGroup {
                    lint_ids: vec![],
                    is_externally_loaded,
                    depr: Some(LintAlias { name, silent: false }),
                },
            );
        }

        if !new {
            bug!("duplicate specification of lint group {}", name);
        }
    }

    /// This lint should give no warning and have no effect.
    ///
    /// This is used by rustc to avoid warning about old rustdoc lints before rustdoc registers them as tool lints.
    #[track_caller]
    pub fn register_ignored(&mut self, name: &str) {
        if self.by_name.insert(name.to_string(), Ignored).is_some() {
            bug!("duplicate specification of lint {}", name);
        }
    }

    /// This lint has been renamed; warn about using the new name and apply the lint.
    #[track_caller]
    pub fn register_renamed(&mut self, old_name: &str, new_name: &str) {
        let Some(&Id(target)) = self.by_name.get(new_name) else {
            bug!("invalid lint renaming of {} to {}", old_name, new_name);
        };
        self.by_name.insert(old_name.to_string(), Renamed(new_name.to_string(), target));
    }

    pub fn register_removed(&mut self, name: &str, reason: &str) {
        self.by_name.insert(name.into(), Removed(reason.into()));
    }

    pub fn find_lints(&self, mut lint_name: &str) -> Result<Vec<LintId>, FindLintError> {
        match self.by_name.get(lint_name) {
            Some(&Id(lint_id)) => Ok(vec![lint_id]),
            Some(&Renamed(_, lint_id)) => Ok(vec![lint_id]),
            Some(&Removed(_)) => Err(FindLintError::Removed),
            Some(&Ignored) => Ok(vec![]),
            None => loop {
                return match self.lint_groups.get(lint_name) {
                    Some(LintGroup { lint_ids, depr, .. }) => {
                        if let Some(LintAlias { name, .. }) = depr {
                            lint_name = name;
                            continue;
                        }
                        Ok(lint_ids.clone())
                    }
                    None => Err(FindLintError::Removed),
                };
            },
        }
    }

    /// True if this symbol represents a lint group name.
    pub fn is_lint_group(&self, lint_name: Symbol) -> bool {
        debug!(
            "is_lint_group(lint_name={:?}, lint_groups={:?})",
            lint_name,
            self.lint_groups.keys().collect::<Vec<_>>()
        );
        let lint_name_str = lint_name.as_str();
        self.lint_groups.contains_key(lint_name_str) || {
            let warnings_name_str = crate::WARNINGS.name_lower();
            lint_name_str == warnings_name_str
        }
    }

    /// Checks the name of a lint for its existence, and whether it was
    /// renamed or removed. Generates a `Diag` containing a
    /// warning for renamed and removed lints. This is over both lint
    /// names from attributes and those passed on the command line. Since
    /// it emits non-fatal warnings and there are *two* lint passes that
    /// inspect attributes, this is only run from the late pass to avoid
    /// printing duplicate warnings.
    pub fn check_lint_name(
        &self,
        lint_name: &str,
        tool_name: Option<Symbol>,
        registered_tools: &RegisteredTools,
    ) -> CheckLintNameResult<'_> {
        if let Some(tool_name) = tool_name {
            // FIXME: rustc and rustdoc are considered tools for lints, but not for attributes.
            if tool_name != sym::rustc
                && tool_name != sym::rustdoc
                && !registered_tools.contains(&Ident::with_dummy_span(tool_name))
            {
                return CheckLintNameResult::NoTool;
            }
        }

        let complete_name = if let Some(tool_name) = tool_name {
            format!("{tool_name}::{lint_name}")
        } else {
            lint_name.to_string()
        };
        // If the lint was scoped with `tool::` check if the tool lint exists
        if let Some(tool_name) = tool_name {
            match self.by_name.get(&complete_name) {
                None => match self.lint_groups.get(&*complete_name) {
                    // If the lint isn't registered, there are two possibilities:
                    None => {
                        // 1. The tool is currently running, so this lint really doesn't exist.
                        // FIXME: should this handle tools that never register a lint, like rustfmt?
                        debug!("lints={:?}", self.by_name);
                        let tool_prefix = format!("{tool_name}::");

                        return if self.by_name.keys().any(|lint| lint.starts_with(&tool_prefix)) {
                            self.no_lint_suggestion(&complete_name, tool_name.as_str())
                        } else {
                            // 2. The tool isn't currently running, so no lints will be registered.
                            // To avoid giving a false positive, ignore all unknown lints.
                            CheckLintNameResult::MissingTool
                        };
                    }
                    Some(LintGroup { lint_ids, .. }) => {
                        return CheckLintNameResult::Tool(lint_ids, None);
                    }
                },
                Some(Id(id)) => return CheckLintNameResult::Tool(slice::from_ref(id), None),
                // If the lint was registered as removed or renamed by the lint tool, we don't need
                // to treat tool_lints and rustc lints different and can use the code below.
                _ => {}
            }
        }
        match self.by_name.get(&complete_name) {
            Some(Renamed(new_name, _)) => CheckLintNameResult::Renamed(new_name.to_string()),
            Some(Removed(reason)) => CheckLintNameResult::Removed(reason.to_string()),
            None => match self.lint_groups.get(&*complete_name) {
                // If neither the lint, nor the lint group exists check if there is a `clippy::`
                // variant of this lint
                None => self.check_tool_name_for_backwards_compat(&complete_name, "clippy"),
                Some(LintGroup { lint_ids, depr, .. }) => {
                    // Check if the lint group name is deprecated
                    if let Some(LintAlias { name, silent }) = depr {
                        let LintGroup { lint_ids, .. } = self.lint_groups.get(name).unwrap();
                        return if *silent {
                            CheckLintNameResult::Ok(lint_ids)
                        } else {
                            CheckLintNameResult::Tool(lint_ids, Some((*name).to_string()))
                        };
                    }
                    CheckLintNameResult::Ok(lint_ids)
                }
            },
            Some(Id(id)) => CheckLintNameResult::Ok(slice::from_ref(id)),
            Some(&Ignored) => CheckLintNameResult::Ok(&[]),
        }
    }

    fn no_lint_suggestion(&self, lint_name: &str, tool_name: &str) -> CheckLintNameResult<'_> {
        let name_lower = lint_name.to_lowercase();

        if lint_name.chars().any(char::is_uppercase) && self.find_lints(&name_lower).is_ok() {
            // First check if the lint name is (partly) in upper case instead of lower case...
            return CheckLintNameResult::NoLint(Some((Symbol::intern(&name_lower), false)));
        }

        // ...if not, search for lints with a similar name
        // Note: find_best_match_for_name depends on the sort order of its input vector.
        // To ensure deterministic output, sort elements of the lint_groups hash map.
        // Also, never suggest deprecated lint groups.
        // We will soon sort, so the initial order does not matter.
        #[allow(rustc::potential_query_instability)]
        let mut groups: Vec<_> = self
            .lint_groups
            .iter()
            .filter_map(|(k, LintGroup { depr, .. })| depr.is_none().then_some(k))
            .collect();
        groups.sort();
        let groups = groups.iter().map(|k| Symbol::intern(k));
        let lints = self.lints.iter().map(|l| Symbol::intern(&l.name_lower()));
        let names: Vec<Symbol> = groups.chain(lints).collect();
        let mut lookups = vec![Symbol::intern(&name_lower)];
        if let Some(stripped) = name_lower.split("::").last() {
            lookups.push(Symbol::intern(stripped));
        }
        let res = find_best_match_for_names(&names, &lookups, None);
        let is_rustc = res.map_or_else(
            || false,
            |s| name_lower.contains("::") && !s.as_str().starts_with(tool_name),
        );
        let suggestion = res.map(|s| (s, is_rustc));
        CheckLintNameResult::NoLint(suggestion)
    }

    fn check_tool_name_for_backwards_compat(
        &self,
        lint_name: &str,
        tool_name: &str,
    ) -> CheckLintNameResult<'_> {
        let complete_name = format!("{tool_name}::{lint_name}");
        match self.by_name.get(&complete_name) {
            None => match self.lint_groups.get(&*complete_name) {
                // Now we are sure, that this lint exists nowhere
                None => self.no_lint_suggestion(lint_name, tool_name),
                Some(LintGroup { lint_ids, depr, .. }) => {
                    // Reaching this would be weird, but let's cover this case anyway
                    if let Some(LintAlias { name, silent }) = depr {
                        let LintGroup { lint_ids, .. } = self.lint_groups.get(name).unwrap();
                        if *silent {
                            CheckLintNameResult::Tool(lint_ids, Some(complete_name))
                        } else {
                            CheckLintNameResult::Tool(lint_ids, Some((*name).to_string()))
                        }
                    } else {
                        CheckLintNameResult::Tool(lint_ids, Some(complete_name))
                    }
                }
            },
            Some(Id(id)) => CheckLintNameResult::Tool(slice::from_ref(id), Some(complete_name)),
            Some(other) => {
                debug!("got renamed lint {:?}", other);
                CheckLintNameResult::NoLint(None)
            }
        }
    }
}

/// Context for lint checking outside of type inference.
pub struct LateContext<'tcx> {
    /// Type context we're checking in.
    pub tcx: TyCtxt<'tcx>,

    /// Current body, or `None` if outside a body.
    pub enclosing_body: Option<hir::BodyId>,

    /// Type-checking results for the current body. Access using the `typeck_results`
    /// and `maybe_typeck_results` methods, which handle querying the typeck results on demand.
    // FIXME(eddyb) move all the code accessing internal fields like this,
    // to this module, to avoid exposing it to lint logic.
    pub(super) cached_typeck_results: Cell<Option<&'tcx ty::TypeckResults<'tcx>>>,

    /// Parameter environment for the item we are in.
    pub param_env: ty::ParamEnv<'tcx>,

    /// Items accessible from the crate being checked.
    pub effective_visibilities: &'tcx EffectiveVisibilities,

    pub last_node_with_lint_attrs: hir::HirId,

    /// Generic type parameters in scope for the item we are in.
    pub generics: Option<&'tcx hir::Generics<'tcx>>,

    /// We are only looking at one module
    pub only_module: bool,
}

/// Context for lint checking of the AST, after expansion, before lowering to HIR.
pub struct EarlyContext<'a> {
    pub builder: LintLevelsBuilder<'a, crate::levels::TopDown>,
    pub buffered: LintBuffer,
}

pub trait LintContext {
    fn sess(&self) -> &Session;

    // FIXME: These methods should not take an Into<MultiSpan> -- instead, callers should need to
    // set the span in their `decorate` function (preferably using set_span).
    /// Emit a lint at the appropriate level, with an optional associated span.
    ///
    /// [`lint_level`]: rustc_middle::lint::lint_level#decorate-signature
    #[rustc_lint_diagnostics]
    fn opt_span_lint<S: Into<MultiSpan>>(
        &self,
        lint: &'static Lint,
        span: Option<S>,
        decorate: impl for<'a, 'b> FnOnce(&'b mut Diag<'a, ()>),
    );

    /// Emit a lint at `span` from a lint struct (some type that implements `LintDiagnostic`,
    /// typically generated by `#[derive(LintDiagnostic)]`).
    fn emit_span_lint<S: Into<MultiSpan>>(
        &self,
        lint: &'static Lint,
        span: S,
        decorator: impl for<'a> LintDiagnostic<'a, ()>,
    ) {
        self.opt_span_lint(lint, Some(span), |lint| {
            decorator.decorate_lint(lint);
        });
    }

    /// Emit a lint at the appropriate level, with an associated span.
    ///
    /// [`lint_level`]: rustc_middle::lint::lint_level#decorate-signature
    #[rustc_lint_diagnostics]
    fn span_lint<S: Into<MultiSpan>>(
        &self,
        lint: &'static Lint,
        span: S,
        decorate: impl for<'a, 'b> FnOnce(&'b mut Diag<'a, ()>),
    ) {
        self.opt_span_lint(lint, Some(span), decorate);
    }

    /// Emit a lint from a lint struct (some type that implements `LintDiagnostic`, typically
    /// generated by `#[derive(LintDiagnostic)]`).
    fn emit_lint(&self, lint: &'static Lint, decorator: impl for<'a> LintDiagnostic<'a, ()>) {
        self.opt_span_lint(lint, None as Option<Span>, |lint| {
            decorator.decorate_lint(lint);
        });
    }

    /// Emit a lint at the appropriate level, with no associated span.
    ///
    /// [`lint_level`]: rustc_middle::lint::lint_level#decorate-signature
    #[rustc_lint_diagnostics]
    fn lint(&self, lint: &'static Lint, decorate: impl for<'a, 'b> FnOnce(&'b mut Diag<'a, ()>)) {
        self.opt_span_lint(lint, None as Option<Span>, decorate);
    }

    /// This returns the lint level for the given lint at the current location.
    fn get_lint_level(&self, lint: &'static Lint) -> LevelAndSource;

    /// This function can be used to manually fulfill an expectation. This can
    /// be used for lints which contain several spans, and should be suppressed,
    /// if either location was marked with an expectation.
    ///
    /// Note that this function should only be called for [`LintExpectationId`]s
    /// retrieved from the current lint pass. Buffered or manually created ids can
    /// cause ICEs.
    fn fulfill_expectation(&self, expectation: LintExpectationId) {
        // We need to make sure that submitted expectation ids are correctly fulfilled suppressed
        // and stored between compilation sessions. To not manually do these steps, we simply create
        // a dummy diagnostic and emit it as usual, which will be suppressed and stored like a
        // normal expected lint diagnostic.
        #[allow(rustc::diagnostic_outside_of_impl)]
        #[allow(rustc::untranslatable_diagnostic)]
        self.sess()
            .dcx()
            .struct_expect(
                "this is a dummy diagnostic, to submit and store an expectation",
                expectation,
            )
            .emit();
    }
}

impl<'a> EarlyContext<'a> {
    pub(crate) fn new(
        sess: &'a Session,
        features: &'a Features,
        lint_added_lints: bool,
        lint_store: &'a LintStore,
        registered_tools: &'a RegisteredTools,
        buffered: LintBuffer,
    ) -> EarlyContext<'a> {
        EarlyContext {
            builder: LintLevelsBuilder::new(
                sess,
                features,
                lint_added_lints,
                lint_store,
                registered_tools,
            ),
            buffered,
        }
    }
}

impl<'tcx> LintContext for LateContext<'tcx> {
    /// Gets the overall compiler `Session` object.
    fn sess(&self) -> &Session {
        self.tcx.sess
    }

    #[rustc_lint_diagnostics]
    fn opt_span_lint<S: Into<MultiSpan>>(
        &self,
        lint: &'static Lint,
        span: Option<S>,
        decorate: impl for<'a, 'b> FnOnce(&'b mut Diag<'a, ()>),
    ) {
        let hir_id = self.last_node_with_lint_attrs;

        match span {
            Some(s) => self.tcx.node_span_lint(lint, hir_id, s, decorate),
            None => self.tcx.node_lint(lint, hir_id, decorate),
        }
    }

    fn get_lint_level(&self, lint: &'static Lint) -> LevelAndSource {
        self.tcx.lint_level_at_node(lint, self.last_node_with_lint_attrs)
    }
}

impl LintContext for EarlyContext<'_> {
    /// Gets the overall compiler `Session` object.
    fn sess(&self) -> &Session {
        self.builder.sess()
    }

    #[rustc_lint_diagnostics]
    fn opt_span_lint<S: Into<MultiSpan>>(
        &self,
        lint: &'static Lint,
        span: Option<S>,
        decorate: impl for<'a, 'b> FnOnce(&'b mut Diag<'a, ()>),
    ) {
        self.builder.opt_span_lint(lint, span.map(|s| s.into()), decorate)
    }

    fn get_lint_level(&self, lint: &'static Lint) -> LevelAndSource {
        self.builder.lint_level(lint)
    }
}

impl<'tcx> LateContext<'tcx> {
    /// The typing mode of the currently visited node. Use this when
    /// building a new `InferCtxt`.
    pub fn typing_mode(&self) -> TypingMode<'tcx> {
        // FIXME(#132279): In case we're in a body, we should use a typing
        // mode which reveals the opaque types defined by that body.
        TypingMode::non_body_analysis()
    }

    pub fn typing_env(&self) -> TypingEnv<'tcx> {
        TypingEnv { typing_mode: self.typing_mode(), param_env: self.param_env }
    }

    pub fn type_is_copy_modulo_regions(&self, ty: Ty<'tcx>) -> bool {
        self.tcx.type_is_copy_modulo_regions(self.typing_env(), ty)
    }

    pub fn type_is_use_cloned_modulo_regions(&self, ty: Ty<'tcx>) -> bool {
        self.tcx.type_is_use_cloned_modulo_regions(self.typing_env(), ty)
    }

    /// Gets the type-checking results for the current body,
    /// or `None` if outside a body.
    pub fn maybe_typeck_results(&self) -> Option<&'tcx ty::TypeckResults<'tcx>> {
        self.cached_typeck_results.get().or_else(|| {
            self.enclosing_body.map(|body| {
                let typeck_results = self.tcx.typeck_body(body);
                self.cached_typeck_results.set(Some(typeck_results));
                typeck_results
            })
        })
    }

    /// Gets the type-checking results for the current body.
    /// As this will ICE if called outside bodies, only call when working with
    /// `Expr` or `Pat` nodes (they are guaranteed to be found only in bodies).
    #[track_caller]
    pub fn typeck_results(&self) -> &'tcx ty::TypeckResults<'tcx> {
        self.maybe_typeck_results().expect("`LateContext::typeck_results` called outside of body")
    }

    /// Returns the final resolution of a `QPath`, or `Res::Err` if unavailable.
    /// Unlike `.typeck_results().qpath_res(qpath, id)`, this can be used even outside
    /// bodies (e.g. for paths in `hir::Ty`), without any risk of ICE-ing.
    pub fn qpath_res(&self, qpath: &hir::QPath<'_>, id: hir::HirId) -> Res {
        match *qpath {
            hir::QPath::Resolved(_, path) => path.res,
            hir::QPath::TypeRelative(..) | hir::QPath::LangItem(..) => self
                .maybe_typeck_results()
                .filter(|typeck_results| typeck_results.hir_owner == id.owner)
                .or_else(|| {
                    self.tcx
                        .has_typeck_results(id.owner.def_id)
                        .then(|| self.tcx.typeck(id.owner.def_id))
                })
                .and_then(|typeck_results| typeck_results.type_dependent_def(id))
                .map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id)),
        }
    }

    /// Gets the absolute path of `def_id` as a vector of `Symbol`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore (no context or def id available)
    /// let def_path = cx.get_def_path(def_id);
    /// if let &[sym::core, sym::option, sym::Option] = &def_path[..] {
    ///     // The given `def_id` is that of an `Option` type
    /// }
    /// ```
    pub fn get_def_path(&self, def_id: DefId) -> Vec<Symbol> {
        struct AbsolutePathPrinter<'tcx> {
            tcx: TyCtxt<'tcx>,
            path: Vec<Symbol>,
        }

        impl<'tcx> Printer<'tcx> for AbsolutePathPrinter<'tcx> {
            fn tcx(&self) -> TyCtxt<'tcx> {
                self.tcx
            }

            fn print_region(&mut self, _region: ty::Region<'_>) -> Result<(), PrintError> {
                Ok(())
            }

            fn print_type(&mut self, _ty: Ty<'tcx>) -> Result<(), PrintError> {
                Ok(())
            }

            fn print_dyn_existential(
                &mut self,
                _predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
            ) -> Result<(), PrintError> {
                Ok(())
            }

            fn print_const(&mut self, _ct: ty::Const<'tcx>) -> Result<(), PrintError> {
                Ok(())
            }

            fn path_crate(&mut self, cnum: CrateNum) -> Result<(), PrintError> {
                self.path = vec![self.tcx.crate_name(cnum)];
                Ok(())
            }

            fn path_qualified(
                &mut self,
                self_ty: Ty<'tcx>,
                trait_ref: Option<ty::TraitRef<'tcx>>,
            ) -> Result<(), PrintError> {
                if trait_ref.is_none() {
                    if let ty::Adt(def, args) = self_ty.kind() {
                        return self.print_def_path(def.did(), args);
                    }
                }

                // This shouldn't ever be needed, but just in case:
                with_no_trimmed_paths!({
                    self.path = vec![match trait_ref {
                        Some(trait_ref) => Symbol::intern(&format!("{trait_ref:?}")),
                        None => Symbol::intern(&format!("<{self_ty}>")),
                    }];
                    Ok(())
                })
            }

            fn path_append_impl(
                &mut self,
                print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
                _disambiguated_data: &DisambiguatedDefPathData,
                self_ty: Ty<'tcx>,
                trait_ref: Option<ty::TraitRef<'tcx>>,
            ) -> Result<(), PrintError> {
                print_prefix(self)?;

                // This shouldn't ever be needed, but just in case:
                self.path.push(match trait_ref {
                    Some(trait_ref) => {
                        with_no_trimmed_paths!(Symbol::intern(&format!(
                            "<impl {} for {}>",
                            trait_ref.print_only_trait_path(),
                            self_ty
                        )))
                    }
                    None => {
                        with_no_trimmed_paths!(Symbol::intern(&format!("<impl {self_ty}>")))
                    }
                });

                Ok(())
            }

            fn path_append(
                &mut self,
                print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
                disambiguated_data: &DisambiguatedDefPathData,
            ) -> Result<(), PrintError> {
                print_prefix(self)?;

                // Skip `::{{extern}}` blocks and `::{{constructor}}` on tuple/unit structs.
                if let DefPathData::ForeignMod | DefPathData::Ctor = disambiguated_data.data {
                    return Ok(());
                }

                self.path.push(Symbol::intern(&disambiguated_data.data.to_string()));
                Ok(())
            }

            fn path_generic_args(
                &mut self,
                print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
                _args: &[GenericArg<'tcx>],
            ) -> Result<(), PrintError> {
                print_prefix(self)
            }
        }

        let mut printer = AbsolutePathPrinter { tcx: self.tcx, path: vec![] };
        printer.print_def_path(def_id, &[]).unwrap();
        printer.path
    }

    /// Returns the associated type `name` for `self_ty` as an implementation of `trait_id`.
    /// Do not invoke without first verifying that the type implements the trait.
    pub fn get_associated_type(
        &self,
        self_ty: Ty<'tcx>,
        trait_id: DefId,
        name: &str,
    ) -> Option<Ty<'tcx>> {
        let tcx = self.tcx;
        tcx.associated_items(trait_id)
            .find_by_name_and_kind(tcx, Ident::from_str(name), ty::AssocKind::Type, trait_id)
            .and_then(|assoc| {
                let proj = Ty::new_projection(tcx, assoc.def_id, [self_ty]);
                tcx.try_normalize_erasing_regions(self.typing_env(), proj).ok()
            })
    }

    /// If the given expression is a local binding, find the initializer expression.
    /// If that initializer expression is another local binding, find its initializer again.
    ///
    /// This process repeats as long as possible (but usually no more than once).
    /// Type-check adjustments are not taken in account in this function.
    ///
    /// Examples:
    /// ```
    /// let abc = 1;
    /// let def = abc + 2;
    /// //        ^^^^^^^ output
    /// let def = def;
    /// dbg!(def);
    /// //   ^^^ input
    /// ```
    pub fn expr_or_init<'a>(&self, mut expr: &'a hir::Expr<'tcx>) -> &'a hir::Expr<'tcx> {
        expr = expr.peel_blocks();

        while let hir::ExprKind::Path(ref qpath) = expr.kind
            && let Some(parent_node) = match self.qpath_res(qpath, expr.hir_id) {
                Res::Local(hir_id) => Some(self.tcx.parent_hir_node(hir_id)),
                _ => None,
            }
            && let Some(init) = match parent_node {
                hir::Node::Expr(expr) => Some(expr),
                hir::Node::LetStmt(hir::LetStmt {
                    init,
                    // Binding is immutable, init cannot be re-assigned
                    pat: Pat { kind: PatKind::Binding(BindingMode::NONE, ..), .. },
                    ..
                }) => *init,
                _ => None,
            }
        {
            expr = init.peel_blocks();
        }
        expr
    }

    /// If the given expression is a local binding, find the initializer expression.
    /// If that initializer expression is another local or **outside** (`const`/`static`)
    /// binding, find its initializer again.
    ///
    /// This process repeats as long as possible (but usually no more than once).
    /// Type-check adjustments are not taken in account in this function.
    ///
    /// Examples:
    /// ```
    /// const ABC: i32 = 1;
    /// //               ^ output
    /// let def = ABC;
    /// dbg!(def);
    /// //   ^^^ input
    ///
    /// // or...
    /// let abc = 1;
    /// let def = abc + 2;
    /// //        ^^^^^^^ output
    /// dbg!(def);
    /// //   ^^^ input
    /// ```
    pub fn expr_or_init_with_outside_body<'a>(
        &self,
        mut expr: &'a hir::Expr<'tcx>,
    ) -> &'a hir::Expr<'tcx> {
        expr = expr.peel_blocks();

        while let hir::ExprKind::Path(ref qpath) = expr.kind
            && let Some(parent_node) = match self.qpath_res(qpath, expr.hir_id) {
                Res::Local(hir_id) => Some(self.tcx.parent_hir_node(hir_id)),
                Res::Def(_, def_id) => self.tcx.hir_get_if_local(def_id),
                _ => None,
            }
            && let Some(init) = match parent_node {
                hir::Node::Expr(expr) => Some(expr),
                hir::Node::LetStmt(hir::LetStmt {
                    init,
                    // Binding is immutable, init cannot be re-assigned
                    pat: Pat { kind: PatKind::Binding(BindingMode::NONE, ..), .. },
                    ..
                }) => *init,
                hir::Node::Item(item) => match item.kind {
                    hir::ItemKind::Const(.., body_id) | hir::ItemKind::Static(.., body_id) => {
                        Some(self.tcx.hir_body(body_id).value)
                    }
                    _ => None,
                },
                _ => None,
            }
        {
            expr = init.peel_blocks();
        }
        expr
    }
}

impl<'tcx> abi::HasDataLayout for LateContext<'tcx> {
    #[inline]
    fn data_layout(&self) -> &abi::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'tcx> ty::layout::HasTyCtxt<'tcx> for LateContext<'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> ty::layout::HasTypingEnv<'tcx> for LateContext<'tcx> {
    #[inline]
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.typing_env()
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for LateContext<'tcx> {
    type LayoutOfResult = Result<TyAndLayout<'tcx>, LayoutError<'tcx>>;

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, _: Span, _: Ty<'tcx>) -> LayoutError<'tcx> {
        err
    }
}
