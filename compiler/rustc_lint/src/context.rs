//! Implementation of lint checking.
//!
//! The lint checking is mostly consolidated into one pass which runs
//! after all other analyses. Throughout compilation, lint warnings
//! can be added via the `add_lint` method on the Session structure. This
//! requires a span and an ID of the node that the lint is being added to. The
//! lint isn't actually emitted at that time because it is unknown what the
//! actual lint level at that location is.
//!
//! To actually emit lint warnings/errors, a separate pass is used.
//! A context keeps track of the current state of all lint levels.
//! Upon entering a node of the ast which can modify the lint settings, the
//! previous lint state is pushed onto a stack and the ast is then recursed
//! upon. As the ast is traversed, this keeps track of the current lint level
//! for all lint attributes.

use self::TargetLint::*;

use crate::errors::{
    CheckNameDeprecated, CheckNameUnknown, CheckNameUnknownTool, CheckNameWarning, RequestedLevel,
    UnsupportedGroup,
};
use crate::levels::LintLevelsBuilder;
use crate::passes::{EarlyLintPassObject, LateLintPassObject};
use rustc_ast::util::unicode::TEXT_FLOW_CONTROL_CHARS;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync;
use rustc_errors::add_elided_lifetime_in_path_suggestion;
use rustc_errors::{
    Applicability, DecorateLint, LintDiagnosticBuilder, MultiSpan, SuggestionStyle,
};
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_hir::definitions::{DefPathData, DisambiguatedDefPathData};
use rustc_middle::middle::privacy::AccessLevels;
use rustc_middle::middle::stability;
use rustc_middle::ty::layout::{LayoutError, LayoutOfHelpers, TyAndLayout};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, print::Printer, subst::GenericArg, RegisteredTools, Ty, TyCtxt};
use rustc_session::lint::{BuiltinLintDiagnostics, LintExpectationId};
use rustc_session::lint::{FutureIncompatibleInfo, Level, Lint, LintBuffer, LintId};
use rustc_session::Session;
use rustc_span::lev_distance::find_best_match_for_name;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::{BytePos, Span};
use rustc_target::abi;

use std::cell::Cell;
use std::iter;
use std::slice;

/// Information about the registered lints.
///
/// This is basically the subset of `Context` that we can
/// build early in the compile pipeline.
pub struct LintStore {
    /// Registered lints.
    lints: Vec<&'static Lint>,

    /// Constructor functions for each variety of lint pass.
    ///
    /// These should only be called once, but since we want to avoid locks or
    /// interior mutability, we don't enforce this (and lints should, in theory,
    /// be compatible with being constructed more than once, though not
    /// necessarily in a sane manner. This is safe though.)
    pub pre_expansion_passes: Vec<Box<dyn Fn() -> EarlyLintPassObject + sync::Send + sync::Sync>>,
    pub early_passes: Vec<Box<dyn Fn() -> EarlyLintPassObject + sync::Send + sync::Sync>>,
    pub late_passes: Vec<Box<dyn Fn() -> LateLintPassObject + sync::Send + sync::Sync>>,
    /// This is unique in that we construct them per-module, so not once.
    pub late_module_passes: Vec<Box<dyn Fn() -> LateLintPassObject + sync::Send + sync::Sync>>,

    /// Lints indexed by name.
    by_name: FxHashMap<String, TargetLint>,

    /// Map of registered lint groups to what lints they expand to.
    lint_groups: FxHashMap<&'static str, LintGroup>,
}

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
    /// This is used by rustc to avoid warning about old rustdoc lints before rustdoc registers them as tool lints.
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
    from_plugin: bool,
    depr: Option<LintAlias>,
}

#[derive(Debug)]
pub enum CheckLintNameResult<'a> {
    Ok(&'a [LintId]),
    /// Lint doesn't exist. Potentially contains a suggestion for a correct lint name.
    NoLint(Option<Symbol>),
    /// The lint refers to a tool that has not been registered.
    NoTool,
    /// The lint is either renamed or removed. This is the warning
    /// message, and an optional new name (`None` if removed).
    Warning(String, Option<String>),
    /// The lint is from a tool. If the Option is None, then either
    /// the lint does not exist in the tool or the code was not
    /// compiled with the tool and therefore the lint was never
    /// added to the `LintStore`. Otherwise the `LintId` will be
    /// returned as if it where a rustc lint.
    Tool(Result<&'a [LintId], (Option<&'a [LintId]>, String)>),
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

    pub fn get_lint_groups<'t>(
        &'t self,
    ) -> impl Iterator<Item = (&'static str, Vec<LintId>, bool)> + 't {
        // This function is not used in a way which observes the order of lints.
        #[allow(rustc::potential_query_instability)]
        self.lint_groups
            .iter()
            .filter(|(_, LintGroup { depr, .. })| {
                // Don't display deprecated lint groups.
                depr.is_none()
            })
            .map(|(k, LintGroup { lint_ids, from_plugin, .. })| {
                (*k, lint_ids.clone(), *from_plugin)
            })
    }

    pub fn register_early_pass(
        &mut self,
        pass: impl Fn() -> EarlyLintPassObject + 'static + sync::Send + sync::Sync,
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
        pass: impl Fn() -> EarlyLintPassObject + 'static + sync::Send + sync::Sync,
    ) {
        self.pre_expansion_passes.push(Box::new(pass));
    }

    pub fn register_late_pass(
        &mut self,
        pass: impl Fn() -> LateLintPassObject + 'static + sync::Send + sync::Sync,
    ) {
        self.late_passes.push(Box::new(pass));
    }

    pub fn register_late_mod_pass(
        &mut self,
        pass: impl Fn() -> LateLintPassObject + 'static + sync::Send + sync::Sync,
    ) {
        self.late_module_passes.push(Box::new(pass));
    }

    // Helper method for register_early/late_pass
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
                            from_plugin: lint.is_plugin,
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
                            from_plugin: lint.is_plugin,
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
                from_plugin: false,
                depr: Some(LintAlias { name: lint_name, silent: true }),
            },
        );
    }

    pub fn register_group(
        &mut self,
        from_plugin: bool,
        name: &'static str,
        deprecated_name: Option<&'static str>,
        to: Vec<LintId>,
    ) {
        let new = self
            .lint_groups
            .insert(name, LintGroup { lint_ids: to, from_plugin, depr: None })
            .is_none();
        if let Some(deprecated) = deprecated_name {
            self.lint_groups.insert(
                deprecated,
                LintGroup {
                    lint_ids: vec![],
                    from_plugin,
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

    /// Checks the validity of lint names derived from the command line.
    pub fn check_lint_name_cmdline(
        &self,
        sess: &Session,
        lint_name: &str,
        level: Level,
        registered_tools: &RegisteredTools,
    ) {
        let (tool_name, lint_name_only) = parse_lint_and_tool_name(lint_name);
        if lint_name_only == crate::WARNINGS.name_lower() && matches!(level, Level::ForceWarn(_)) {
            sess.emit_err(UnsupportedGroup { lint_group: crate::WARNINGS.name_lower() });
            return;
        }
        let lint_name = lint_name.to_string();
        match self.check_lint_name(lint_name_only, tool_name, registered_tools) {
            CheckLintNameResult::Warning(msg, _) => {
                sess.emit_warning(CheckNameWarning {
                    msg,
                    sub: RequestedLevel { level, lint_name },
                });
            }
            CheckLintNameResult::NoLint(suggestion) => {
                sess.emit_err(CheckNameUnknown {
                    lint_name: lint_name.clone(),
                    suggestion,
                    sub: RequestedLevel { level, lint_name },
                });
            }
            CheckLintNameResult::Tool(result) => {
                if let Err((Some(_), new_name)) = result {
                    sess.emit_warning(CheckNameDeprecated {
                        lint_name: lint_name.clone(),
                        new_name,
                        sub: RequestedLevel { level, lint_name },
                    });
                }
            }
            CheckLintNameResult::NoTool => {
                sess.emit_err(CheckNameUnknownTool {
                    tool_name: tool_name.unwrap(),
                    sub: RequestedLevel { level, lint_name },
                });
            }
            _ => {}
        };
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
    /// renamed or removed. Generates a DiagnosticBuilder containing a
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
            format!("{}::{}", tool_name, lint_name)
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
                        debug!("lints={:?}", self.by_name.keys().collect::<Vec<_>>());
                        let tool_prefix = format!("{}::", tool_name);
                        return if self.by_name.keys().any(|lint| lint.starts_with(&tool_prefix)) {
                            self.no_lint_suggestion(&complete_name)
                        } else {
                            // 2. The tool isn't currently running, so no lints will be registered.
                            // To avoid giving a false positive, ignore all unknown lints.
                            CheckLintNameResult::Tool(Err((None, String::new())))
                        };
                    }
                    Some(LintGroup { lint_ids, .. }) => {
                        return CheckLintNameResult::Tool(Ok(&lint_ids));
                    }
                },
                Some(&Id(ref id)) => return CheckLintNameResult::Tool(Ok(slice::from_ref(id))),
                // If the lint was registered as removed or renamed by the lint tool, we don't need
                // to treat tool_lints and rustc lints different and can use the code below.
                _ => {}
            }
        }
        match self.by_name.get(&complete_name) {
            Some(&Renamed(ref new_name, _)) => CheckLintNameResult::Warning(
                format!("lint `{}` has been renamed to `{}`", complete_name, new_name),
                Some(new_name.to_owned()),
            ),
            Some(&Removed(ref reason)) => CheckLintNameResult::Warning(
                format!("lint `{}` has been removed: {}", complete_name, reason),
                None,
            ),
            None => match self.lint_groups.get(&*complete_name) {
                // If neither the lint, nor the lint group exists check if there is a `clippy::`
                // variant of this lint
                None => self.check_tool_name_for_backwards_compat(&complete_name, "clippy"),
                Some(LintGroup { lint_ids, depr, .. }) => {
                    // Check if the lint group name is deprecated
                    if let Some(LintAlias { name, silent }) = depr {
                        let LintGroup { lint_ids, .. } = self.lint_groups.get(name).unwrap();
                        return if *silent {
                            CheckLintNameResult::Ok(&lint_ids)
                        } else {
                            CheckLintNameResult::Tool(Err((Some(&lint_ids), (*name).to_string())))
                        };
                    }
                    CheckLintNameResult::Ok(&lint_ids)
                }
            },
            Some(&Id(ref id)) => CheckLintNameResult::Ok(slice::from_ref(id)),
            Some(&Ignored) => CheckLintNameResult::Ok(&[]),
        }
    }

    fn no_lint_suggestion(&self, lint_name: &str) -> CheckLintNameResult<'_> {
        let name_lower = lint_name.to_lowercase();

        if lint_name.chars().any(char::is_uppercase) && self.find_lints(&name_lower).is_ok() {
            // First check if the lint name is (partly) in upper case instead of lower case...
            return CheckLintNameResult::NoLint(Some(Symbol::intern(&name_lower)));
        }
        // ...if not, search for lints with a similar name
        let groups = self.lint_groups.keys().copied().map(Symbol::intern);
        let lints = self.lints.iter().map(|l| Symbol::intern(&l.name_lower()));
        let names: Vec<Symbol> = groups.chain(lints).collect();
        let suggestion = find_best_match_for_name(&names, Symbol::intern(&name_lower), None);
        CheckLintNameResult::NoLint(suggestion)
    }

    fn check_tool_name_for_backwards_compat(
        &self,
        lint_name: &str,
        tool_name: &str,
    ) -> CheckLintNameResult<'_> {
        let complete_name = format!("{}::{}", tool_name, lint_name);
        match self.by_name.get(&complete_name) {
            None => match self.lint_groups.get(&*complete_name) {
                // Now we are sure, that this lint exists nowhere
                None => self.no_lint_suggestion(lint_name),
                Some(LintGroup { lint_ids, depr, .. }) => {
                    // Reaching this would be weird, but let's cover this case anyway
                    if let Some(LintAlias { name, silent }) = depr {
                        let LintGroup { lint_ids, .. } = self.lint_groups.get(name).unwrap();
                        return if *silent {
                            CheckLintNameResult::Tool(Err((Some(&lint_ids), complete_name)))
                        } else {
                            CheckLintNameResult::Tool(Err((Some(&lint_ids), (*name).to_string())))
                        };
                    }
                    CheckLintNameResult::Tool(Err((Some(&lint_ids), complete_name)))
                }
            },
            Some(&Id(ref id)) => {
                CheckLintNameResult::Tool(Err((Some(slice::from_ref(id)), complete_name)))
            }
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
    pub access_levels: &'tcx AccessLevels,

    /// The store of registered lints and the lint levels.
    pub lint_store: &'tcx LintStore,

    pub last_node_with_lint_attrs: hir::HirId,

    /// Generic type parameters in scope for the item we are in.
    pub generics: Option<&'tcx hir::Generics<'tcx>>,

    /// We are only looking at one module
    pub only_module: bool,
}

/// Context for lint checking of the AST, after expansion, before lowering to HIR.
pub struct EarlyContext<'a> {
    pub builder: LintLevelsBuilder<'a>,
    pub buffered: LintBuffer,
}

pub trait LintPassObject: Sized {}

impl LintPassObject for EarlyLintPassObject {}

impl LintPassObject for LateLintPassObject {}

pub trait LintContext: Sized {
    type PassObject: LintPassObject;

    fn sess(&self) -> &Session;
    fn lints(&self) -> &LintStore;

    fn lookup_with_diagnostics(
        &self,
        lint: &'static Lint,
        span: Option<impl Into<MultiSpan>>,
        decorate: impl for<'a> FnOnce(LintDiagnosticBuilder<'a, ()>),
        diagnostic: BuiltinLintDiagnostics,
    ) {
        self.lookup(lint, span, |lint| {
            // We first generate a blank diagnostic.
            let mut db = lint.build("");

            // Now, set up surrounding context.
            let sess = self.sess();
            match diagnostic {
                BuiltinLintDiagnostics::UnicodeTextFlow(span, content) => {
                    let spans: Vec<_> = content
                        .char_indices()
                        .filter_map(|(i, c)| {
                            TEXT_FLOW_CONTROL_CHARS.contains(&c).then(|| {
                                let lo = span.lo() + BytePos(2 + i as u32);
                                (c, span.with_lo(lo).with_hi(lo + BytePos(c.len_utf8() as u32)))
                            })
                        })
                        .collect();
                    let (an, s) = match spans.len() {
                        1 => ("an ", ""),
                        _ => ("", "s"),
                    };
                    db.span_label(span, &format!(
                        "this comment contains {}invisible unicode text flow control codepoint{}",
                        an,
                        s,
                    ));
                    for (c, span) in &spans {
                        db.span_label(*span, format!("{:?}", c));
                    }
                    db.note(
                        "these kind of unicode codepoints change the way text flows on \
                         applications that support them, but can cause confusion because they \
                         change the order of characters on the screen",
                    );
                    if !spans.is_empty() {
                        db.multipart_suggestion_with_style(
                            "if their presence wasn't intentional, you can remove them",
                            spans.into_iter().map(|(_, span)| (span, "".to_string())).collect(),
                            Applicability::MachineApplicable,
                            SuggestionStyle::HideCodeAlways,
                        );
                    }
                },
                BuiltinLintDiagnostics::Normal => (),
                BuiltinLintDiagnostics::AbsPathWithModule(span) => {
                    let (sugg, app) = match sess.source_map().span_to_snippet(span) {
                        Ok(ref s) => {
                            // FIXME(Manishearth) ideally the emitting code
                            // can tell us whether or not this is global
                            let opt_colon =
                                if s.trim_start().starts_with("::") { "" } else { "::" };

                            (format!("crate{}{}", opt_colon, s), Applicability::MachineApplicable)
                        }
                        Err(_) => ("crate::<path>".to_string(), Applicability::HasPlaceholders),
                    };
                    db.span_suggestion(span, "use `crate`", sugg, app);
                }
                BuiltinLintDiagnostics::ProcMacroDeriveResolutionFallback(span) => {
                    db.span_label(
                        span,
                        "names from parent modules are not accessible without an explicit import",
                    );
                }
                BuiltinLintDiagnostics::MacroExpandedMacroExportsAccessedByAbsolutePaths(
                    span_def,
                ) => {
                    db.span_note(span_def, "the macro is defined here");
                }
                BuiltinLintDiagnostics::ElidedLifetimesInPaths(
                    n,
                    path_span,
                    incl_angl_brckt,
                    insertion_span,
                ) => {
                    add_elided_lifetime_in_path_suggestion(
                        sess.source_map(),
                        &mut db,
                        n,
                        path_span,
                        incl_angl_brckt,
                        insertion_span,
                    );
                }
                BuiltinLintDiagnostics::UnknownCrateTypes(span, note, sugg) => {
                    db.span_suggestion(span, &note, sugg, Applicability::MaybeIncorrect);
                }
                BuiltinLintDiagnostics::UnusedImports(message, replaces, in_test_module) => {
                    if !replaces.is_empty() {
                        db.tool_only_multipart_suggestion(
                            &message,
                            replaces,
                            Applicability::MachineApplicable,
                        );
                    }

                    if let Some(span) = in_test_module {
                        db.span_help(
                            self.sess().source_map().guess_head_span(span),
                            "consider adding a `#[cfg(test)]` to the containing module",
                        );
                    }
                }
                BuiltinLintDiagnostics::RedundantImport(spans, ident) => {
                    for (span, is_imported) in spans {
                        let introduced = if is_imported { "imported" } else { "defined" };
                        db.span_label(
                            span,
                            format!("the item `{}` is already {} here", ident, introduced),
                        );
                    }
                }
                BuiltinLintDiagnostics::DeprecatedMacro(suggestion, span) => {
                    stability::deprecation_suggestion(&mut db, "macro", suggestion, span)
                }
                BuiltinLintDiagnostics::UnusedDocComment(span) => {
                    db.span_label(span, "rustdoc does not generate documentation for macro invocations");
                    db.help("to document an item produced by a macro, \
                                  the macro must produce the documentation as part of its expansion");
                }
                BuiltinLintDiagnostics::PatternsInFnsWithoutBody(span, ident) => {
                    db.span_suggestion(span, "remove `mut` from the parameter", ident, Applicability::MachineApplicable);
                }
                BuiltinLintDiagnostics::MissingAbi(span, default_abi) => {
                    db.span_label(span, "ABI should be specified here");
                    db.help(&format!("the default ABI is {}", default_abi.name()));
                }
                BuiltinLintDiagnostics::LegacyDeriveHelpers(span) => {
                    db.span_label(span, "the attribute is introduced here");
                }
                BuiltinLintDiagnostics::ProcMacroBackCompat(note) => {
                    db.note(&note);
                }
                BuiltinLintDiagnostics::OrPatternsBackCompat(span,suggestion) => {
                    db.span_suggestion(span, "use pat_param to preserve semantics", suggestion, Applicability::MachineApplicable);
                }
                BuiltinLintDiagnostics::ReservedPrefix(span) => {
                    db.span_label(span, "unknown prefix");
                    db.span_suggestion_verbose(
                        span.shrink_to_hi(),
                        "insert whitespace here to avoid this being parsed as a prefix in Rust 2021",
                        " ",
                        Applicability::MachineApplicable,
                    );
                }
                BuiltinLintDiagnostics::UnusedBuiltinAttribute {
                    attr_name,
                    macro_name,
                    invoc_span
                } => {
                    db.span_note(
                        invoc_span,
                        &format!("the built-in attribute `{attr_name}` will be ignored, since it's applied to the macro invocation `{macro_name}`")
                    );
                }
                BuiltinLintDiagnostics::TrailingMacro(is_trailing, name) => {
                    if is_trailing {
                        db.note("macro invocations at the end of a block are treated as expressions");
                        db.note(&format!("to ignore the value produced by the macro, add a semicolon after the invocation of `{name}`"));
                    }
                }
                BuiltinLintDiagnostics::BreakWithLabelAndLoop(span) => {
                    db.multipart_suggestion(
                        "wrap this expression in parentheses",
                        vec![(span.shrink_to_lo(), "(".to_string()),
                             (span.shrink_to_hi(), ")".to_string())],
                        Applicability::MachineApplicable
                    );
                }
                BuiltinLintDiagnostics::NamedAsmLabel(help) => {
                    db.help(&help);
                    db.note("see the asm section of Rust By Example <https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html#labels> for more information");
                },
                BuiltinLintDiagnostics::UnexpectedCfg((name, name_span), None) => {
                    let Some(names_valid) = &sess.parse_sess.check_config.names_valid else {
                        bug!("it shouldn't be possible to have a diagnostic on a name if name checking is not enabled");
                    };
                    let possibilities: Vec<Symbol> = names_valid.iter().map(|s| *s).collect();

                    // Suggest the most probable if we found one
                    if let Some(best_match) = find_best_match_for_name(&possibilities, name, None) {
                        db.span_suggestion(name_span, "did you mean", best_match, Applicability::MaybeIncorrect);
                    }
                },
                BuiltinLintDiagnostics::UnexpectedCfg((name, name_span), Some((value, value_span))) => {
                    let Some(values) = &sess.parse_sess.check_config.values_valid.get(&name) else {
                        bug!("it shouldn't be possible to have a diagnostic on a value whose name is not in values");
                    };
                    let possibilities: Vec<Symbol> = values.iter().map(|&s| s).collect();

                    // Show the full list if all possible values for a given name, but don't do it
                    // for names as the possibilities could be very long
                    if !possibilities.is_empty() {
                        {
                            let mut possibilities = possibilities.iter().map(Symbol::as_str).collect::<Vec<_>>();
                            possibilities.sort();

                            let possibilities = possibilities.join(", ");
                            db.note(&format!("expected values for `{name}` are: {possibilities}"));
                        }

                        // Suggest the most probable if we found one
                        if let Some(best_match) = find_best_match_for_name(&possibilities, value, None) {
                            db.span_suggestion(value_span, "did you mean", format!("\"{best_match}\""), Applicability::MaybeIncorrect);
                        }
                    } else {
                        db.note(&format!("no expected value for `{name}`"));
                        if name != sym::feature {
                            db.span_suggestion(name_span.shrink_to_hi().to(value_span), "remove the value", "", Applicability::MaybeIncorrect);
                        }
                    }
                },
                BuiltinLintDiagnostics::DeprecatedWhereclauseLocation(new_span, suggestion) => {
                    db.multipart_suggestion(
                        "move it to the end of the type declaration",
                        vec![(db.span.primary_span().unwrap(), "".to_string()), (new_span, suggestion)],
                        Applicability::MachineApplicable,
                    );
                    db.note(
                        "see issue #89122 <https://github.com/rust-lang/rust/issues/89122> for more information",
                    );
                },
                BuiltinLintDiagnostics::SingleUseLifetime {
                    param_span,
                    use_span: Some((use_span, elide)),
                    deletion_span,
                } => {
                    debug!(?param_span, ?use_span, ?deletion_span);
                    db.span_label(param_span, "this lifetime...");
                    db.span_label(use_span, "...is used only here");
                    let msg = "elide the single-use lifetime";
                    let (use_span, replace_lt) = if elide {
                        let use_span = sess.source_map().span_extend_while(
                            use_span,
                            char::is_whitespace,
                        ).unwrap_or(use_span);
                        (use_span, String::new())
                    } else {
                        (use_span, "'_".to_owned())
                    };
                    db.multipart_suggestion(
                        msg,
                        vec![(deletion_span, String::new()), (use_span, replace_lt)],
                        Applicability::MachineApplicable,
                    );
                },
                BuiltinLintDiagnostics::SingleUseLifetime {
                    param_span: _,
                    use_span: None,
                    deletion_span,
                } => {
                    debug!(?deletion_span);
                    db.span_suggestion(
                        deletion_span,
                        "elide the unused lifetime",
                        "",
                        Applicability::MachineApplicable,
                    );
                },
                BuiltinLintDiagnostics::NamedArgumentUsedPositionally{ position_sp_to_replace, position_sp_for_msg, named_arg_sp, named_arg_name, is_formatting_arg} => {
                    db.span_label(named_arg_sp, "this named argument is referred to by position in formatting string");
                    if let Some(positional_arg_for_msg) = position_sp_for_msg {
                        let msg = format!("this formatting argument uses named argument `{}` by position", named_arg_name);
                        db.span_label(positional_arg_for_msg, msg);
                    }

                    if let Some(positional_arg_to_replace) = position_sp_to_replace {
                        let name = if is_formatting_arg { named_arg_name + "$" } else { named_arg_name };
                        let span_to_replace = if let Ok(positional_arg_content) =
                            self.sess().source_map().span_to_snippet(positional_arg_to_replace) && positional_arg_content.starts_with(':') {
                            positional_arg_to_replace.shrink_to_lo()
                        } else {
                            positional_arg_to_replace
                        };
                        db.span_suggestion_verbose(
                            span_to_replace,
                            "use the named argument by name to avoid ambiguity",
                            name,
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }
            // Rewrap `db`, and pass control to the user.
            decorate(LintDiagnosticBuilder::new(db));
        });
    }

    // FIXME: These methods should not take an Into<MultiSpan> -- instead, callers should need to
    // set the span in their `decorate` function (preferably using set_span).
    fn lookup<S: Into<MultiSpan>>(
        &self,
        lint: &'static Lint,
        span: Option<S>,
        decorate: impl for<'a> FnOnce(LintDiagnosticBuilder<'a, ()>),
    );

    /// Emit a lint at `span` from a lint struct (some type that implements `DecorateLint`,
    /// typically generated by `#[derive(LintDiagnostic)]`).
    fn emit_spanned_lint<S: Into<MultiSpan>>(
        &self,
        lint: &'static Lint,
        span: S,
        decorator: impl for<'a> DecorateLint<'a, ()>,
    ) {
        self.lookup(lint, Some(span), |diag| decorator.decorate_lint(diag));
    }

    fn struct_span_lint<S: Into<MultiSpan>>(
        &self,
        lint: &'static Lint,
        span: S,
        decorate: impl for<'a> FnOnce(LintDiagnosticBuilder<'a, ()>),
    ) {
        self.lookup(lint, Some(span), decorate);
    }

    /// Emit a lint from a lint struct (some type that implements `DecorateLint`, typically
    /// generated by `#[derive(LintDiagnostic)]`).
    fn emit_lint(&self, lint: &'static Lint, decorator: impl for<'a> DecorateLint<'a, ()>) {
        self.lookup(lint, None as Option<Span>, |diag| decorator.decorate_lint(diag));
    }

    /// Emit a lint at the appropriate level, with no associated span.
    fn lint(
        &self,
        lint: &'static Lint,
        decorate: impl for<'a> FnOnce(LintDiagnosticBuilder<'a, ()>),
    ) {
        self.lookup(lint, None as Option<Span>, decorate);
    }

    /// This returns the lint level for the given lint at the current location.
    fn get_lint_level(&self, lint: &'static Lint) -> Level;

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
        // a dummy diagnostic and emit is as usual, which will be suppressed and stored like a normal
        // expected lint diagnostic.
        self.sess()
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
        warn_about_weird_lints: bool,
        lint_store: &'a LintStore,
        registered_tools: &'a RegisteredTools,
        buffered: LintBuffer,
    ) -> EarlyContext<'a> {
        EarlyContext {
            builder: LintLevelsBuilder::new(
                sess,
                warn_about_weird_lints,
                lint_store,
                registered_tools,
            ),
            buffered,
        }
    }
}

impl LintContext for LateContext<'_> {
    type PassObject = LateLintPassObject;

    /// Gets the overall compiler `Session` object.
    fn sess(&self) -> &Session {
        &self.tcx.sess
    }

    fn lints(&self) -> &LintStore {
        &*self.lint_store
    }

    fn lookup<S: Into<MultiSpan>>(
        &self,
        lint: &'static Lint,
        span: Option<S>,
        decorate: impl for<'a> FnOnce(LintDiagnosticBuilder<'a, ()>),
    ) {
        let hir_id = self.last_node_with_lint_attrs;

        match span {
            Some(s) => self.tcx.struct_span_lint_hir(lint, hir_id, s, decorate),
            None => self.tcx.struct_lint_node(lint, hir_id, decorate),
        }
    }

    fn get_lint_level(&self, lint: &'static Lint) -> Level {
        self.tcx.lint_level_at_node(lint, self.last_node_with_lint_attrs).0
    }
}

impl LintContext for EarlyContext<'_> {
    type PassObject = EarlyLintPassObject;

    /// Gets the overall compiler `Session` object.
    fn sess(&self) -> &Session {
        &self.builder.sess()
    }

    fn lints(&self) -> &LintStore {
        self.builder.lint_store()
    }

    fn lookup<S: Into<MultiSpan>>(
        &self,
        lint: &'static Lint,
        span: Option<S>,
        decorate: impl for<'a> FnOnce(LintDiagnosticBuilder<'a, ()>),
    ) {
        self.builder.struct_lint(lint, span.map(|s| s.into()), decorate)
    }

    fn get_lint_level(&self, lint: &'static Lint) -> Level {
        self.builder.lint_level(lint).0
    }
}

impl<'tcx> LateContext<'tcx> {
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
            hir::QPath::Resolved(_, ref path) => path.res,
            hir::QPath::TypeRelative(..) | hir::QPath::LangItem(..) => self
                .maybe_typeck_results()
                .filter(|typeck_results| typeck_results.hir_owner == id.owner)
                .or_else(|| {
                    if self.tcx.has_typeck_results(id.owner.to_def_id()) {
                        Some(self.tcx.typeck(id.owner))
                    } else {
                        None
                    }
                })
                .and_then(|typeck_results| typeck_results.type_dependent_def(id))
                .map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id)),
        }
    }

    /// Check if a `DefId`'s path matches the given absolute type path usage.
    ///
    /// Anonymous scopes such as `extern` imports are matched with `kw::Empty`;
    /// inherent `impl` blocks are matched with the name of the type.
    ///
    /// Instead of using this method, it is often preferable to instead use
    /// `rustc_diagnostic_item` or a `lang_item`. This is less prone to errors
    /// as paths get invalidated if the target definition moves.
    ///
    /// # Examples
    ///
    /// ```rust,ignore (no context or def id available)
    /// if cx.match_def_path(def_id, &[sym::core, sym::option, sym::Option]) {
    ///     // The given `def_id` is that of an `Option` type
    /// }
    /// ```
    ///
    /// Used by clippy, but should be replaced by diagnostic items eventually.
    pub fn match_def_path(&self, def_id: DefId, path: &[Symbol]) -> bool {
        let names = self.get_def_path(def_id);

        names.len() == path.len() && iter::zip(names, path).all(|(a, &b)| a == b)
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
        pub struct AbsolutePathPrinter<'tcx> {
            pub tcx: TyCtxt<'tcx>,
        }

        impl<'tcx> Printer<'tcx> for AbsolutePathPrinter<'tcx> {
            type Error = !;

            type Path = Vec<Symbol>;
            type Region = ();
            type Type = ();
            type DynExistential = ();
            type Const = ();

            fn tcx(&self) -> TyCtxt<'tcx> {
                self.tcx
            }

            fn print_region(self, _region: ty::Region<'_>) -> Result<Self::Region, Self::Error> {
                Ok(())
            }

            fn print_type(self, _ty: Ty<'tcx>) -> Result<Self::Type, Self::Error> {
                Ok(())
            }

            fn print_dyn_existential(
                self,
                _predicates: &'tcx ty::List<ty::Binder<'tcx, ty::ExistentialPredicate<'tcx>>>,
            ) -> Result<Self::DynExistential, Self::Error> {
                Ok(())
            }

            fn print_const(self, _ct: ty::Const<'tcx>) -> Result<Self::Const, Self::Error> {
                Ok(())
            }

            fn path_crate(self, cnum: CrateNum) -> Result<Self::Path, Self::Error> {
                Ok(vec![self.tcx.crate_name(cnum)])
            }

            fn path_qualified(
                self,
                self_ty: Ty<'tcx>,
                trait_ref: Option<ty::TraitRef<'tcx>>,
            ) -> Result<Self::Path, Self::Error> {
                if trait_ref.is_none() {
                    if let ty::Adt(def, substs) = self_ty.kind() {
                        return self.print_def_path(def.did(), substs);
                    }
                }

                // This shouldn't ever be needed, but just in case:
                with_no_trimmed_paths!({
                    Ok(vec![match trait_ref {
                        Some(trait_ref) => Symbol::intern(&format!("{:?}", trait_ref)),
                        None => Symbol::intern(&format!("<{}>", self_ty)),
                    }])
                })
            }

            fn path_append_impl(
                self,
                print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
                _disambiguated_data: &DisambiguatedDefPathData,
                self_ty: Ty<'tcx>,
                trait_ref: Option<ty::TraitRef<'tcx>>,
            ) -> Result<Self::Path, Self::Error> {
                let mut path = print_prefix(self)?;

                // This shouldn't ever be needed, but just in case:
                path.push(match trait_ref {
                    Some(trait_ref) => {
                        with_no_trimmed_paths!(Symbol::intern(&format!(
                            "<impl {} for {}>",
                            trait_ref.print_only_trait_path(),
                            self_ty
                        )))
                    }
                    None => {
                        with_no_trimmed_paths!(Symbol::intern(&format!("<impl {}>", self_ty)))
                    }
                });

                Ok(path)
            }

            fn path_append(
                self,
                print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
                disambiguated_data: &DisambiguatedDefPathData,
            ) -> Result<Self::Path, Self::Error> {
                let mut path = print_prefix(self)?;

                // Skip `::{{extern}}` blocks and `::{{constructor}}` on tuple/unit structs.
                if let DefPathData::ForeignMod | DefPathData::Ctor = disambiguated_data.data {
                    return Ok(path);
                }

                path.push(Symbol::intern(&disambiguated_data.data.to_string()));
                Ok(path)
            }

            fn path_generic_args(
                self,
                print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
                _args: &[GenericArg<'tcx>],
            ) -> Result<Self::Path, Self::Error> {
                print_prefix(self)
            }
        }

        AbsolutePathPrinter { tcx: self.tcx }.print_def_path(def_id, &[]).unwrap()
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

impl<'tcx> ty::layout::HasParamEnv<'tcx> for LateContext<'tcx> {
    #[inline]
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for LateContext<'tcx> {
    type LayoutOfResult = Result<TyAndLayout<'tcx>, LayoutError<'tcx>>;

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, _: Span, _: Ty<'tcx>) -> LayoutError<'tcx> {
        err
    }
}

pub fn parse_lint_and_tool_name(lint_name: &str) -> (Option<Symbol>, &str) {
    match lint_name.split_once("::") {
        Some((tool_name, lint_name)) => {
            let tool_name = Symbol::intern(tool_name);

            (Some(tool_name), lint_name)
        }
        None => (None, lint_name),
    }
}
