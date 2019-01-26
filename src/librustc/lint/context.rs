//! Implementation of lint checking.
//!
//! The lint checking is mostly consolidated into one pass which runs
//! after all other analyses. Throughout compilation, lint warnings
//! can be added via the `add_lint` method on the Session structure. This
//! requires a span and an id of the node that the lint is being added to. The
//! lint isn't actually emitted at that time because it is unknown what the
//! actual lint level at that location is.
//!
//! To actually emit lint warnings/errors, a separate pass is used.
//! A context keeps track of the current state of all lint levels.
//! Upon entering a node of the ast which can modify the lint settings, the
//! previous lint state is pushed onto a stack and the ast is then recursed
//! upon.  As the ast is traversed, this keeps track of the current lint level
//! for all lint attributes.

use self::TargetLint::*;

use std::slice;
use rustc_data_structures::sync::ReadGuard;
use lint::{EarlyLintPass, EarlyLintPassObject, LateLintPassObject};
use lint::{LintArray, Level, Lint, LintId, LintPass, LintBuffer};
use lint::builtin::BuiltinLintDiagnostics;
use lint::levels::{LintLevelSets, LintLevelsBuilder};
use middle::privacy::AccessLevels;
use rustc_serialize::{Decoder, Decodable, Encoder, Encodable};
use session::{config, early_error, Session};
use ty::{self, TyCtxt, Ty};
use ty::layout::{LayoutError, LayoutOf, TyLayout};
use util::nodemap::FxHashMap;
use util::common::time;

use std::default::Default as StdDefault;
use syntax::ast;
use syntax::edition;
use syntax_pos::{MultiSpan, Span, symbol::{LocalInternedString, Symbol}};
use errors::DiagnosticBuilder;
use hir;
use hir::def_id::LOCAL_CRATE;
use hir::intravisit as hir_visit;
use syntax::util::lev_distance::find_best_match_for_name;
use syntax::visit as ast_visit;

/// Information about the registered lints.
///
/// This is basically the subset of `Context` that we can
/// build early in the compile pipeline.
pub struct LintStore {
    /// Registered lints. The bool is true if the lint was
    /// added by a plugin.
    lints: Vec<(&'static Lint, bool)>,

    /// Trait objects for each lint pass.
    /// This is only `None` while performing a lint pass.
    pre_expansion_passes: Option<Vec<EarlyLintPassObject>>,
    early_passes: Option<Vec<EarlyLintPassObject>>,
    late_passes: Option<Vec<LateLintPassObject>>,

    /// Lints indexed by name.
    by_name: FxHashMap<String, TargetLint>,

    /// Map of registered lint groups to what lints they expand to.
    lint_groups: FxHashMap<&'static str, LintGroup>,

    /// Extra info for future incompatibility lints, describing the
    /// issue or RFC that caused the incompatibility.
    future_incompatible: FxHashMap<LintId, FutureIncompatibleInfo>,
}

pub struct LintSession<'a, PassObject> {
    /// Reference to the store of registered lints.
    lints: ReadGuard<'a, LintStore>,

    /// Trait objects for each lint pass.
    passes: Option<Vec<PassObject>>,
}

/// Lints that are buffered up early on in the `Session` before the
/// `LintLevels` is calculated
#[derive(PartialEq, RustcEncodable, RustcDecodable, Debug)]
pub struct BufferedEarlyLint {
    pub lint_id: LintId,
    pub ast_id: ast::NodeId,
    pub span: MultiSpan,
    pub msg: String,
    pub diagnostic: BuiltinLintDiagnostics,
}

/// Extra information for a future incompatibility lint. See the call
/// to `register_future_incompatible` in `librustc_lint/lib.rs` for
/// guidelines.
pub struct FutureIncompatibleInfo {
    pub id: LintId,
    /// e.g., a URL for an issue/PR/RFC or error code
    pub reference: &'static str,
    /// If this is an edition fixing lint, the edition in which
    /// this lint becomes obsolete
    pub edition: Option<edition::Edition>,
}

/// The target of the `by_name` map, which accounts for renaming/deprecation.
enum TargetLint {
    /// A direct lint target
    Id(LintId),

    /// Temporary renaming, used for easing migration pain; see #16545
    Renamed(String, LintId),

    /// Lint with this name existed previously, but has been removed/deprecated.
    /// The string argument is the reason for removal.
    Removed(String),
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

pub enum CheckLintNameResult<'a> {
    Ok(&'a [LintId]),
    /// Lint doesn't exist. Potentially contains a suggestion for a correct lint name.
    NoLint(Option<Symbol>),
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
            pre_expansion_passes: Some(vec![]),
            early_passes: Some(vec![]),
            late_passes: Some(vec![]),
            by_name: Default::default(),
            future_incompatible: Default::default(),
            lint_groups: Default::default(),
        }
    }

    pub fn get_lints<'t>(&'t self) -> &'t [(&'static Lint, bool)] {
        &self.lints
    }

    pub fn get_lint_groups<'t>(&'t self) -> Vec<(&'static str, Vec<LintId>, bool)> {
        self.lint_groups.iter()
            .filter(|(_, LintGroup { depr, .. })| {
                // Don't display deprecated lint groups.
                depr.is_none()
            })
            .map(|(k, LintGroup { lint_ids, from_plugin, .. })| {
                (*k, lint_ids.clone(), *from_plugin)
            })
            .collect()
    }

    pub fn register_early_pass(&mut self,
                               sess: Option<&Session>,
                               from_plugin: bool,
                               register_only: bool,
                               pass: EarlyLintPassObject) {
        self.push_pass(sess, from_plugin, &pass);
        if !register_only {
            self.early_passes.as_mut().unwrap().push(pass);
        }
    }

    pub fn register_pre_expansion_pass(
        &mut self,
        sess: Option<&Session>,
        from_plugin: bool,
        register_only: bool,
        pass: EarlyLintPassObject,
    ) {
        self.push_pass(sess, from_plugin, &pass);
        if !register_only {
            self.pre_expansion_passes.as_mut().unwrap().push(pass);
        }
    }

    pub fn register_late_pass(&mut self,
                              sess: Option<&Session>,
                              from_plugin: bool,
                              pass: LateLintPassObject) {
        self.push_pass(sess, from_plugin, &pass);
        self.late_passes.as_mut().unwrap().push(pass);
    }

    // Helper method for register_early/late_pass
    fn push_pass<P: LintPass + ?Sized + 'static>(&mut self,
                                        sess: Option<&Session>,
                                        from_plugin: bool,
                                        pass: &Box<P>) {
        for lint in pass.get_lints() {
            self.lints.push((lint, from_plugin));

            let id = LintId::of(lint);
            if self.by_name.insert(lint.name_lower(), Id(id)).is_some() {
                let msg = format!("duplicate specification of lint {}", lint.name_lower());
                match (sess, from_plugin) {
                    // We load builtin lints first, so a duplicate is a compiler bug.
                    // Use early_error when handling -W help with no crate.
                    (None, _) => early_error(config::ErrorOutputType::default(), &msg[..]),
                    (Some(_), false) => bug!("{}", msg),

                    // A duplicate name from a plugin is a user error.
                    (Some(sess), true)  => sess.err(&msg[..]),
                }
            }
        }
    }

    pub fn register_future_incompatible(&mut self,
                                        sess: Option<&Session>,
                                        lints: Vec<FutureIncompatibleInfo>) {

        for edition in edition::ALL_EDITIONS {
            let lints = lints.iter().filter(|f| f.edition == Some(*edition)).map(|f| f.id)
                             .collect::<Vec<_>>();
            if !lints.is_empty() {
                self.register_group(sess, false, edition.lint_name(), None, lints)
            }
        }

        let mut future_incompatible = Vec::with_capacity(lints.len());
        for lint in lints {
            future_incompatible.push(lint.id);
            self.future_incompatible.insert(lint.id, lint);
        }

        self.register_group(
            sess,
            false,
            "future_incompatible",
            None,
            future_incompatible,
        );
    }

    pub fn future_incompatible(&self, id: LintId) -> Option<&FutureIncompatibleInfo> {
        self.future_incompatible.get(&id)
    }

    pub fn register_group_alias(
        &mut self,
        lint_name: &'static str,
        alias: &'static str,
    ) {
        self.lint_groups.insert(alias, LintGroup {
            lint_ids: vec![],
            from_plugin: false,
            depr: Some(LintAlias { name: lint_name, silent: true }),
        });
    }

    pub fn register_group(
        &mut self,
        sess: Option<&Session>,
        from_plugin: bool,
        name: &'static str,
        deprecated_name: Option<&'static str>,
        to: Vec<LintId>,
    ) {
        let new = self
            .lint_groups
            .insert(name, LintGroup {
                lint_ids: to,
                from_plugin,
                depr: None,
            })
            .is_none();
        if let Some(deprecated) = deprecated_name {
            self.lint_groups.insert(deprecated, LintGroup {
                lint_ids: vec![],
                from_plugin,
                depr: Some(LintAlias { name, silent: false }),
            });
        }

        if !new {
            let msg = format!("duplicate specification of lint group {}", name);
            match (sess, from_plugin) {
                // We load builtin lints first, so a duplicate is a compiler bug.
                // Use early_error when handling -W help with no crate.
                (None, _) => early_error(config::ErrorOutputType::default(), &msg[..]),
                (Some(_), false) => bug!("{}", msg),

                // A duplicate name from a plugin is a user error.
                (Some(sess), true)  => sess.err(&msg[..]),
            }
        }
    }

    pub fn register_renamed(&mut self, old_name: &str, new_name: &str) {
        let target = match self.by_name.get(new_name) {
            Some(&Id(lint_id)) => lint_id.clone(),
            _ => bug!("invalid lint renaming of {} to {}", old_name, new_name)
        };
        self.by_name.insert(old_name.to_string(), Renamed(new_name.to_string(), target));
    }

    pub fn register_removed(&mut self, name: &str, reason: &str) {
        self.by_name.insert(name.into(), Removed(reason.into()));
    }

    pub fn find_lints(&self, mut lint_name: &str) -> Result<Vec<LintId>, FindLintError> {
        match self.by_name.get(lint_name) {
            Some(&Id(lint_id)) => Ok(vec![lint_id]),
            Some(&Renamed(_, lint_id)) => {
                Ok(vec![lint_id])
            },
            Some(&Removed(_)) => {
                Err(FindLintError::Removed)
            },
            None => {
                loop {
                    return match self.lint_groups.get(lint_name) {
                        Some(LintGroup {lint_ids, depr, .. }) => {
                            if let Some(LintAlias { name, .. }) = depr {
                                lint_name = name;
                                continue;
                            }
                            Ok(lint_ids.clone())
                        }
                        None => Err(FindLintError::Removed)
                    };
                }
            }
        }
    }

    /// Checks the validity of lint names derived from the command line
    pub fn check_lint_name_cmdline(&self,
                                   sess: &Session,
                                   lint_name: &str,
                                   level: Level) {
        let db = match self.check_lint_name(lint_name, None) {
            CheckLintNameResult::Ok(_) => None,
            CheckLintNameResult::Warning(ref msg, _) => {
                Some(sess.struct_warn(msg))
            },
            CheckLintNameResult::NoLint(suggestion) => {
                let mut err = struct_err!(sess, E0602, "unknown lint: `{}`", lint_name);

                if let Some(suggestion) = suggestion {
                    err.help(&format!("did you mean: `{}`", suggestion));
                }

                Some(err)
            }
            CheckLintNameResult::Tool(result) => match result {
                Err((Some(_), new_name)) => Some(sess.struct_warn(&format!(
                    "lint name `{}` is deprecated \
                     and does not have an effect anymore. \
                     Use: {}",
                    lint_name, new_name
                ))),
                _ => None,
            },
        };

        if let Some(mut db) = db {
            let msg = format!("requested on the command line with `{} {}`",
                              match level {
                                  Level::Allow => "-A",
                                  Level::Warn => "-W",
                                  Level::Deny => "-D",
                                  Level::Forbid => "-F",
                              },
                              lint_name);
            db.note(&msg);
            db.emit();
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
        tool_name: Option<LocalInternedString>,
    ) -> CheckLintNameResult<'_> {
        let complete_name = if let Some(tool_name) = tool_name {
            format!("{}::{}", tool_name, lint_name)
        } else {
            lint_name.to_string()
        };
        // If the lint was scoped with `tool::` check if the tool lint exists
        if let Some(_) = tool_name {
            match self.by_name.get(&complete_name) {
                None => match self.lint_groups.get(&*complete_name) {
                    None => return CheckLintNameResult::Tool(Err((None, String::new()))),
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
                format!(
                    "lint `{}` has been renamed to `{}`",
                    complete_name, new_name
                ),
                Some(new_name.to_owned()),
            ),
            Some(&Removed(ref reason)) => CheckLintNameResult::Warning(
                format!("lint `{}` has been removed: `{}`", complete_name, reason),
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
                            CheckLintNameResult::Tool(Err((
                                Some(&lint_ids),
                                name.to_string(),
                            )))
                        };
                    }
                    CheckLintNameResult::Ok(&lint_ids)
                }
            },
            Some(&Id(ref id)) => CheckLintNameResult::Ok(slice::from_ref(id)),
        }
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
                None => {
                    let symbols = self.by_name.keys()
                        .map(|name| Symbol::intern(&name))
                        .collect::<Vec<_>>();

                    let suggestion =
                        find_best_match_for_name(symbols.iter(), &lint_name.to_lowercase(), None);

                    CheckLintNameResult::NoLint(suggestion)
                }
                Some(LintGroup { lint_ids, depr, .. }) => {
                    // Reaching this would be weird, but let's cover this case anyway
                    if let Some(LintAlias { name, silent }) = depr {
                        let LintGroup { lint_ids, .. } = self.lint_groups.get(name).unwrap();
                        return if *silent {
                            CheckLintNameResult::Tool(Err((Some(&lint_ids), complete_name)))
                        } else {
                            CheckLintNameResult::Tool(Err((
                                Some(&lint_ids),
                                name.to_string(),
                            )))
                        };
                    }
                    CheckLintNameResult::Tool(Err((Some(&lint_ids), complete_name)))
                }
            },
            Some(&Id(ref id)) => {
                CheckLintNameResult::Tool(Err((Some(slice::from_ref(id)), complete_name)))
            }
            _ => CheckLintNameResult::NoLint(None),
        }
    }
}

/// Context for lint checking after type checking.
pub struct LateContext<'a, 'tcx: 'a> {
    /// Type context we're checking in.
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,

    /// Side-tables for the body we are in.
    pub tables: &'a ty::TypeckTables<'tcx>,

    /// Parameter environment for the item we are in.
    pub param_env: ty::ParamEnv<'tcx>,

    /// Items accessible from the crate being checked.
    pub access_levels: &'a AccessLevels,

    /// The store of registered lints and the lint levels.
    lint_sess: LintSession<'tcx, LateLintPassObject>,

    last_ast_node_with_lint_attrs: ast::NodeId,

    /// Generic type parameters in scope for the item we are in.
    pub generics: Option<&'tcx hir::Generics>,
}

/// Context for lint checking of the AST, after expansion, before lowering to
/// HIR.
pub struct EarlyContext<'a> {
    /// Type context we're checking in.
    pub sess: &'a Session,

    /// The crate being checked.
    pub krate: &'a ast::Crate,

    builder: LintLevelsBuilder<'a>,

    /// The store of registered lints and the lint levels.
    lint_sess: LintSession<'a, EarlyLintPassObject>,

    buffered: LintBuffer,
}

pub struct EarlyContextAndPass<'a, T: EarlyLintPass> {
    context: EarlyContext<'a>,
    pass: T,
}

/// Convenience macro for calling a `LintPass` method on every pass in the context.
macro_rules! run_lints { ($cx:expr, $f:ident, $($args:expr),*) => ({
    // Move the vector of passes out of `$cx` so that we can
    // iterate over it mutably while passing `$cx` to the methods.
    let mut passes = $cx.lint_sess_mut().passes.take().unwrap();
    for obj in &mut passes {
        obj.$f($cx, $($args),*);
    }
    $cx.lint_sess_mut().passes = Some(passes);
}) }

pub trait LintPassObject: Sized {}

impl LintPassObject for EarlyLintPassObject {}

impl LintPassObject for LateLintPassObject {}


pub trait LintContext<'tcx>: Sized {
    type PassObject: LintPassObject;

    fn sess(&self) -> &Session;
    fn lints(&self) -> &LintStore;
    fn lint_sess(&self) -> &LintSession<'tcx, Self::PassObject>;
    fn lint_sess_mut(&mut self) -> &mut LintSession<'tcx, Self::PassObject>;

    fn lookup_and_emit<S: Into<MultiSpan>>(&self,
                                           lint: &'static Lint,
                                           span: Option<S>,
                                           msg: &str) {
        self.lookup(lint, span, msg).emit();
    }

    fn lookup_and_emit_with_diagnostics<S: Into<MultiSpan>>(&self,
                                                            lint: &'static Lint,
                                                            span: Option<S>,
                                                            msg: &str,
                                                            diagnostic: BuiltinLintDiagnostics) {
        let mut db = self.lookup(lint, span, msg);
        diagnostic.run(self.sess(), &mut db);
        db.emit();
    }

    fn lookup<S: Into<MultiSpan>>(&self,
                                  lint: &'static Lint,
                                  span: Option<S>,
                                  msg: &str)
                                  -> DiagnosticBuilder<'_>;

    /// Emit a lint at the appropriate level, for a particular span.
    fn span_lint<S: Into<MultiSpan>>(&self, lint: &'static Lint, span: S, msg: &str) {
        self.lookup_and_emit(lint, Some(span), msg);
    }

    fn struct_span_lint<S: Into<MultiSpan>>(&self,
                                            lint: &'static Lint,
                                            span: S,
                                            msg: &str)
                                            -> DiagnosticBuilder<'_> {
        self.lookup(lint, Some(span), msg)
    }

    /// Emit a lint and note at the appropriate level, for a particular span.
    fn span_lint_note(&self, lint: &'static Lint, span: Span, msg: &str,
                      note_span: Span, note: &str) {
        let mut err = self.lookup(lint, Some(span), msg);
        if note_span == span {
            err.note(note);
        } else {
            err.span_note(note_span, note);
        }
        err.emit();
    }

    /// Emit a lint and help at the appropriate level, for a particular span.
    fn span_lint_help(&self, lint: &'static Lint, span: Span,
                      msg: &str, help: &str) {
        let mut err = self.lookup(lint, Some(span), msg);
        self.span_lint(lint, span, msg);
        err.span_help(span, help);
        err.emit();
    }

    /// Emit a lint at the appropriate level, with no associated span.
    fn lint(&self, lint: &'static Lint, msg: &str) {
        self.lookup_and_emit(lint, None as Option<Span>, msg);
    }
}


impl<'a> EarlyContext<'a> {
    fn new(
        sess: &'a Session,
        krate: &'a ast::Crate,
        buffered: LintBuffer,
    ) -> EarlyContext<'a> {
        EarlyContext {
            sess,
            krate,
            lint_sess: LintSession {
                lints: sess.lint_store.borrow(),
                passes: None,
            },
            builder: LintLevelSets::builder(sess),
            buffered,
        }
    }
}

macro_rules! run_early_pass { ($cx:expr, $f:ident, $($args:expr),*) => ({
    $cx.pass.$f(&$cx.context, $($args),*);
}) }

impl<'a, T: EarlyLintPass> EarlyContextAndPass<'a, T> {
    fn check_id(&mut self, id: ast::NodeId) {
        for early_lint in self.context.buffered.take(id) {
            self.context.lookup_and_emit_with_diagnostics(
                early_lint.lint_id.lint,
                Some(early_lint.span.clone()),
                &early_lint.msg,
                early_lint.diagnostic
            );
        }
    }

    /// Merge the lints specified by any lint attributes into the
    /// current lint context, call the provided function, then reset the
    /// lints in effect to their previous state.
    fn with_lint_attrs<F>(&mut self,
                          id: ast::NodeId,
                          attrs: &'a [ast::Attribute],
                          f: F)
        where F: FnOnce(&mut Self)
    {
        let push = self.context.builder.push(attrs);
        self.check_id(id);
        self.enter_attrs(attrs);
        f(self);
        self.exit_attrs(attrs);
        self.context.builder.pop(push);
    }

    fn enter_attrs(&mut self, attrs: &'a [ast::Attribute]) {
        debug!("early context: enter_attrs({:?})", attrs);
        run_early_pass!(self, enter_lint_attrs, attrs);
    }

    fn exit_attrs(&mut self, attrs: &'a [ast::Attribute]) {
        debug!("early context: exit_attrs({:?})", attrs);
        run_early_pass!(self, exit_lint_attrs, attrs);
    }
}

impl<'a, 'tcx> LintContext<'tcx> for LateContext<'a, 'tcx> {
    type PassObject = LateLintPassObject;

    /// Get the overall compiler `Session` object.
    fn sess(&self) -> &Session {
        &self.tcx.sess
    }

    fn lints(&self) -> &LintStore {
        &*self.lint_sess.lints
    }

    fn lint_sess(&self) -> &LintSession<'tcx, Self::PassObject> {
        &self.lint_sess
    }

    fn lint_sess_mut(&mut self) -> &mut LintSession<'tcx, Self::PassObject> {
        &mut self.lint_sess
    }

    fn lookup<S: Into<MultiSpan>>(&self,
                                  lint: &'static Lint,
                                  span: Option<S>,
                                  msg: &str)
                                  -> DiagnosticBuilder<'_> {
        let id = self.last_ast_node_with_lint_attrs;
        match span {
            Some(s) => self.tcx.struct_span_lint_node(lint, id, s, msg),
            None => self.tcx.struct_lint_node(lint, id, msg),
        }
    }
}

impl<'a> LintContext<'a> for EarlyContext<'a> {
    type PassObject = EarlyLintPassObject;

    /// Get the overall compiler `Session` object.
    fn sess(&self) -> &Session {
        &self.sess
    }

    fn lints(&self) -> &LintStore {
        &*self.lint_sess.lints
    }

    fn lint_sess(&self) -> &LintSession<'a, Self::PassObject> {
        &self.lint_sess
    }

    fn lint_sess_mut(&mut self) -> &mut LintSession<'a, Self::PassObject> {
        &mut self.lint_sess
    }

    fn lookup<S: Into<MultiSpan>>(&self,
                                  lint: &'static Lint,
                                  span: Option<S>,
                                  msg: &str)
                                  -> DiagnosticBuilder<'_> {
        self.builder.struct_lint(lint, span.map(|s| s.into()), msg)
    }
}

impl<'a, 'tcx> LateContext<'a, 'tcx> {
    /// Merge the lints specified by any lint attributes into the
    /// current lint context, call the provided function, then reset the
    /// lints in effect to their previous state.
    fn with_lint_attrs<F>(&mut self,
                          id: ast::NodeId,
                          attrs: &'tcx [ast::Attribute],
                          f: F)
        where F: FnOnce(&mut Self)
    {
        let prev = self.last_ast_node_with_lint_attrs;
        self.last_ast_node_with_lint_attrs = id;
        self.enter_attrs(attrs);
        f(self);
        self.exit_attrs(attrs);
        self.last_ast_node_with_lint_attrs = prev;
    }

    fn enter_attrs(&mut self, attrs: &'tcx [ast::Attribute]) {
        debug!("late context: enter_attrs({:?})", attrs);
        run_lints!(self, enter_lint_attrs, attrs);
    }

    fn exit_attrs(&mut self, attrs: &'tcx [ast::Attribute]) {
        debug!("late context: exit_attrs({:?})", attrs);
        run_lints!(self, exit_lint_attrs, attrs);
    }

    fn with_param_env<F>(&mut self, id: ast::NodeId, f: F)
        where F: FnOnce(&mut Self),
    {
        let old_param_env = self.param_env;
        self.param_env = self.tcx.param_env(self.tcx.hir().local_def_id(id));
        f(self);
        self.param_env = old_param_env;
    }
    pub fn current_lint_root(&self) -> ast::NodeId {
        self.last_ast_node_with_lint_attrs
    }
}

impl<'a, 'tcx> LayoutOf for LateContext<'a, 'tcx> {
    type Ty = Ty<'tcx>;
    type TyLayout = Result<TyLayout<'tcx>, LayoutError<'tcx>>;

    fn layout_of(&self, ty: Ty<'tcx>) -> Self::TyLayout {
        self.tcx.layout_of(self.param_env.and(ty))
    }
}

impl<'a, 'tcx> hir_visit::Visitor<'tcx> for LateContext<'a, 'tcx> {
    /// Because lints are scoped lexically, we want to walk nested
    /// items in the context of the outer item, so enable
    /// deep-walking.
    fn nested_visit_map<'this>(&'this mut self) -> hir_visit::NestedVisitorMap<'this, 'tcx> {
        hir_visit::NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let old_tables = self.tables;
        self.tables = self.tcx.body_tables(body);
        let body = self.tcx.hir().body(body);
        self.visit_body(body);
        self.tables = old_tables;
    }

    fn visit_body(&mut self, body: &'tcx hir::Body) {
        run_lints!(self, check_body, body);
        hir_visit::walk_body(self, body);
        run_lints!(self, check_body_post, body);
    }

    fn visit_item(&mut self, it: &'tcx hir::Item) {
        let generics = self.generics.take();
        self.generics = it.node.generics();
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            cx.with_param_env(it.id, |cx| {
                run_lints!(cx, check_item, it);
                hir_visit::walk_item(cx, it);
                run_lints!(cx, check_item_post, it);
            });
        });
        self.generics = generics;
    }

    fn visit_foreign_item(&mut self, it: &'tcx hir::ForeignItem) {
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            cx.with_param_env(it.id, |cx| {
                run_lints!(cx, check_foreign_item, it);
                hir_visit::walk_foreign_item(cx, it);
                run_lints!(cx, check_foreign_item_post, it);
            });
        })
    }

    fn visit_pat(&mut self, p: &'tcx hir::Pat) {
        run_lints!(self, check_pat, p);
        hir_visit::walk_pat(self, p);
    }

    fn visit_expr(&mut self, e: &'tcx hir::Expr) {
        self.with_lint_attrs(e.id, &e.attrs, |cx| {
            run_lints!(cx, check_expr, e);
            hir_visit::walk_expr(cx, e);
            run_lints!(cx, check_expr_post, e);
        })
    }

    fn visit_stmt(&mut self, s: &'tcx hir::Stmt) {
        // statement attributes are actually just attributes on one of
        // - item
        // - local
        // - expression
        // so we keep track of lint levels there
        run_lints!(self, check_stmt, s);
        hir_visit::walk_stmt(self, s);
    }

    fn visit_fn(&mut self, fk: hir_visit::FnKind<'tcx>, decl: &'tcx hir::FnDecl,
                body_id: hir::BodyId, span: Span, id: ast::NodeId) {
        // Wrap in tables here, not just in visit_nested_body,
        // in order for `check_fn` to be able to use them.
        let old_tables = self.tables;
        self.tables = self.tcx.body_tables(body_id);
        let body = self.tcx.hir().body(body_id);
        run_lints!(self, check_fn, fk, decl, body, span, id);
        hir_visit::walk_fn(self, fk, decl, body_id, span, id);
        run_lints!(self, check_fn_post, fk, decl, body, span, id);
        self.tables = old_tables;
    }

    fn visit_variant_data(&mut self,
                        s: &'tcx hir::VariantData,
                        name: ast::Name,
                        g: &'tcx hir::Generics,
                        item_id: ast::NodeId,
                        _: Span) {
        run_lints!(self, check_struct_def, s, name, g, item_id);
        hir_visit::walk_struct_def(self, s);
        run_lints!(self, check_struct_def_post, s, name, g, item_id);
    }

    fn visit_struct_field(&mut self, s: &'tcx hir::StructField) {
        self.with_lint_attrs(s.id, &s.attrs, |cx| {
            run_lints!(cx, check_struct_field, s);
            hir_visit::walk_struct_field(cx, s);
        })
    }

    fn visit_variant(&mut self,
                     v: &'tcx hir::Variant,
                     g: &'tcx hir::Generics,
                     item_id: ast::NodeId) {
        self.with_lint_attrs(v.node.data.id(), &v.node.attrs, |cx| {
            run_lints!(cx, check_variant, v, g);
            hir_visit::walk_variant(cx, v, g, item_id);
            run_lints!(cx, check_variant_post, v, g);
        })
    }

    fn visit_ty(&mut self, t: &'tcx hir::Ty) {
        run_lints!(self, check_ty, t);
        hir_visit::walk_ty(self, t);
    }

    fn visit_name(&mut self, sp: Span, name: ast::Name) {
        run_lints!(self, check_name, sp, name);
    }

    fn visit_mod(&mut self, m: &'tcx hir::Mod, s: Span, n: ast::NodeId) {
        run_lints!(self, check_mod, m, s, n);
        hir_visit::walk_mod(self, m, n);
        run_lints!(self, check_mod_post, m, s, n);
    }

    fn visit_local(&mut self, l: &'tcx hir::Local) {
        self.with_lint_attrs(l.id, &l.attrs, |cx| {
            run_lints!(cx, check_local, l);
            hir_visit::walk_local(cx, l);
        })
    }

    fn visit_block(&mut self, b: &'tcx hir::Block) {
        run_lints!(self, check_block, b);
        hir_visit::walk_block(self, b);
        run_lints!(self, check_block_post, b);
    }

    fn visit_arm(&mut self, a: &'tcx hir::Arm) {
        run_lints!(self, check_arm, a);
        hir_visit::walk_arm(self, a);
    }

    fn visit_generic_param(&mut self, p: &'tcx hir::GenericParam) {
        run_lints!(self, check_generic_param, p);
        hir_visit::walk_generic_param(self, p);
    }

    fn visit_generics(&mut self, g: &'tcx hir::Generics) {
        run_lints!(self, check_generics, g);
        hir_visit::walk_generics(self, g);
    }

    fn visit_where_predicate(&mut self, p: &'tcx hir::WherePredicate) {
        run_lints!(self, check_where_predicate, p);
        hir_visit::walk_where_predicate(self, p);
    }

    fn visit_poly_trait_ref(&mut self, t: &'tcx hir::PolyTraitRef,
                            m: hir::TraitBoundModifier) {
        run_lints!(self, check_poly_trait_ref, t, m);
        hir_visit::walk_poly_trait_ref(self, t, m);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {
        let generics = self.generics.take();
        self.generics = Some(&trait_item.generics);
        self.with_lint_attrs(trait_item.id, &trait_item.attrs, |cx| {
            cx.with_param_env(trait_item.id, |cx| {
                run_lints!(cx, check_trait_item, trait_item);
                hir_visit::walk_trait_item(cx, trait_item);
                run_lints!(cx, check_trait_item_post, trait_item);
            });
        });
        self.generics = generics;
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        let generics = self.generics.take();
        self.generics = Some(&impl_item.generics);
        self.with_lint_attrs(impl_item.id, &impl_item.attrs, |cx| {
            cx.with_param_env(impl_item.id, |cx| {
                run_lints!(cx, check_impl_item, impl_item);
                hir_visit::walk_impl_item(cx, impl_item);
                run_lints!(cx, check_impl_item_post, impl_item);
            });
        });
        self.generics = generics;
    }

    fn visit_lifetime(&mut self, lt: &'tcx hir::Lifetime) {
        run_lints!(self, check_lifetime, lt);
        hir_visit::walk_lifetime(self, lt);
    }

    fn visit_path(&mut self, p: &'tcx hir::Path, id: hir::HirId) {
        run_lints!(self, check_path, p, id);
        hir_visit::walk_path(self, p);
    }

    fn visit_attribute(&mut self, attr: &'tcx ast::Attribute) {
        run_lints!(self, check_attribute, attr);
    }
}

impl<'a, T: EarlyLintPass> ast_visit::Visitor<'a> for EarlyContextAndPass<'a, T> {
    fn visit_item(&mut self, it: &'a ast::Item) {
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            run_early_pass!(cx, check_item, it);
            ast_visit::walk_item(cx, it);
            run_early_pass!(cx, check_item_post, it);
        })
    }

    fn visit_foreign_item(&mut self, it: &'a ast::ForeignItem) {
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            run_early_pass!(cx, check_foreign_item, it);
            ast_visit::walk_foreign_item(cx, it);
            run_early_pass!(cx, check_foreign_item_post, it);
        })
    }

    fn visit_pat(&mut self, p: &'a ast::Pat) {
        let mut visit_subpats = true;
        run_early_pass!(self, check_pat, p, &mut visit_subpats);
        self.check_id(p.id);
        if visit_subpats {
            ast_visit::walk_pat(self, p);
        }
    }

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        self.with_lint_attrs(e.id, &e.attrs, |cx| {
            run_early_pass!(cx, check_expr, e);
            ast_visit::walk_expr(cx, e);
        })
    }

    fn visit_stmt(&mut self, s: &'a ast::Stmt) {
        run_early_pass!(self, check_stmt, s);
        self.check_id(s.id);
        ast_visit::walk_stmt(self, s);
    }

    fn visit_fn(&mut self, fk: ast_visit::FnKind<'a>, decl: &'a ast::FnDecl,
                span: Span, id: ast::NodeId) {
        run_early_pass!(self, check_fn, fk, decl, span, id);
        self.check_id(id);
        ast_visit::walk_fn(self, fk, decl, span);
        run_early_pass!(self, check_fn_post, fk, decl, span, id);
    }

    fn visit_variant_data(&mut self,
                        s: &'a ast::VariantData,
                        ident: ast::Ident,
                        g: &'a ast::Generics,
                        item_id: ast::NodeId,
                        _: Span) {
        run_early_pass!(self, check_struct_def, s, ident, g, item_id);
        self.check_id(s.id());
        ast_visit::walk_struct_def(self, s);
        run_early_pass!(self, check_struct_def_post, s, ident, g, item_id);
    }

    fn visit_struct_field(&mut self, s: &'a ast::StructField) {
        self.with_lint_attrs(s.id, &s.attrs, |cx| {
            run_early_pass!(cx, check_struct_field, s);
            ast_visit::walk_struct_field(cx, s);
        })
    }

    fn visit_variant(&mut self, v: &'a ast::Variant, g: &'a ast::Generics, item_id: ast::NodeId) {
        self.with_lint_attrs(item_id, &v.node.attrs, |cx| {
            run_early_pass!(cx, check_variant, v, g);
            ast_visit::walk_variant(cx, v, g, item_id);
            run_early_pass!(cx, check_variant_post, v, g);
        })
    }

    fn visit_ty(&mut self, t: &'a ast::Ty) {
        run_early_pass!(self, check_ty, t);
        self.check_id(t.id);
        ast_visit::walk_ty(self, t);
    }

    fn visit_ident(&mut self, ident: ast::Ident) {
        run_early_pass!(self, check_ident, ident);
    }

    fn visit_mod(&mut self, m: &'a ast::Mod, s: Span, _a: &[ast::Attribute], n: ast::NodeId) {
        run_early_pass!(self, check_mod, m, s, n);
        self.check_id(n);
        ast_visit::walk_mod(self, m);
        run_early_pass!(self, check_mod_post, m, s, n);
    }

    fn visit_local(&mut self, l: &'a ast::Local) {
        self.with_lint_attrs(l.id, &l.attrs, |cx| {
            run_early_pass!(cx, check_local, l);
            ast_visit::walk_local(cx, l);
        })
    }

    fn visit_block(&mut self, b: &'a ast::Block) {
        run_early_pass!(self, check_block, b);
        self.check_id(b.id);
        ast_visit::walk_block(self, b);
        run_early_pass!(self, check_block_post, b);
    }

    fn visit_arm(&mut self, a: &'a ast::Arm) {
        run_early_pass!(self, check_arm, a);
        ast_visit::walk_arm(self, a);
    }

    fn visit_expr_post(&mut self, e: &'a ast::Expr) {
        run_early_pass!(self, check_expr_post, e);
    }

    fn visit_generic_param(&mut self, param: &'a ast::GenericParam) {
        run_early_pass!(self, check_generic_param, param);
        ast_visit::walk_generic_param(self, param);
    }

    fn visit_generics(&mut self, g: &'a ast::Generics) {
        run_early_pass!(self, check_generics, g);
        ast_visit::walk_generics(self, g);
    }

    fn visit_where_predicate(&mut self, p: &'a ast::WherePredicate) {
        run_early_pass!(self, check_where_predicate, p);
        ast_visit::walk_where_predicate(self, p);
    }

    fn visit_poly_trait_ref(&mut self, t: &'a ast::PolyTraitRef, m: &'a ast::TraitBoundModifier) {
        run_early_pass!(self, check_poly_trait_ref, t, m);
        ast_visit::walk_poly_trait_ref(self, t, m);
    }

    fn visit_trait_item(&mut self, trait_item: &'a ast::TraitItem) {
        self.with_lint_attrs(trait_item.id, &trait_item.attrs, |cx| {
            run_early_pass!(cx, check_trait_item, trait_item);
            ast_visit::walk_trait_item(cx, trait_item);
            run_early_pass!(cx, check_trait_item_post, trait_item);
        });
    }

    fn visit_impl_item(&mut self, impl_item: &'a ast::ImplItem) {
        self.with_lint_attrs(impl_item.id, &impl_item.attrs, |cx| {
            run_early_pass!(cx, check_impl_item, impl_item);
            ast_visit::walk_impl_item(cx, impl_item);
            run_early_pass!(cx, check_impl_item_post, impl_item);
        });
    }

    fn visit_lifetime(&mut self, lt: &'a ast::Lifetime) {
        run_early_pass!(self, check_lifetime, lt);
        self.check_id(lt.id);
    }

    fn visit_path(&mut self, p: &'a ast::Path, id: ast::NodeId) {
        run_early_pass!(self, check_path, p, id);
        self.check_id(id);
        ast_visit::walk_path(self, p);
    }

    fn visit_attribute(&mut self, attr: &'a ast::Attribute) {
        run_early_pass!(self, check_attribute, attr);
    }

    fn visit_mac_def(&mut self, mac: &'a ast::MacroDef, id: ast::NodeId) {
        run_early_pass!(self, check_mac_def, mac, id);
        self.check_id(id);
    }

    fn visit_mac(&mut self, mac: &'a ast::Mac) {
        // FIXME(#54110): So, this setup isn't really right. I think
        // that (a) the libsyntax visitor ought to be doing this as
        // part of `walk_mac`, and (b) we should be calling
        // `visit_path`, *but* that would require a `NodeId`, and I
        // want to get #53686 fixed quickly. -nmatsakis
        ast_visit::walk_path(self, &mac.node.path);

        run_early_pass!(self, check_mac, mac);
    }
}


/// Perform lint checking on a crate.
///
/// Consumes the `lint_store` field of the `Session`.
pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let access_levels = &tcx.privacy_access_levels(LOCAL_CRATE);

    let krate = tcx.hir().krate();
    let passes = tcx.sess.lint_store.borrow_mut().late_passes.take();

    let passes = {
        let mut cx = LateContext {
            tcx,
            tables: &ty::TypeckTables::empty(None),
            param_env: ty::ParamEnv::empty(),
            access_levels,
            lint_sess: LintSession {
                passes,
                lints: tcx.sess.lint_store.borrow(),
            },
            last_ast_node_with_lint_attrs: ast::CRATE_NODE_ID,
            generics: None,
        };

        // Visit the whole crate.
        cx.with_lint_attrs(ast::CRATE_NODE_ID, &krate.attrs, |cx| {
            // since the root module isn't visited as an item (because it isn't an
            // item), warn for it here.
            run_lints!(cx, check_crate, krate);

            hir_visit::walk_crate(cx, krate);

            run_lints!(cx, check_crate_post, krate);
        });
        cx.lint_sess.passes
    };

    // Put the lint store levels and passes back in the session.
    tcx.sess.lint_store.borrow_mut().late_passes = passes;
}

struct EarlyLintPassObjects<'a> {
    lints: &'a mut [EarlyLintPassObject],
}

impl LintPass for EarlyLintPassObjects<'_> {
    fn name(&self) -> &'static str {
        panic!()
    }

    fn get_lints(&self) -> LintArray {
        panic!()
    }
}

macro_rules! expand_early_lint_pass_impl_methods {
    ([$($(#[$attr:meta])* fn $name:ident($($param:ident: $arg:ty),*);)*]) => (
        $(fn $name(&mut self, context: &EarlyContext<'_>, $($param: $arg),*) {
            for obj in self.lints.iter_mut() {
                obj.$name(context, $($param),*);
            }
        })*
    )
}

macro_rules! early_lint_pass_impl {
    ([], [$($methods:tt)*]) => (
        impl EarlyLintPass for EarlyLintPassObjects<'_> {
            expand_early_lint_pass_impl_methods!([$($methods)*]);
        }
    )
}

early_lint_methods!(early_lint_pass_impl, []);


fn early_lint_crate<T: EarlyLintPass>(
    sess: &Session,
    krate: &ast::Crate,
    pass: T,
    buffered: LintBuffer,
) -> LintBuffer {
    let mut cx = EarlyContextAndPass {
        context: EarlyContext::new(sess, krate, buffered),
        pass,
    };

    // Visit the whole crate.
    cx.with_lint_attrs(ast::CRATE_NODE_ID, &krate.attrs, |cx| {
        // since the root module isn't visited as an item (because it isn't an
        // item), warn for it here.
        run_early_pass!(cx, check_crate, krate);

        ast_visit::walk_crate(cx, krate);

        run_early_pass!(cx, check_crate_post, krate);
    });
    cx.context.buffered
}

pub fn check_ast_crate<T: EarlyLintPass>(
    sess: &Session,
    krate: &ast::Crate,
    pre_expansion: bool,
    builtin_lints: T,
) {
    let (mut passes, mut buffered) = if pre_expansion {
        (
            sess.lint_store.borrow_mut().pre_expansion_passes.take().unwrap(),
            LintBuffer::default(),
        )
    } else {
        (
            sess.lint_store.borrow_mut().early_passes.take().unwrap(),
            sess.buffered_lints.borrow_mut().take().unwrap(),
        )
    };

    if !sess.opts.debugging_opts.no_interleave_lints {
        buffered = early_lint_crate(sess, krate, builtin_lints, buffered);

        if !passes.is_empty() {
            buffered = early_lint_crate(
                sess,
                krate,
                EarlyLintPassObjects { lints: &mut passes[..] },
                buffered,
            );
        }
    } else {
        for pass in &mut passes {
            buffered = time(sess, &format!("running lint: {}", pass.name()), || {
                early_lint_crate(
                    sess,
                    krate,
                    EarlyLintPassObjects { lints: slice::from_mut(pass) },
                    buffered,
                )
            });
        }
    }

    // Put the lint store levels and passes back in the session.
    if pre_expansion {
        sess.lint_store.borrow_mut().pre_expansion_passes = Some(passes);
    } else {
        sess.lint_store.borrow_mut().early_passes = Some(passes);
    }

    // All of the buffered lints should have been emitted at this point.
    // If not, that means that we somehow buffered a lint for a node id
    // that was not lint-checked (perhaps it doesn't exist?). This is a bug.
    //
    // Rustdoc runs everybody-loops before the early lints and removes
    // function bodies, so it's totally possible for linted
    // node ids to not exist (e.g., macros defined within functions for the
    // unused_macro lint) anymore. So we only run this check
    // when we're not in rustdoc mode. (see issue #47639)
    if !sess.opts.actually_rustdoc {
        for (_id, lints) in buffered.map {
            for early_lint in lints {
                sess.delay_span_bug(early_lint.span, "failed to process buffered lint here");
            }
        }
    }
}

impl Encodable for LintId {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(&self.lint.name.to_lowercase())
    }
}

impl Decodable for LintId {
    #[inline]
    fn decode<D: Decoder>(d: &mut D) -> Result<LintId, D::Error> {
        let s = d.read_str()?;
        ty::tls::with(|tcx| {
            match tcx.sess.lint_store.borrow().find_lints(&s) {
                Ok(ids) => {
                    if ids.len() != 0 {
                        panic!("invalid lint-id `{}`", s);
                    }
                    Ok(ids[0])
                }
                Err(_) => panic!("invalid lint-id `{}`", s),
            }
        })
    }
}
