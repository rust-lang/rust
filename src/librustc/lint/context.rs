// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of lint checking.
//!
//! The lint checking is mostly consolidated into one pass which runs just
//! before translation to LLVM bytecode. Throughout compilation, lint warnings
//! can be added via the `add_lint` method on the Session structure. This
//! requires a span and an id of the node that the lint is being added to. The
//! lint isn't actually emitted at that time because it is unknown what the
//! actual lint level at that location is.
//!
//! To actually emit lint warnings/errors, a separate pass is used just before
//! translation. A context keeps track of the current state of all lint levels.
//! Upon entering a node of the ast which can modify the lint settings, the
//! previous lint state is pushed onto a stack and the ast is then recursed
//! upon.  As the ast is traversed, this keeps track of the current lint level
//! for all lint attributes.
use self::TargetLint::*;

use middle::privacy::AccessLevels;
use traits::Reveal;
use ty::{self, TyCtxt};
use session::{config, early_error, Session};
use lint::{Level, LevelSource, Lint, LintId, LintPass, LintSource};
use lint::{EarlyLintPassObject, LateLintPassObject};
use lint::{Default, CommandLine, Node, Allow, Warn, Deny, Forbid};
use lint::builtin;
use rustc_serialize::{Decoder, Decodable, Encoder, Encodable};
use util::nodemap::FxHashMap;

use std::cmp;
use std::default::Default as StdDefault;
use std::mem;
use std::fmt;
use std::cell::{Ref, RefCell};
use syntax::attr;
use syntax::ast;
use syntax::symbol::Symbol;
use syntax_pos::{MultiSpan, Span};
use errors::{self, Diagnostic, DiagnosticBuilder};
use hir;
use hir::def_id::LOCAL_CRATE;
use hir::intravisit as hir_visit;
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
    /// This is only `None` while performing a lint pass. See the definition
    /// of `LintSession::new`.
    early_passes: Option<Vec<EarlyLintPassObject>>,
    late_passes: Option<Vec<LateLintPassObject>>,

    /// Lints indexed by name.
    by_name: FxHashMap<String, TargetLint>,

    /// Current levels of each lint, and where they were set.
    levels: LintLevels,

    /// Map of registered lint groups to what lints they expand to. The bool
    /// is true if the lint group was added by a plugin.
    lint_groups: FxHashMap<&'static str, (Vec<LintId>, bool)>,

    /// Extra info for future incompatibility lints, descibing the
    /// issue or RFC that caused the incompatibility.
    future_incompatible: FxHashMap<LintId, FutureIncompatibleInfo>,
}


#[derive(Default)]
struct LintLevels {
    /// Current levels of each lint, and where they were set.
    levels: FxHashMap<LintId, LevelSource>,

    /// Maximum level a lint can be
    lint_cap: Option<Level>,
}


pub struct LintSession<'a, PassObject> {
    /// Reference to the store of registered lints.
    lints: Ref<'a, LintStore>,

    /// The current lint levels.
    levels: LintLevels,

    /// When recursing into an attributed node of the ast which modifies lint
    /// levels, this stack keeps track of the previous lint levels of whatever
    /// was modified.
    stack: Vec<(LintId, LevelSource)>,

    /// Trait objects for each lint pass.
    passes: Option<Vec<PassObject>>,
}


/// When you call `add_lint` on the session, you wind up storing one
/// of these, which records a "potential lint" at a particular point.
#[derive(PartialEq, RustcEncodable, RustcDecodable)]
pub struct EarlyLint {
    /// what lint is this? (e.g., `dead_code`)
    pub id: LintId,

    /// the main message
    pub diagnostic: Diagnostic,
}

impl fmt::Debug for EarlyLint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("EarlyLint")
            .field("id", &self.id)
            .field("span", &self.diagnostic.span)
            .field("diagnostic", &self.diagnostic)
            .finish()
    }
}

pub trait IntoEarlyLint {
    fn into_early_lint(self, id: LintId) -> EarlyLint;
}

impl<'a, S: Into<MultiSpan>> IntoEarlyLint for (S, &'a str) {
    fn into_early_lint(self, id: LintId) -> EarlyLint {
        let (span, msg) = self;
        let mut diagnostic = Diagnostic::new(errors::Level::Warning, msg);
        diagnostic.set_span(span);
        EarlyLint {
            id: id,
            diagnostic: diagnostic,
        }
    }
}

impl IntoEarlyLint for Diagnostic {
    fn into_early_lint(self, id: LintId) -> EarlyLint {
        EarlyLint {
            id: id,
            diagnostic: self,
        }
    }
}

/// Extra information for a future incompatibility lint. See the call
/// to `register_future_incompatible` in `librustc_lint/lib.rs` for
/// guidelines.
pub struct FutureIncompatibleInfo {
    pub id: LintId,
    pub reference: &'static str // e.g., a URL for an issue/PR/RFC or error code
}

/// The targed of the `by_name` map, which accounts for renaming/deprecation.
enum TargetLint {
    /// A direct lint target
    Id(LintId),

    /// Temporary renaming, used for easing migration pain; see #16545
    Renamed(String, LintId),

    /// Lint with this name existed previously, but has been removed/deprecated.
    /// The string argument is the reason for removal.
    Removed(String),
}

enum FindLintError {
    NotFound,
    Removed,
}

impl LintStore {
    pub fn new() -> LintStore {
        LintStore {
            lints: vec![],
            early_passes: Some(vec![]),
            late_passes: Some(vec![]),
            by_name: FxHashMap(),
            levels: LintLevels::default(),
            future_incompatible: FxHashMap(),
            lint_groups: FxHashMap(),
        }
    }

    pub fn get_lints<'t>(&'t self) -> &'t [(&'static Lint, bool)] {
        &self.lints
    }

    pub fn get_lint_groups<'t>(&'t self) -> Vec<(&'static str, Vec<LintId>, bool)> {
        self.lint_groups.iter().map(|(k, v)| (*k,
                                              v.0.clone(),
                                              v.1)).collect()
    }

    pub fn register_early_pass(&mut self,
                               sess: Option<&Session>,
                               from_plugin: bool,
                               pass: EarlyLintPassObject) {
        self.push_pass(sess, from_plugin, &pass);
        self.early_passes.as_mut().unwrap().push(pass);
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
        for &lint in pass.get_lints() {
            self.lints.push((*lint, from_plugin));

            let id = LintId::of(*lint);
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

            self.levels.set(id, (lint.default_level, Default));
        }
    }

    pub fn register_future_incompatible(&mut self,
                                        sess: Option<&Session>,
                                        lints: Vec<FutureIncompatibleInfo>) {
        let ids = lints.iter().map(|f| f.id).collect();
        self.register_group(sess, false, "future_incompatible", ids);
        for info in lints {
            self.future_incompatible.insert(info.id, info);
        }
    }

    pub fn future_incompatible(&self, id: LintId) -> Option<&FutureIncompatibleInfo> {
        self.future_incompatible.get(&id)
    }

    pub fn register_group(&mut self, sess: Option<&Session>,
                          from_plugin: bool, name: &'static str,
                          to: Vec<LintId>) {
        let new = self.lint_groups.insert(name, (to, from_plugin)).is_none();

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

    fn find_lint(&self, lint_name: &str) -> Result<LintId, FindLintError> {
        match self.by_name.get(lint_name) {
            Some(&Id(lint_id)) => Ok(lint_id),
            Some(&Renamed(_, lint_id)) => {
                Ok(lint_id)
            },
            Some(&Removed(_)) => {
                Err(FindLintError::Removed)
            },
            None => Err(FindLintError::NotFound)
        }
    }

    pub fn process_command_line(&mut self, sess: &Session) {
        for &(ref lint_name, level) in &sess.opts.lint_opts {
            check_lint_name_cmdline(sess, self,
                                    &lint_name[..], level);

            let lint_flag_val = Symbol::intern(&lint_name);
            match self.find_lint(&lint_name[..]) {
                Ok(lint_id) => self.levels.set(lint_id, (level, CommandLine(lint_flag_val))),
                Err(FindLintError::Removed) => { }
                Err(_) => {
                    match self.lint_groups.iter().map(|(&x, pair)| (x, pair.0.clone()))
                                                 .collect::<FxHashMap<&'static str,
                                                                      Vec<LintId>>>()
                                                 .get(&lint_name[..]) {
                        Some(v) => {
                            for lint_id in v {
                                self.levels.set(*lint_id, (level, CommandLine(lint_flag_val)));
                            }
                        }
                        None => {
                            // The lint or lint group doesn't exist.
                            // This is an error, but it was handled
                            // by check_lint_name_cmdline.
                        }
                    }
                }
            }
        }

        self.levels.set_lint_cap(sess.opts.lint_cap);
    }
}


impl LintLevels {
    fn get_source(&self, lint: LintId) -> LevelSource {
        match self.levels.get(&lint) {
            Some(&s) => s,
            None => (Allow, Default),
        }
    }

    fn set(&mut self, lint: LintId, mut lvlsrc: LevelSource) {
        if let Some(cap) = self.lint_cap {
            lvlsrc.0 = cmp::min(lvlsrc.0, cap);
        }
        if lvlsrc.0 == Allow {
            self.levels.remove(&lint);
        } else {
            self.levels.insert(lint, lvlsrc);
        }
    }

    fn set_lint_cap(&mut self, lint_cap: Option<Level>) {
        self.lint_cap = lint_cap;
        if let Some(cap) = lint_cap {
            for (_, level) in &mut self.levels {
                level.0 = cmp::min(level.0, cap);
            }
        }
    }
}


impl<'a, PassObject: LintPassObject> LintSession<'a, PassObject> {
    /// Creates a new `LintSession`, by moving out the `LintStore`'s initial
    /// lint levels and pass objects. These can be restored using the `restore`
    /// method.
    fn new(store: &'a RefCell<LintStore>) -> LintSession<'a, PassObject> {
        let mut s = store.borrow_mut();
        let levels = mem::replace(&mut s.levels, LintLevels::default());
        let passes = PassObject::take_passes(&mut *s);
        drop(s);
        LintSession {
            lints: store.borrow(),
            stack: Vec::new(),
            levels,
            passes,
        }
    }

    /// Restores the levels back to the original lint store.
    fn restore(self, store: &RefCell<LintStore>) {
        drop(self.lints);
        let mut s = store.borrow_mut();
        s.levels = self.levels;
        PassObject::restore_passes(&mut *s, self.passes);
    }

    fn get_source(&self, lint_id: LintId) -> LevelSource {
        self.levels.get_source(lint_id)
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
}

/// Context for lint checking of the AST, after expansion, before lowering to
/// HIR.
pub struct EarlyContext<'a> {
    /// Type context we're checking in.
    pub sess: &'a Session,

    /// The crate being checked.
    pub krate: &'a ast::Crate,

    /// The store of registered lints and the lint levels.
    lint_sess: LintSession<'a, EarlyLintPassObject>,
}

/// Convenience macro for calling a `LintPass` method on every pass in the context.
macro_rules! run_lints { ($cx:expr, $f:ident, $ps:ident, $($args:expr),*) => ({
    // Move the vector of passes out of `$cx` so that we can
    // iterate over it mutably while passing `$cx` to the methods.
    let mut passes = $cx.lint_sess_mut().passes.take().unwrap();
    for obj in &mut passes {
        obj.$f($cx, $($args),*);
    }
    $cx.lint_sess_mut().passes = Some(passes);
}) }

/// Parse the lint attributes into a vector, with `Err`s for malformed lint
/// attributes. Writing this as an iterator is an enormous mess.
// See also the hir version just below.
pub fn gather_attrs(attrs: &[ast::Attribute]) -> Vec<Result<(ast::Name, Level, Span), Span>> {
    let mut out = vec![];
    for attr in attrs {
        let r = gather_attr(attr);
        out.extend(r.into_iter());
    }
    out
}

pub fn gather_attr(attr: &ast::Attribute) -> Vec<Result<(ast::Name, Level, Span), Span>> {
    let mut out = vec![];

    let level = match attr.name().and_then(|name| Level::from_str(&name.as_str())) {
        None => return out,
        Some(lvl) => lvl,
    };

    let meta = unwrap_or!(attr.meta(), return out);
    attr::mark_used(attr);

    let metas = if let Some(metas) = meta.meta_item_list() {
        metas
    } else {
        out.push(Err(meta.span));
        return out;
    };

    for li in metas {
        out.push(li.word().map_or(Err(li.span), |word| Ok((word.name(), level, word.span))));
    }

    out
}

/// Emit a lint as a warning or an error (or not at all)
/// according to `level`.
///
/// This lives outside of `Context` so it can be used by checks
/// in trans that run after the main lint pass is finished. Most
/// lints elsewhere in the compiler should call
/// `Session::add_lint()` instead.
pub fn raw_emit_lint<S: Into<MultiSpan>>(sess: &Session,
                                         lints: &LintStore,
                                         lint: &'static Lint,
                                         lvlsrc: LevelSource,
                                         span: Option<S>,
                                         msg: &str) {
    raw_struct_lint(sess, lints, lint, lvlsrc, span, msg).emit();
}

pub fn raw_struct_lint<'a, S>(sess: &'a Session,
                              lints: &LintStore,
                              lint: &'static Lint,
                              lvlsrc: LevelSource,
                              span: Option<S>,
                              msg: &str)
                              -> DiagnosticBuilder<'a>
    where S: Into<MultiSpan>
{
    let (level, source) = lvlsrc;
    if level == Allow {
        return sess.diagnostic().struct_dummy();
    }

    let name = lint.name_lower();
    let mut def = None;

    // Except for possible note details, forbid behaves like deny.
    let effective_level = if level == Forbid { Deny } else { level };

    let mut err = match (effective_level, span) {
        (Warn, Some(sp)) => sess.struct_span_warn(sp, &msg[..]),
        (Warn, None)     => sess.struct_warn(&msg[..]),
        (Deny, Some(sp)) => sess.struct_span_err(sp, &msg[..]),
        (Deny, None)     => sess.struct_err(&msg[..]),
        _ => bug!("impossible level in raw_emit_lint"),
    };

    match source {
        Default => {
            err.note(&format!("#[{}({})] on by default", level.as_str(), name));
        },
        CommandLine(lint_flag_val) => {
            let flag = match level {
                Warn => "-W", Deny => "-D", Forbid => "-F",
                Allow => bug!("earlier conditional return should handle Allow case")
            };
            let hyphen_case_lint_name = name.replace("_", "-");
            if lint_flag_val.as_str() == name {
                err.note(&format!("requested on the command line with `{} {}`",
                                  flag, hyphen_case_lint_name));
            } else {
                let hyphen_case_flag_val = lint_flag_val.as_str().replace("_", "-");
                err.note(&format!("`{} {}` implied by `{} {}`",
                                  flag, hyphen_case_lint_name, flag, hyphen_case_flag_val));
            }
        },
        Node(lint_attr_name, src) => {
            def = Some(src);
            if lint_attr_name.as_str() != name {
                let level_str = level.as_str();
                err.note(&format!("#[{}({})] implied by #[{}({})]",
                                  level_str, name, level_str, lint_attr_name));
            }
        }
    }

    // Check for future incompatibility lints and issue a stronger warning.
    if let Some(future_incompatible) = lints.future_incompatible(LintId::of(lint)) {
        let explanation = format!("this was previously accepted by the compiler \
                                   but is being phased out; \
                                   it will become a hard error in a future release!");
        let citation = format!("for more information, see {}",
                               future_incompatible.reference);
        err.warn(&explanation);
        err.note(&citation);
    }

    if let Some(span) = def {
        sess.diag_span_note_once(&mut err, lint, span, "lint level defined here");
    }

    err
}


pub trait LintPassObject: Sized {
    fn take_passes(store: &mut LintStore) -> Option<Vec<Self>>;
    fn restore_passes(store: &mut LintStore, passes: Option<Vec<Self>>);
}

impl LintPassObject for EarlyLintPassObject {
    fn take_passes(store: &mut LintStore) -> Option<Vec<Self>> {
        store.early_passes.take()
    }

    fn restore_passes(store: &mut LintStore, passes: Option<Vec<Self>>) {
        store.early_passes = passes;
    }
}

impl LintPassObject for LateLintPassObject {
    fn take_passes(store: &mut LintStore) -> Option<Vec<Self>> {
        store.late_passes.take()
    }

    fn restore_passes(store: &mut LintStore, passes: Option<Vec<Self>>) {
        store.late_passes = passes;
    }
}


pub trait LintContext<'tcx>: Sized {
    type PassObject: LintPassObject;

    fn sess(&self) -> &Session;
    fn lints(&self) -> &LintStore;
    fn lint_sess(&self) -> &LintSession<'tcx, Self::PassObject>;
    fn lint_sess_mut(&mut self) -> &mut LintSession<'tcx, Self::PassObject>;
    fn enter_attrs(&mut self, attrs: &'tcx [ast::Attribute]);
    fn exit_attrs(&mut self, attrs: &'tcx [ast::Attribute]);

    /// Get the level of `lint` at the current position of the lint
    /// traversal.
    fn current_level(&self, lint: &'static Lint) -> Level {
        self.lint_sess().get_source(LintId::of(lint)).0
    }

    fn level_src(&self, lint: &'static Lint) -> Option<LevelSource> {
        let ref levels = self.lint_sess().levels;
        levels.levels.get(&LintId::of(lint)).map(|ls| match ls {
            &(Warn, _) => {
                let lint_id = LintId::of(builtin::WARNINGS);
                let warn_src = levels.get_source(lint_id);
                if warn_src.0 != Warn {
                    warn_src
                } else {
                    *ls
                }
            }
            _ => *ls
        })
    }

    fn lookup_and_emit<S: Into<MultiSpan>>(&self,
                                           lint: &'static Lint,
                                           span: Option<S>,
                                           msg: &str) {
        let (level, src) = match self.level_src(lint) {
            None => return,
            Some(pair) => pair,
        };

        raw_emit_lint(&self.sess(), self.lints(), lint, (level, src), span, msg);
    }

    fn lookup<S: Into<MultiSpan>>(&self,
                                  lint: &'static Lint,
                                  span: Option<S>,
                                  msg: &str)
                                  -> DiagnosticBuilder {
        let (level, src) = match self.level_src(lint) {
            None => return self.sess().diagnostic().struct_dummy(),
            Some(pair) => pair,
        };

        raw_struct_lint(&self.sess(), self.lints(), lint, (level, src), span, msg)
    }

    /// Emit a lint at the appropriate level, for a particular span.
    fn span_lint<S: Into<MultiSpan>>(&self, lint: &'static Lint, span: S, msg: &str) {
        self.lookup_and_emit(lint, Some(span), msg);
    }

    fn early_lint(&self, early_lint: &EarlyLint) {
        let span = early_lint.diagnostic.span.primary_span().expect("early lint w/o primary span");
        let mut err = self.struct_span_lint(early_lint.id.lint,
                                            span,
                                            &early_lint.diagnostic.message());
        err.copy_details_not_message(&early_lint.diagnostic);
        err.emit();
    }

    fn struct_span_lint<S: Into<MultiSpan>>(&self,
                                            lint: &'static Lint,
                                            span: S,
                                            msg: &str)
                                            -> DiagnosticBuilder {
        self.lookup(lint, Some(span), msg)
    }

    /// Emit a lint and note at the appropriate level, for a particular span.
    fn span_lint_note(&self, lint: &'static Lint, span: Span, msg: &str,
                      note_span: Span, note: &str) {
        let mut err = self.lookup(lint, Some(span), msg);
        if self.current_level(lint) != Level::Allow {
            if note_span == span {
                err.note(note);
            } else {
                err.span_note(note_span, note);
            }
        }
        err.emit();
    }

    /// Emit a lint and help at the appropriate level, for a particular span.
    fn span_lint_help(&self, lint: &'static Lint, span: Span,
                      msg: &str, help: &str) {
        let mut err = self.lookup(lint, Some(span), msg);
        self.span_lint(lint, span, msg);
        if self.current_level(lint) != Level::Allow {
            err.span_help(span, help);
        }
        err.emit();
    }

    /// Emit a lint at the appropriate level, with no associated span.
    fn lint(&self, lint: &'static Lint, msg: &str) {
        self.lookup_and_emit(lint, None as Option<Span>, msg);
    }

    /// Merge the lints specified by any lint attributes into the
    /// current lint context, call the provided function, then reset the
    /// lints in effect to their previous state.
    fn with_lint_attrs<F>(&mut self,
                          attrs: &'tcx [ast::Attribute],
                          f: F)
        where F: FnOnce(&mut Self),
    {
        // Parse all of the lint attributes, and then add them all to the
        // current dictionary of lint information. Along the way, keep a history
        // of what we changed so we can roll everything back after invoking the
        // specified closure
        let mut pushed = 0;

        for result in gather_attrs(attrs) {
            let (is_group, lint_level_spans) = match result {
                Err(span) => {
                    span_err!(self.sess(), span, E0452,
                              "malformed lint attribute");
                    continue;
                }
                Ok((lint_name, level, span)) => {
                    match self.lints().find_lint(&lint_name.as_str()) {
                        Ok(lint_id) => (false, vec![(lint_id, level, span)]),
                        Err(FindLintError::NotFound) => {
                            match self.lints().lint_groups.get(&*lint_name.as_str()) {
                                Some(&(ref v, _)) => (true,
                                                      v.iter()
                                                      .map(|lint_id: &LintId|
                                                           (*lint_id, level, span))
                                                      .collect()),
                                None => {
                                    // The lint or lint group doesn't exist.
                                    // This is an error, but it was handled
                                    // by check_lint_name_attribute.
                                    continue;
                                }
                            }
                        }
                        Err(FindLintError::Removed) => continue,
                    }
                }
            };

            let lint_attr_name = result.expect("lint attribute should be well-formed").0;

            for (lint_id, level, span) in lint_level_spans {
                let (now, now_source) = self.lint_sess().get_source(lint_id);
                if now == Forbid && level != Forbid {
                    let forbidden_lint_name = match now_source {
                        LintSource::Default => lint_id.to_string(),
                        LintSource::Node(name, _) => name.to_string(),
                        LintSource::CommandLine(name) => name.to_string(),
                    };
                    let mut diag_builder = struct_span_err!(self.sess(), span, E0453,
                                                            "{}({}) overruled by outer forbid({})",
                                                            level.as_str(), lint_attr_name,
                                                            forbidden_lint_name);
                    diag_builder.span_label(span, "overruled by previous forbid");
                    match now_source {
                        LintSource::Default => &mut diag_builder,
                        LintSource::Node(_, forbid_source_span) => {
                            diag_builder.span_label(forbid_source_span,
                                                    "`forbid` level set here")
                        },
                        LintSource::CommandLine(_) => {
                            diag_builder.note("`forbid` lint level was set on command line")
                        }
                    }.emit();
                    if is_group { // don't set a separate error for every lint in the group
                        break;
                    }
                } else if now != level {
                    let cx = self.lint_sess_mut();
                    cx.stack.push((lint_id, (now, now_source)));
                    pushed += 1;
                    cx.levels.set(lint_id, (level, Node(lint_attr_name, span)));
                }
            }
        }

        self.enter_attrs(attrs);
        f(self);
        self.exit_attrs(attrs);

        // rollback
        let cx = self.lint_sess_mut();
        for _ in 0..pushed {
            let (lint, lvlsrc) = cx.stack.pop().unwrap();
            cx.levels.set(lint, lvlsrc);
        }
    }
}


impl<'a> EarlyContext<'a> {
    fn new(sess: &'a Session,
           krate: &'a ast::Crate) -> EarlyContext<'a> {
        EarlyContext {
            sess: sess,
            krate: krate,
            lint_sess: LintSession::new(&sess.lint_store),
        }
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

    fn enter_attrs(&mut self, attrs: &'tcx [ast::Attribute]) {
        debug!("late context: enter_attrs({:?})", attrs);
        run_lints!(self, enter_lint_attrs, late_passes, attrs);
    }

    fn exit_attrs(&mut self, attrs: &'tcx [ast::Attribute]) {
        debug!("late context: exit_attrs({:?})", attrs);
        run_lints!(self, exit_lint_attrs, late_passes, attrs);
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

    fn enter_attrs(&mut self, attrs: &'a [ast::Attribute]) {
        debug!("early context: enter_attrs({:?})", attrs);
        run_lints!(self, enter_lint_attrs, early_passes, attrs);
    }

    fn exit_attrs(&mut self, attrs: &'a [ast::Attribute]) {
        debug!("early context: exit_attrs({:?})", attrs);
        run_lints!(self, exit_lint_attrs, early_passes, attrs);
    }
}

impl<'a, 'tcx> LateContext<'a, 'tcx> {
    fn with_param_env<F>(&mut self, id: ast::NodeId, f: F)
        where F: FnOnce(&mut Self),
    {
        let old_param_env = self.param_env;
        self.param_env = self.tcx.param_env(self.tcx.hir.local_def_id(id));
        f(self);
        self.param_env = old_param_env;
    }
}

impl<'a, 'tcx> hir_visit::Visitor<'tcx> for LateContext<'a, 'tcx> {
    /// Because lints are scoped lexically, we want to walk nested
    /// items in the context of the outer item, so enable
    /// deep-walking.
    fn nested_visit_map<'this>(&'this mut self) -> hir_visit::NestedVisitorMap<'this, 'tcx> {
        hir_visit::NestedVisitorMap::All(&self.tcx.hir)
    }

    // Output any lints that were previously added to the session.
    fn visit_id(&mut self, id: ast::NodeId) {
        let lints = self.sess().lints.borrow_mut().take(id);
        for early_lint in lints.iter().chain(self.tables.lints.get(id)) {
            debug!("LateContext::visit_id: id={:?} early_lint={:?}", id, early_lint);
            self.early_lint(early_lint);
        }
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let old_tables = self.tables;
        self.tables = self.tcx.body_tables(body);
        let body = self.tcx.hir.body(body);
        self.visit_body(body);
        self.tables = old_tables;
    }

    fn visit_body(&mut self, body: &'tcx hir::Body) {
        run_lints!(self, check_body, late_passes, body);
        hir_visit::walk_body(self, body);
        run_lints!(self, check_body_post, late_passes, body);
    }

    fn visit_item(&mut self, it: &'tcx hir::Item) {
        self.with_lint_attrs(&it.attrs, |cx| {
            cx.with_param_env(it.id, |cx| {
                run_lints!(cx, check_item, late_passes, it);
                hir_visit::walk_item(cx, it);
                run_lints!(cx, check_item_post, late_passes, it);
            });
        })
    }

    fn visit_foreign_item(&mut self, it: &'tcx hir::ForeignItem) {
        self.with_lint_attrs(&it.attrs, |cx| {
            cx.with_param_env(it.id, |cx| {
                run_lints!(cx, check_foreign_item, late_passes, it);
                hir_visit::walk_foreign_item(cx, it);
                run_lints!(cx, check_foreign_item_post, late_passes, it);
            });
        })
    }

    fn visit_pat(&mut self, p: &'tcx hir::Pat) {
        run_lints!(self, check_pat, late_passes, p);
        hir_visit::walk_pat(self, p);
    }

    fn visit_expr(&mut self, e: &'tcx hir::Expr) {
        self.with_lint_attrs(&e.attrs, |cx| {
            run_lints!(cx, check_expr, late_passes, e);
            hir_visit::walk_expr(cx, e);
            run_lints!(cx, check_expr_post, late_passes, e);
        })
    }

    fn visit_stmt(&mut self, s: &'tcx hir::Stmt) {
        // statement attributes are actually just attributes on one of
        // - item
        // - local
        // - expression
        // so we keep track of lint levels there
        run_lints!(self, check_stmt, late_passes, s);
        hir_visit::walk_stmt(self, s);
    }

    fn visit_fn(&mut self, fk: hir_visit::FnKind<'tcx>, decl: &'tcx hir::FnDecl,
                body_id: hir::BodyId, span: Span, id: ast::NodeId) {
        // Wrap in tables here, not just in visit_nested_body,
        // in order for `check_fn` to be able to use them.
        let old_tables = self.tables;
        self.tables = self.tcx.body_tables(body_id);
        let body = self.tcx.hir.body(body_id);
        run_lints!(self, check_fn, late_passes, fk, decl, body, span, id);
        hir_visit::walk_fn(self, fk, decl, body_id, span, id);
        run_lints!(self, check_fn_post, late_passes, fk, decl, body, span, id);
        self.tables = old_tables;
    }

    fn visit_variant_data(&mut self,
                        s: &'tcx hir::VariantData,
                        name: ast::Name,
                        g: &'tcx hir::Generics,
                        item_id: ast::NodeId,
                        _: Span) {
        run_lints!(self, check_struct_def, late_passes, s, name, g, item_id);
        hir_visit::walk_struct_def(self, s);
        run_lints!(self, check_struct_def_post, late_passes, s, name, g, item_id);
    }

    fn visit_struct_field(&mut self, s: &'tcx hir::StructField) {
        self.with_lint_attrs(&s.attrs, |cx| {
            run_lints!(cx, check_struct_field, late_passes, s);
            hir_visit::walk_struct_field(cx, s);
        })
    }

    fn visit_variant(&mut self,
                     v: &'tcx hir::Variant,
                     g: &'tcx hir::Generics,
                     item_id: ast::NodeId) {
        self.with_lint_attrs(&v.node.attrs, |cx| {
            run_lints!(cx, check_variant, late_passes, v, g);
            hir_visit::walk_variant(cx, v, g, item_id);
            run_lints!(cx, check_variant_post, late_passes, v, g);
        })
    }

    fn visit_ty(&mut self, t: &'tcx hir::Ty) {
        run_lints!(self, check_ty, late_passes, t);
        hir_visit::walk_ty(self, t);
    }

    fn visit_name(&mut self, sp: Span, name: ast::Name) {
        run_lints!(self, check_name, late_passes, sp, name);
    }

    fn visit_mod(&mut self, m: &'tcx hir::Mod, s: Span, n: ast::NodeId) {
        run_lints!(self, check_mod, late_passes, m, s, n);
        hir_visit::walk_mod(self, m, n);
        run_lints!(self, check_mod_post, late_passes, m, s, n);
    }

    fn visit_local(&mut self, l: &'tcx hir::Local) {
        self.with_lint_attrs(&l.attrs, |cx| {
            run_lints!(cx, check_local, late_passes, l);
            hir_visit::walk_local(cx, l);
        })
    }

    fn visit_block(&mut self, b: &'tcx hir::Block) {
        run_lints!(self, check_block, late_passes, b);
        hir_visit::walk_block(self, b);
        run_lints!(self, check_block_post, late_passes, b);
    }

    fn visit_arm(&mut self, a: &'tcx hir::Arm) {
        run_lints!(self, check_arm, late_passes, a);
        hir_visit::walk_arm(self, a);
    }

    fn visit_decl(&mut self, d: &'tcx hir::Decl) {
        run_lints!(self, check_decl, late_passes, d);
        hir_visit::walk_decl(self, d);
    }

    fn visit_generics(&mut self, g: &'tcx hir::Generics) {
        run_lints!(self, check_generics, late_passes, g);
        hir_visit::walk_generics(self, g);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {
        self.with_lint_attrs(&trait_item.attrs, |cx| {
            cx.with_param_env(trait_item.id, |cx| {
                run_lints!(cx, check_trait_item, late_passes, trait_item);
                hir_visit::walk_trait_item(cx, trait_item);
                run_lints!(cx, check_trait_item_post, late_passes, trait_item);
            });
        });
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        self.with_lint_attrs(&impl_item.attrs, |cx| {
            cx.with_param_env(impl_item.id, |cx| {
                run_lints!(cx, check_impl_item, late_passes, impl_item);
                hir_visit::walk_impl_item(cx, impl_item);
                run_lints!(cx, check_impl_item_post, late_passes, impl_item);
            });
        });
    }

    fn visit_lifetime(&mut self, lt: &'tcx hir::Lifetime) {
        run_lints!(self, check_lifetime, late_passes, lt);
        hir_visit::walk_lifetime(self, lt);
    }

    fn visit_lifetime_def(&mut self, lt: &'tcx hir::LifetimeDef) {
        run_lints!(self, check_lifetime_def, late_passes, lt);
        hir_visit::walk_lifetime_def(self, lt);
    }

    fn visit_path(&mut self, p: &'tcx hir::Path, id: ast::NodeId) {
        run_lints!(self, check_path, late_passes, p, id);
        hir_visit::walk_path(self, p);
    }

    fn visit_attribute(&mut self, attr: &'tcx ast::Attribute) {
        check_lint_name_attribute(self, attr);
        run_lints!(self, check_attribute, late_passes, attr);
    }
}

impl<'a> ast_visit::Visitor<'a> for EarlyContext<'a> {
    fn visit_item(&mut self, it: &'a ast::Item) {
        self.with_lint_attrs(&it.attrs, |cx| {
            run_lints!(cx, check_item, early_passes, it);
            ast_visit::walk_item(cx, it);
            run_lints!(cx, check_item_post, early_passes, it);
        })
    }

    fn visit_foreign_item(&mut self, it: &'a ast::ForeignItem) {
        self.with_lint_attrs(&it.attrs, |cx| {
            run_lints!(cx, check_foreign_item, early_passes, it);
            ast_visit::walk_foreign_item(cx, it);
            run_lints!(cx, check_foreign_item_post, early_passes, it);
        })
    }

    fn visit_pat(&mut self, p: &'a ast::Pat) {
        run_lints!(self, check_pat, early_passes, p);
        ast_visit::walk_pat(self, p);
    }

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        self.with_lint_attrs(&e.attrs, |cx| {
            run_lints!(cx, check_expr, early_passes, e);
            ast_visit::walk_expr(cx, e);
        })
    }

    fn visit_stmt(&mut self, s: &'a ast::Stmt) {
        run_lints!(self, check_stmt, early_passes, s);
        ast_visit::walk_stmt(self, s);
    }

    fn visit_fn(&mut self, fk: ast_visit::FnKind<'a>, decl: &'a ast::FnDecl,
                span: Span, id: ast::NodeId) {
        run_lints!(self, check_fn, early_passes, fk, decl, span, id);
        ast_visit::walk_fn(self, fk, decl, span);
        run_lints!(self, check_fn_post, early_passes, fk, decl, span, id);
    }

    fn visit_variant_data(&mut self,
                        s: &'a ast::VariantData,
                        ident: ast::Ident,
                        g: &'a ast::Generics,
                        item_id: ast::NodeId,
                        _: Span) {
        run_lints!(self, check_struct_def, early_passes, s, ident, g, item_id);
        ast_visit::walk_struct_def(self, s);
        run_lints!(self, check_struct_def_post, early_passes, s, ident, g, item_id);
    }

    fn visit_struct_field(&mut self, s: &'a ast::StructField) {
        self.with_lint_attrs(&s.attrs, |cx| {
            run_lints!(cx, check_struct_field, early_passes, s);
            ast_visit::walk_struct_field(cx, s);
        })
    }

    fn visit_variant(&mut self, v: &'a ast::Variant, g: &'a ast::Generics, item_id: ast::NodeId) {
        self.with_lint_attrs(&v.node.attrs, |cx| {
            run_lints!(cx, check_variant, early_passes, v, g);
            ast_visit::walk_variant(cx, v, g, item_id);
            run_lints!(cx, check_variant_post, early_passes, v, g);
        })
    }

    fn visit_ty(&mut self, t: &'a ast::Ty) {
        run_lints!(self, check_ty, early_passes, t);
        ast_visit::walk_ty(self, t);
    }

    fn visit_ident(&mut self, sp: Span, id: ast::Ident) {
        run_lints!(self, check_ident, early_passes, sp, id);
    }

    fn visit_mod(&mut self, m: &'a ast::Mod, s: Span, _a: &[ast::Attribute], n: ast::NodeId) {
        run_lints!(self, check_mod, early_passes, m, s, n);
        ast_visit::walk_mod(self, m);
        run_lints!(self, check_mod_post, early_passes, m, s, n);
    }

    fn visit_local(&mut self, l: &'a ast::Local) {
        self.with_lint_attrs(&l.attrs, |cx| {
            run_lints!(cx, check_local, early_passes, l);
            ast_visit::walk_local(cx, l);
        })
    }

    fn visit_block(&mut self, b: &'a ast::Block) {
        run_lints!(self, check_block, early_passes, b);
        ast_visit::walk_block(self, b);
        run_lints!(self, check_block_post, early_passes, b);
    }

    fn visit_arm(&mut self, a: &'a ast::Arm) {
        run_lints!(self, check_arm, early_passes, a);
        ast_visit::walk_arm(self, a);
    }

    fn visit_expr_post(&mut self, e: &'a ast::Expr) {
        run_lints!(self, check_expr_post, early_passes, e);
    }

    fn visit_generics(&mut self, g: &'a ast::Generics) {
        run_lints!(self, check_generics, early_passes, g);
        ast_visit::walk_generics(self, g);
    }

    fn visit_trait_item(&mut self, trait_item: &'a ast::TraitItem) {
        self.with_lint_attrs(&trait_item.attrs, |cx| {
            run_lints!(cx, check_trait_item, early_passes, trait_item);
            ast_visit::walk_trait_item(cx, trait_item);
            run_lints!(cx, check_trait_item_post, early_passes, trait_item);
        });
    }

    fn visit_impl_item(&mut self, impl_item: &'a ast::ImplItem) {
        self.with_lint_attrs(&impl_item.attrs, |cx| {
            run_lints!(cx, check_impl_item, early_passes, impl_item);
            ast_visit::walk_impl_item(cx, impl_item);
            run_lints!(cx, check_impl_item_post, early_passes, impl_item);
        });
    }

    fn visit_lifetime(&mut self, lt: &'a ast::Lifetime) {
        run_lints!(self, check_lifetime, early_passes, lt);
    }

    fn visit_lifetime_def(&mut self, lt: &'a ast::LifetimeDef) {
        run_lints!(self, check_lifetime_def, early_passes, lt);
    }

    fn visit_path(&mut self, p: &'a ast::Path, id: ast::NodeId) {
        run_lints!(self, check_path, early_passes, p, id);
        ast_visit::walk_path(self, p);
    }

    fn visit_path_list_item(&mut self, prefix: &'a ast::Path, item: &'a ast::PathListItem) {
        run_lints!(self, check_path_list_item, early_passes, item);
        ast_visit::walk_path_list_item(self, prefix, item);
    }

    fn visit_attribute(&mut self, attr: &'a ast::Attribute) {
        run_lints!(self, check_attribute, early_passes, attr);
    }

    fn visit_mac_def(&mut self, _mac: &'a ast::MacroDef, id: ast::NodeId) {
        let lints = self.sess.lints.borrow_mut().take(id);
        for early_lint in lints {
            self.early_lint(&early_lint);
        }
    }
}

enum CheckLintNameResult {
    Ok,
    // Lint doesn't exist
    NoLint,
    // The lint is either renamed or removed. This is the warning
    // message.
    Warning(String),
}

/// Checks the name of a lint for its existence, and whether it was
/// renamed or removed. Generates a DiagnosticBuilder containing a
/// warning for renamed and removed lints. This is over both lint
/// names from attributes and those passed on the command line. Since
/// it emits non-fatal warnings and there are *two* lint passes that
/// inspect attributes, this is only run from the late pass to avoid
/// printing duplicate warnings.
fn check_lint_name(lint_cx: &LintStore,
                   lint_name: &str) -> CheckLintNameResult {
    match lint_cx.by_name.get(lint_name) {
        Some(&Renamed(ref new_name, _)) => {
            CheckLintNameResult::Warning(
                format!("lint {} has been renamed to {}", lint_name, new_name)
            )
        },
        Some(&Removed(ref reason)) => {
            CheckLintNameResult::Warning(
                format!("lint {} has been removed: {}", lint_name, reason)
            )
        },
        None => {
            match lint_cx.lint_groups.get(lint_name) {
                None => {
                    CheckLintNameResult::NoLint
                }
                Some(_) => {
                    /* lint group exists */
                    CheckLintNameResult::Ok
                }
            }
        }
        Some(_) => {
            /* lint exists */
            CheckLintNameResult::Ok
        }
    }
}

// Checks the validity of lint names derived from attributes
fn check_lint_name_attribute(cx: &LateContext, attr: &ast::Attribute) {
    for result in gather_attr(attr) {
        match result {
            Err(_) => {
                // Malformed lint attr. Reported by with_lint_attrs
                continue;
            }
            Ok((lint_name, _, span)) => {
                match check_lint_name(&cx.lint_sess.lints, &lint_name.as_str()) {
                    CheckLintNameResult::Ok => (),
                    CheckLintNameResult::Warning(ref msg) => {
                        cx.span_lint(builtin::RENAMED_AND_REMOVED_LINTS,
                                     span, msg);
                    }
                    CheckLintNameResult::NoLint => {
                        cx.span_lint(builtin::UNKNOWN_LINTS, span,
                                     &format!("unknown lint: `{}`",
                                              lint_name));
                    }
                }
            }
        }
    }
}

// Checks the validity of lint names derived from the command line
fn check_lint_name_cmdline(sess: &Session, lint_cx: &LintStore,
                           lint_name: &str, level: Level) {
    let db = match check_lint_name(lint_cx, lint_name) {
        CheckLintNameResult::Ok => None,
        CheckLintNameResult::Warning(ref msg) => {
            Some(sess.struct_warn(msg))
        },
        CheckLintNameResult::NoLint => {
            Some(struct_err!(sess, E0602, "unknown lint: `{}`", lint_name))
        }
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


/// Perform lint checking on a crate.
///
/// Consumes the `lint_store` field of the `Session`.
pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let access_levels = &tcx.privacy_access_levels(LOCAL_CRATE);

    let krate = tcx.hir.krate();

    let mut cx = LateContext {
        tcx: tcx,
        tables: &ty::TypeckTables::empty(),
        param_env: ty::ParamEnv::empty(Reveal::UserFacing),
        access_levels: access_levels,
        lint_sess: LintSession::new(&tcx.sess.lint_store),
    };

    // Visit the whole crate.
    cx.with_lint_attrs(&krate.attrs, |cx| {
        // since the root module isn't visited as an item (because it isn't an
        // item), warn for it here.
        run_lints!(cx, check_crate, late_passes, krate);

        hir_visit::walk_crate(cx, krate);

        run_lints!(cx, check_crate_post, late_passes, krate);
    });

    // If we missed any lints added to the session, then there's a bug somewhere
    // in the iteration code.
    if let Some((id, v)) = tcx.sess.lints.borrow().get_any() {
        for early_lint in v {
            span_bug!(early_lint.diagnostic.span.clone(),
                      "unprocessed lint {:?} at {}",
                      early_lint, tcx.hir.node_to_string(*id));
        }
    }

    // Put the lint store levels and passes back in the session.
    cx.lint_sess.restore(&tcx.sess.lint_store);
}

pub fn check_ast_crate(sess: &Session, krate: &ast::Crate) {
    let mut cx = EarlyContext::new(sess, krate);

    // Visit the whole crate.
    cx.with_lint_attrs(&krate.attrs, |cx| {
        // Lints may be assigned to the whole crate.
        let lints = cx.sess.lints.borrow_mut().take(ast::CRATE_NODE_ID);
        for early_lint in lints {
            cx.early_lint(&early_lint);
        }

        // since the root module isn't visited as an item (because it isn't an
        // item), warn for it here.
        run_lints!(cx, check_crate, early_passes, krate);

        ast_visit::walk_crate(cx, krate);

        run_lints!(cx, check_crate_post, early_passes, krate);
    });

    // Put the lint store levels and passes back in the session.
    cx.lint_sess.restore(&sess.lint_store);

    // If we missed any lints added to the session, then there's a bug somewhere
    // in the iteration code.
    for (_, v) in sess.lints.borrow().get_any() {
        for early_lint in v {
            span_bug!(early_lint.diagnostic.span.clone(), "unprocessed lint {:?}", early_lint);
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
            match tcx.sess.lint_store.borrow().find_lint(&s) {
                Ok(id) => Ok(id),
                Err(_) => panic!("invalid lint-id `{}`", s),
            }
        })
    }
}
