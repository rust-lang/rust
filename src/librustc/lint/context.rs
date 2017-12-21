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

use std::slice;
use lint::{EarlyLintPassObject, LateLintPassObject};
use lint::{Level, Lint, LintId, LintPass, LintBuffer};
use lint::levels::{LintLevelSets, LintLevelsBuilder};
use middle::privacy::AccessLevels;
use rustc_serialize::{Decoder, Decodable, Encoder, Encodable};
use session::{config, early_error, Session};
use traits::Reveal;
use ty::{self, TyCtxt, Ty};
use ty::layout::{LayoutError, LayoutOf, TyLayout};
use util::nodemap::FxHashMap;

use std::default::Default as StdDefault;
use std::cell::{Ref, RefCell};
use syntax::ast;
use syntax_pos::{MultiSpan, Span};
use errors::DiagnosticBuilder;
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

    /// Map of registered lint groups to what lints they expand to. The bool
    /// is true if the lint group was added by a plugin.
    lint_groups: FxHashMap<&'static str, (Vec<LintId>, bool)>,

    /// Extra info for future incompatibility lints, describing the
    /// issue or RFC that caused the incompatibility.
    future_incompatible: FxHashMap<LintId, FutureIncompatibleInfo>,
}

pub struct LintSession<'a, PassObject> {
    /// Reference to the store of registered lints.
    lints: Ref<'a, LintStore>,

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
}

/// Extra information for a future incompatibility lint. See the call
/// to `register_future_incompatible` in `librustc_lint/lib.rs` for
/// guidelines.
pub struct FutureIncompatibleInfo {
    pub id: LintId,
    pub reference: &'static str // e.g., a URL for an issue/PR/RFC or error code
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

pub enum CheckLintNameResult<'a> {
    Ok(&'a [LintId]),
    /// Lint doesn't exist
    NoLint,
    /// The lint is either renamed or removed. This is the warning
    /// message.
    Warning(String),
}

impl LintStore {
    pub fn new() -> LintStore {
        LintStore {
            lints: vec![],
            early_passes: Some(vec![]),
            late_passes: Some(vec![]),
            by_name: FxHashMap(),
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

    pub fn find_lints(&self, lint_name: &str) -> Result<Vec<LintId>, FindLintError> {
        match self.by_name.get(lint_name) {
            Some(&Id(lint_id)) => Ok(vec![lint_id]),
            Some(&Renamed(_, lint_id)) => {
                Ok(vec![lint_id])
            },
            Some(&Removed(_)) => {
                Err(FindLintError::Removed)
            },
            None => {
                match self.lint_groups.get(lint_name) {
                    Some(v) => Ok(v.0.clone()),
                    None => Err(FindLintError::Removed)
                }
            }
        }
    }

    /// Checks the validity of lint names derived from the command line
    pub fn check_lint_name_cmdline(&self,
                                   sess: &Session,
                                   lint_name: &str,
                                   level: Level) {
        let db = match self.check_lint_name(lint_name) {
            CheckLintNameResult::Ok(_) => None,
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

    /// Checks the name of a lint for its existence, and whether it was
    /// renamed or removed. Generates a DiagnosticBuilder containing a
    /// warning for renamed and removed lints. This is over both lint
    /// names from attributes and those passed on the command line. Since
    /// it emits non-fatal warnings and there are *two* lint passes that
    /// inspect attributes, this is only run from the late pass to avoid
    /// printing duplicate warnings.
    pub fn check_lint_name(&self, lint_name: &str) -> CheckLintNameResult {
        match self.by_name.get(lint_name) {
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
                match self.lint_groups.get(lint_name) {
                    None => CheckLintNameResult::NoLint,
                    Some(ids) => CheckLintNameResult::Ok(&ids.0),
                }
            }
            Some(&Id(ref id)) => CheckLintNameResult::Ok(slice::from_ref(id)),
        }
    }
}

impl<'a, PassObject: LintPassObject> LintSession<'a, PassObject> {
    /// Creates a new `LintSession`, by moving out the `LintStore`'s initial
    /// lint levels and pass objects. These can be restored using the `restore`
    /// method.
    fn new(store: &'a RefCell<LintStore>) -> LintSession<'a, PassObject> {
        let mut s = store.borrow_mut();
        let passes = PassObject::take_passes(&mut *s);
        drop(s);
        LintSession {
            lints: store.borrow(),
            passes,
        }
    }

    /// Restores the levels back to the original lint store.
    fn restore(self, store: &RefCell<LintStore>) {
        drop(self.lints);
        let mut s = store.borrow_mut();
        PassObject::restore_passes(&mut *s, self.passes);
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

    fn lookup_and_emit<S: Into<MultiSpan>>(&self,
                                           lint: &'static Lint,
                                           span: Option<S>,
                                           msg: &str) {
        self.lookup(lint, span, msg).emit();
    }

    fn lookup<S: Into<MultiSpan>>(&self,
                                  lint: &'static Lint,
                                  span: Option<S>,
                                  msg: &str)
                                  -> DiagnosticBuilder;

    /// Emit a lint at the appropriate level, for a particular span.
    fn span_lint<S: Into<MultiSpan>>(&self, lint: &'static Lint, span: S, msg: &str) {
        self.lookup_and_emit(lint, Some(span), msg);
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

    /// Merge the lints specified by any lint attributes into the
    /// current lint context, call the provided function, then reset the
    /// lints in effect to their previous state.
    fn with_lint_attrs<F>(&mut self,
                          id: ast::NodeId,
                          attrs: &'tcx [ast::Attribute],
                          f: F)
        where F: FnOnce(&mut Self);
}


impl<'a> EarlyContext<'a> {
    fn new(sess: &'a Session,
           krate: &'a ast::Crate) -> EarlyContext<'a> {
        EarlyContext {
            sess,
            krate,
            lint_sess: LintSession::new(&sess.lint_store),
            builder: LintLevelSets::builder(sess),
            buffered: sess.buffered_lints.borrow_mut().take().unwrap(),
        }
    }

    fn check_id(&mut self, id: ast::NodeId) {
        for early_lint in self.buffered.take(id) {
            self.lookup_and_emit(early_lint.lint_id.lint,
                                 Some(early_lint.span.clone()),
                                 &early_lint.msg);
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

    fn lookup<S: Into<MultiSpan>>(&self,
                                  lint: &'static Lint,
                                  span: Option<S>,
                                  msg: &str)
                                  -> DiagnosticBuilder {
        let id = self.last_ast_node_with_lint_attrs;
        match span {
            Some(s) => self.tcx.struct_span_lint_node(lint, id, s, msg),
            None => self.tcx.struct_lint_node(lint, id, msg),
        }
    }

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

    fn lookup<S: Into<MultiSpan>>(&self,
                                  lint: &'static Lint,
                                  span: Option<S>,
                                  msg: &str)
                                  -> DiagnosticBuilder {
        self.builder.struct_lint(lint, span.map(|s| s.into()), msg)
    }

    fn with_lint_attrs<F>(&mut self,
                          id: ast::NodeId,
                          attrs: &'a [ast::Attribute],
                          f: F)
        where F: FnOnce(&mut Self)
    {
        let push = self.builder.push(attrs);
        self.check_id(id);
        self.enter_attrs(attrs);
        f(self);
        self.exit_attrs(attrs);
        self.builder.pop(push);
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

impl<'a, 'tcx> LayoutOf<Ty<'tcx>> for &'a LateContext<'a, 'tcx> {
    type TyLayout = Result<TyLayout<'tcx>, LayoutError<'tcx>>;

    fn layout_of(self, ty: Ty<'tcx>) -> Self::TyLayout {
        (self.tcx, self.param_env.reveal_all()).layout_of(ty)
    }
}

impl<'a, 'tcx> hir_visit::Visitor<'tcx> for LateContext<'a, 'tcx> {
    /// Because lints are scoped lexically, we want to walk nested
    /// items in the context of the outer item, so enable
    /// deep-walking.
    fn nested_visit_map<'this>(&'this mut self) -> hir_visit::NestedVisitorMap<'this, 'tcx> {
        hir_visit::NestedVisitorMap::All(&self.tcx.hir)
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
        let generics = self.generics.take();
        self.generics = it.node.generics();
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            cx.with_param_env(it.id, |cx| {
                run_lints!(cx, check_item, late_passes, it);
                hir_visit::walk_item(cx, it);
                run_lints!(cx, check_item_post, late_passes, it);
            });
        });
        self.generics = generics;
    }

    fn visit_foreign_item(&mut self, it: &'tcx hir::ForeignItem) {
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
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
        self.with_lint_attrs(e.id, &e.attrs, |cx| {
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
        self.with_lint_attrs(s.id, &s.attrs, |cx| {
            run_lints!(cx, check_struct_field, late_passes, s);
            hir_visit::walk_struct_field(cx, s);
        })
    }

    fn visit_variant(&mut self,
                     v: &'tcx hir::Variant,
                     g: &'tcx hir::Generics,
                     item_id: ast::NodeId) {
        self.with_lint_attrs(v.node.data.id(), &v.node.attrs, |cx| {
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
        self.with_lint_attrs(l.id, &l.attrs, |cx| {
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

    fn visit_generic_param(&mut self, p: &'tcx hir::GenericParam) {
        run_lints!(self, check_generic_param, late_passes, p);
        hir_visit::walk_generic_param(self, p);
    }

    fn visit_generics(&mut self, g: &'tcx hir::Generics) {
        run_lints!(self, check_generics, late_passes, g);
        hir_visit::walk_generics(self, g);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {
        let generics = self.generics.take();
        self.generics = Some(&trait_item.generics);
        self.with_lint_attrs(trait_item.id, &trait_item.attrs, |cx| {
            cx.with_param_env(trait_item.id, |cx| {
                run_lints!(cx, check_trait_item, late_passes, trait_item);
                hir_visit::walk_trait_item(cx, trait_item);
                run_lints!(cx, check_trait_item_post, late_passes, trait_item);
            });
        });
        self.generics = generics;
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        let generics = self.generics.take();
        self.generics = Some(&impl_item.generics);
        self.with_lint_attrs(impl_item.id, &impl_item.attrs, |cx| {
            cx.with_param_env(impl_item.id, |cx| {
                run_lints!(cx, check_impl_item, late_passes, impl_item);
                hir_visit::walk_impl_item(cx, impl_item);
                run_lints!(cx, check_impl_item_post, late_passes, impl_item);
            });
        });
        self.generics = generics;
    }

    fn visit_lifetime(&mut self, lt: &'tcx hir::Lifetime) {
        run_lints!(self, check_lifetime, late_passes, lt);
        hir_visit::walk_lifetime(self, lt);
    }

    fn visit_path(&mut self, p: &'tcx hir::Path, id: ast::NodeId) {
        run_lints!(self, check_path, late_passes, p, id);
        hir_visit::walk_path(self, p);
    }

    fn visit_attribute(&mut self, attr: &'tcx ast::Attribute) {
        run_lints!(self, check_attribute, late_passes, attr);
    }
}

impl<'a> ast_visit::Visitor<'a> for EarlyContext<'a> {
    fn visit_item(&mut self, it: &'a ast::Item) {
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            run_lints!(cx, check_item, early_passes, it);
            ast_visit::walk_item(cx, it);
            run_lints!(cx, check_item_post, early_passes, it);
        })
    }

    fn visit_foreign_item(&mut self, it: &'a ast::ForeignItem) {
        self.with_lint_attrs(it.id, &it.attrs, |cx| {
            run_lints!(cx, check_foreign_item, early_passes, it);
            ast_visit::walk_foreign_item(cx, it);
            run_lints!(cx, check_foreign_item_post, early_passes, it);
        })
    }

    fn visit_pat(&mut self, p: &'a ast::Pat) {
        run_lints!(self, check_pat, early_passes, p);
        self.check_id(p.id);
        ast_visit::walk_pat(self, p);
    }

    fn visit_expr(&mut self, e: &'a ast::Expr) {
        self.with_lint_attrs(e.id, &e.attrs, |cx| {
            run_lints!(cx, check_expr, early_passes, e);
            ast_visit::walk_expr(cx, e);
        })
    }

    fn visit_stmt(&mut self, s: &'a ast::Stmt) {
        run_lints!(self, check_stmt, early_passes, s);
        self.check_id(s.id);
        ast_visit::walk_stmt(self, s);
    }

    fn visit_fn(&mut self, fk: ast_visit::FnKind<'a>, decl: &'a ast::FnDecl,
                span: Span, id: ast::NodeId) {
        run_lints!(self, check_fn, early_passes, fk, decl, span, id);
        self.check_id(id);
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
        self.check_id(s.id());
        ast_visit::walk_struct_def(self, s);
        run_lints!(self, check_struct_def_post, early_passes, s, ident, g, item_id);
    }

    fn visit_struct_field(&mut self, s: &'a ast::StructField) {
        self.with_lint_attrs(s.id, &s.attrs, |cx| {
            run_lints!(cx, check_struct_field, early_passes, s);
            ast_visit::walk_struct_field(cx, s);
        })
    }

    fn visit_variant(&mut self, v: &'a ast::Variant, g: &'a ast::Generics, item_id: ast::NodeId) {
        self.with_lint_attrs(item_id, &v.node.attrs, |cx| {
            run_lints!(cx, check_variant, early_passes, v, g);
            ast_visit::walk_variant(cx, v, g, item_id);
            run_lints!(cx, check_variant_post, early_passes, v, g);
        })
    }

    fn visit_ty(&mut self, t: &'a ast::Ty) {
        run_lints!(self, check_ty, early_passes, t);
        self.check_id(t.id);
        ast_visit::walk_ty(self, t);
    }

    fn visit_ident(&mut self, sp: Span, id: ast::Ident) {
        run_lints!(self, check_ident, early_passes, sp, id);
    }

    fn visit_mod(&mut self, m: &'a ast::Mod, s: Span, _a: &[ast::Attribute], n: ast::NodeId) {
        run_lints!(self, check_mod, early_passes, m, s, n);
        self.check_id(n);
        ast_visit::walk_mod(self, m);
        run_lints!(self, check_mod_post, early_passes, m, s, n);
    }

    fn visit_local(&mut self, l: &'a ast::Local) {
        self.with_lint_attrs(l.id, &l.attrs, |cx| {
            run_lints!(cx, check_local, early_passes, l);
            ast_visit::walk_local(cx, l);
        })
    }

    fn visit_block(&mut self, b: &'a ast::Block) {
        run_lints!(self, check_block, early_passes, b);
        self.check_id(b.id);
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

    fn visit_generic_param(&mut self, param: &'a ast::GenericParam) {
        run_lints!(self, check_generic_param, early_passes, param);
        ast_visit::walk_generic_param(self, param);
    }

    fn visit_generics(&mut self, g: &'a ast::Generics) {
        run_lints!(self, check_generics, early_passes, g);
        ast_visit::walk_generics(self, g);
    }

    fn visit_trait_item(&mut self, trait_item: &'a ast::TraitItem) {
        self.with_lint_attrs(trait_item.id, &trait_item.attrs, |cx| {
            run_lints!(cx, check_trait_item, early_passes, trait_item);
            ast_visit::walk_trait_item(cx, trait_item);
            run_lints!(cx, check_trait_item_post, early_passes, trait_item);
        });
    }

    fn visit_impl_item(&mut self, impl_item: &'a ast::ImplItem) {
        self.with_lint_attrs(impl_item.id, &impl_item.attrs, |cx| {
            run_lints!(cx, check_impl_item, early_passes, impl_item);
            ast_visit::walk_impl_item(cx, impl_item);
            run_lints!(cx, check_impl_item_post, early_passes, impl_item);
        });
    }

    fn visit_lifetime(&mut self, lt: &'a ast::Lifetime) {
        run_lints!(self, check_lifetime, early_passes, lt);
        self.check_id(lt.id);
    }

    fn visit_path(&mut self, p: &'a ast::Path, id: ast::NodeId) {
        run_lints!(self, check_path, early_passes, p, id);
        self.check_id(id);
        ast_visit::walk_path(self, p);
    }

    fn visit_attribute(&mut self, attr: &'a ast::Attribute) {
        run_lints!(self, check_attribute, early_passes, attr);
    }

    fn visit_mac_def(&mut self, _mac: &'a ast::MacroDef, id: ast::NodeId) {
        self.check_id(id);
    }
}


/// Perform lint checking on a crate.
///
/// Consumes the `lint_store` field of the `Session`.
pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let access_levels = &tcx.privacy_access_levels(LOCAL_CRATE);

    let krate = tcx.hir.krate();

    let mut cx = LateContext {
        tcx,
        tables: &ty::TypeckTables::empty(None),
        param_env: ty::ParamEnv::empty(Reveal::UserFacing),
        access_levels,
        lint_sess: LintSession::new(&tcx.sess.lint_store),
        last_ast_node_with_lint_attrs: ast::CRATE_NODE_ID,
        generics: None,
    };

    // Visit the whole crate.
    cx.with_lint_attrs(ast::CRATE_NODE_ID, &krate.attrs, |cx| {
        // since the root module isn't visited as an item (because it isn't an
        // item), warn for it here.
        run_lints!(cx, check_crate, late_passes, krate);

        hir_visit::walk_crate(cx, krate);

        run_lints!(cx, check_crate_post, late_passes, krate);
    });

    // Put the lint store levels and passes back in the session.
    cx.lint_sess.restore(&tcx.sess.lint_store);
}

pub fn check_ast_crate(sess: &Session, krate: &ast::Crate) {
    let mut cx = EarlyContext::new(sess, krate);

    // Visit the whole crate.
    cx.with_lint_attrs(ast::CRATE_NODE_ID, &krate.attrs, |cx| {
        // since the root module isn't visited as an item (because it isn't an
        // item), warn for it here.
        run_lints!(cx, check_crate, early_passes, krate);

        ast_visit::walk_crate(cx, krate);

        run_lints!(cx, check_crate_post, early_passes, krate);
    });

    // Put the lint store levels and passes back in the session.
    cx.lint_sess.restore(&sess.lint_store);

    // Emit all buffered lints from early on in the session now that we've
    // calculated the lint levels for all AST nodes.
    for (_id, lints) in cx.buffered.map {
        for early_lint in lints {
            span_bug!(early_lint.span, "failed to process bufferd lint here");
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
