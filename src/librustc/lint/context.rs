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

use middle::privacy::ExportedItems;
use middle::ty::{self, Ty};
use session::{early_error, Session};
use lint::{Level, LevelSource, Lint, LintId, LintArray, LintPass};
use lint::{EarlyLintPass, EarlyLintPassObject, LateLintPass, LateLintPassObject};
use lint::{Default, CommandLine, Node, Allow, Warn, Deny, Forbid};
use lint::builtin;
use util::nodemap::FnvHashMap;

use std::cell::RefCell;
use std::cmp;
use std::default;
use std::mem;
use syntax::ast_util::{self, IdVisitingOperation};
use syntax::attr::{self, AttrMetaMethods};
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use syntax::ast;
use rustc_front::hir;
use rustc_front::util;
use rustc_front::visit as hir_visit;
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
    /// This is only `None` while iterating over the objects. See the definition
    /// of run_lints.
    early_passes: Option<Vec<EarlyLintPassObject>>,
    late_passes: Option<Vec<LateLintPassObject>>,

    /// Lints indexed by name.
    by_name: FnvHashMap<String, TargetLint>,

    /// Current levels of each lint, and where they were set.
    levels: FnvHashMap<LintId, LevelSource>,

    /// Map of registered lint groups to what lints they expand to. The bool
    /// is true if the lint group was added by a plugin.
    lint_groups: FnvHashMap<&'static str, (Vec<LintId>, bool)>,

    /// Maximum level a lint can be
    lint_cap: Option<Level>,
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
    Removed
}

impl LintStore {
    fn get_level_source(&self, lint: LintId) -> LevelSource {
        match self.levels.get(&lint) {
            Some(&s) => s,
            None => (Allow, Default),
        }
    }

    fn set_level(&mut self, lint: LintId, mut lvlsrc: LevelSource) {
        if let Some(cap) = self.lint_cap {
            lvlsrc.0 = cmp::min(lvlsrc.0, cap);
        }
        if lvlsrc.0 == Allow {
            self.levels.remove(&lint);
        } else {
            self.levels.insert(lint, lvlsrc);
        }
    }

    pub fn new() -> LintStore {
        LintStore {
            lints: vec!(),
            early_passes: Some(vec!()),
            late_passes: Some(vec!()),
            by_name: FnvHashMap(),
            levels: FnvHashMap(),
            lint_groups: FnvHashMap(),
            lint_cap: None,
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
                    (None, _) => early_error(default::Default::default(), &msg[..]),
                    (Some(sess), false) => sess.bug(&msg[..]),

                    // A duplicate name from a plugin is a user error.
                    (Some(sess), true)  => sess.err(&msg[..]),
                }
            }

            if lint.default_level != Allow {
                self.levels.insert(id, (lint.default_level, Default));
            }
        }
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
                (None, _) => early_error(default::Default::default(), &msg[..]),
                (Some(sess), false) => sess.bug(&msg[..]),

                // A duplicate name from a plugin is a user error.
                (Some(sess), true)  => sess.err(&msg[..]),
            }
        }
    }

    pub fn register_renamed(&mut self, old_name: &str, new_name: &str) {
        let target = match self.by_name.get(new_name) {
            Some(&Id(lint_id)) => lint_id.clone(),
            _ => panic!("invalid lint renaming of {} to {}", old_name, new_name)
        };
        self.by_name.insert(old_name.to_string(), Renamed(new_name.to_string(), target));
    }

    pub fn register_removed(&mut self, name: &str, reason: &str) {
        self.by_name.insert(name.into(), Removed(reason.into()));
    }

    #[allow(unused_variables)]
    fn find_lint(&self, lint_name: &str, sess: &Session, span: Option<Span>)
                 -> Result<LintId, FindLintError>
    {
        match self.by_name.get(lint_name) {
            Some(&Id(lint_id)) => Ok(lint_id),
            Some(&Renamed(ref new_name, lint_id)) => {
                let warning = format!("lint {} has been renamed to {}",
                                      lint_name, new_name);
                match span {
                    Some(span) => sess.span_warn(span, &warning[..]),
                    None => sess.warn(&warning[..]),
                };
                Ok(lint_id)
            },
            Some(&Removed(ref reason)) => {
                let warning = format!("lint {} has been removed: {}", lint_name, reason);
                match span {
                    Some(span) => sess.span_warn(span, &warning[..]),
                    None => sess.warn(&warning[..])
                }
                Err(FindLintError::Removed)
            },
            None => Err(FindLintError::NotFound)
        }
    }

    pub fn process_command_line(&mut self, sess: &Session) {
        for &(ref lint_name, level) in &sess.opts.lint_opts {
            match self.find_lint(&lint_name[..], sess, None) {
                Ok(lint_id) => self.set_level(lint_id, (level, CommandLine)),
                Err(_) => {
                    match self.lint_groups.iter().map(|(&x, pair)| (x, pair.0.clone()))
                                                 .collect::<FnvHashMap<&'static str,
                                                                       Vec<LintId>>>()
                                                 .get(&lint_name[..]) {
                        Some(v) => {
                            v.iter()
                             .map(|lint_id: &LintId|
                                     self.set_level(*lint_id, (level, CommandLine)))
                             .collect::<Vec<()>>();
                        }
                        None => sess.err(&format!("unknown {} flag: {}",
                                                 level.as_str(), lint_name)),
                    }
                }
            }
        }

        self.lint_cap = sess.opts.lint_cap;
        if let Some(cap) = self.lint_cap {
            for level in self.levels.iter_mut().map(|p| &mut (p.1).0) {
                *level = cmp::min(*level, cap);
            }
        }
    }
}

/// Context for lint checking after type checking.
pub struct LateContext<'a, 'tcx: 'a> {
    /// Type context we're checking in.
    pub tcx: &'a ty::ctxt<'tcx>,

    /// The crate being checked.
    pub krate: &'a hir::Crate,

    /// Items exported from the crate being checked.
    pub exported_items: &'a ExportedItems,

    /// The store of registered lints.
    lints: LintStore,

    /// When recursing into an attributed node of the ast which modifies lint
    /// levels, this stack keeps track of the previous lint levels of whatever
    /// was modified.
    level_stack: Vec<(LintId, LevelSource)>,

    /// Level of lints for certain NodeIds, stored here because the body of
    /// the lint needs to run in trans.
    node_levels: RefCell<FnvHashMap<(ast::NodeId, LintId), LevelSource>>,
}

/// Context for lint checking of the AST, after expansion, before lowering to
/// HIR.
pub struct EarlyContext<'a> {
    /// Type context we're checking in.
    pub sess: &'a Session,

    /// The crate being checked.
    pub krate: &'a ast::Crate,

    /// The store of registered lints.
    lints: LintStore,

    /// When recursing into an attributed node of the ast which modifies lint
    /// levels, this stack keeps track of the previous lint levels of whatever
    /// was modified.
    level_stack: Vec<(LintId, LevelSource)>,
}

/// Convenience macro for calling a `LintPass` method on every pass in the context.
macro_rules! run_lints { ($cx:expr, $f:ident, $ps:ident, $($args:expr),*) => ({
    // Move the vector of passes out of `$cx` so that we can
    // iterate over it mutably while passing `$cx` to the methods.
    let mut passes = $cx.mut_lints().$ps.take().unwrap();
    for obj in &mut passes {
        obj.$f($cx, $($args),*);
    }
    $cx.mut_lints().$ps = Some(passes);
}) }

/// Parse the lint attributes into a vector, with `Err`s for malformed lint
/// attributes. Writing this as an iterator is an enormous mess.
// See also the hir version just below.
pub fn gather_attrs(attrs: &[ast::Attribute])
                    -> Vec<Result<(InternedString, Level, Span), Span>> {
    let mut out = vec!();
    for attr in attrs {
        let level = match Level::from_str(&attr.name()) {
            None => continue,
            Some(lvl) => lvl,
        };

        attr::mark_used(attr);

        let meta = &attr.node.value;
        let metas = match meta.node {
            ast::MetaList(_, ref metas) => metas,
            _ => {
                out.push(Err(meta.span));
                continue;
            }
        };

        for meta in metas {
            out.push(match meta.node {
                ast::MetaWord(ref lint_name) => Ok((lint_name.clone(), level, meta.span)),
                _ => Err(meta.span),
            });
        }
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
pub fn raw_emit_lint(sess: &Session, lint: &'static Lint,
                     lvlsrc: LevelSource, span: Option<Span>, msg: &str) {
    let (mut level, source) = lvlsrc;
    if level == Allow { return }

    let name = lint.name_lower();
    let mut def = None;
    let msg = match source {
        Default => {
            format!("{}, #[{}({})] on by default", msg,
                    level.as_str(), name)
        },
        CommandLine => {
            format!("{} [-{} {}]", msg,
                    match level {
                        Warn => 'W', Deny => 'D', Forbid => 'F',
                        Allow => panic!()
                    }, name.replace("_", "-"))
        },
        Node(src) => {
            def = Some(src);
            msg.to_string()
        }
    };

    // For purposes of printing, we can treat forbid as deny.
    if level == Forbid { level = Deny; }

    match (level, span) {
        (Warn, Some(sp)) => sess.span_warn(sp, &msg[..]),
        (Warn, None)     => sess.warn(&msg[..]),
        (Deny, Some(sp)) => sess.span_err(sp, &msg[..]),
        (Deny, None)     => sess.err(&msg[..]),
        _ => sess.bug("impossible level in raw_emit_lint"),
    }

    if let Some(span) = def {
        sess.span_note(span, "lint level defined here");
    }
}

pub trait LintContext: Sized {
    fn sess(&self) -> &Session;
    fn lints(&self) -> &LintStore;
    fn mut_lints(&mut self) -> &mut LintStore;
    fn level_stack(&mut self) -> &mut Vec<(LintId, LevelSource)>;
    fn enter_attrs(&mut self, attrs: &[ast::Attribute]);
    fn exit_attrs(&mut self, attrs: &[ast::Attribute]);

    /// Get the level of `lint` at the current position of the lint
    /// traversal.
    fn current_level(&self, lint: &'static Lint) -> Level {
        self.lints().levels.get(&LintId::of(lint)).map_or(Allow, |&(lvl, _)| lvl)
    }

    fn lookup_and_emit(&self, lint: &'static Lint, span: Option<Span>, msg: &str) {
        let (level, src) = match self.lints().levels.get(&LintId::of(lint)) {
            None => return,
            Some(&(Warn, src)) => {
                let lint_id = LintId::of(builtin::WARNINGS);
                (self.lints().get_level_source(lint_id).0, src)
            }
            Some(&pair) => pair,
        };

        raw_emit_lint(&self.sess(), lint, (level, src), span, msg);
    }

    /// Emit a lint at the appropriate level, for a particular span.
    fn span_lint(&self, lint: &'static Lint, span: Span, msg: &str) {
        self.lookup_and_emit(lint, Some(span), msg);
    }

    /// Emit a lint and note at the appropriate level, for a particular span.
    fn span_lint_note(&self, lint: &'static Lint, span: Span, msg: &str,
                      note_span: Span, note: &str) {
        self.span_lint(lint, span, msg);
        if self.current_level(lint) != Level::Allow {
            if note_span == span {
                self.sess().fileline_note(note_span, note)
            } else {
                self.sess().span_note(note_span, note)
            }
        }
    }

    /// Emit a lint and help at the appropriate level, for a particular span.
    fn span_lint_help(&self, lint: &'static Lint, span: Span,
                      msg: &str, help: &str) {
        self.span_lint(lint, span, msg);
        if self.current_level(lint) != Level::Allow {
            self.sess().span_help(span, help)
        }
    }

    /// Emit a lint at the appropriate level, with no associated span.
    fn lint(&self, lint: &'static Lint, msg: &str) {
        self.lookup_and_emit(lint, None, msg);
    }

    /// Merge the lints specified by any lint attributes into the
    /// current lint context, call the provided function, then reset the
    /// lints in effect to their previous state.
    fn with_lint_attrs<F>(&mut self,
                          attrs: &[ast::Attribute],
                          f: F)
        where F: FnOnce(&mut Self),
    {
        // Parse all of the lint attributes, and then add them all to the
        // current dictionary of lint information. Along the way, keep a history
        // of what we changed so we can roll everything back after invoking the
        // specified closure
        let mut pushed = 0;

        for result in gather_attrs(attrs) {
            let v = match result {
                Err(span) => {
                    span_err!(self.sess(), span, E0452,
                              "malformed lint attribute");
                    continue;
                }
                Ok((lint_name, level, span)) => {
                    match self.lints().find_lint(&lint_name, &self.sess(), Some(span)) {
                        Ok(lint_id) => vec![(lint_id, level, span)],
                        Err(FindLintError::NotFound) => {
                            match self.lints().lint_groups.get(&lint_name[..]) {
                                Some(&(ref v, _)) => v.iter()
                                                      .map(|lint_id: &LintId|
                                                           (*lint_id, level, span))
                                                      .collect(),
                                None => {
                                    self.span_lint(builtin::UNKNOWN_LINTS, span,
                                                   &format!("unknown `{}` attribute: `{}`",
                                                            level.as_str(), lint_name));
                                    continue;
                                }
                            }
                        },
                        Err(FindLintError::Removed) => { continue; }
                    }
                }
            };

            for (lint_id, level, span) in v {
                let now = self.lints().get_level_source(lint_id).0;
                if now == Forbid && level != Forbid {
                    let lint_name = lint_id.as_str();
                    span_err!(self.sess(), span, E0453,
                              "{}({}) overruled by outer forbid({})",
                              level.as_str(), lint_name,
                              lint_name);
                } else if now != level {
                    let src = self.lints().get_level_source(lint_id).1;
                    self.level_stack().push((lint_id, (now, src)));
                    pushed += 1;
                    self.mut_lints().set_level(lint_id, (level, Node(span)));
                }
            }
        }

        self.enter_attrs(attrs);
        f(self);
        self.exit_attrs(attrs);

        // rollback
        for _ in 0..pushed {
            let (lint, lvlsrc) = self.level_stack().pop().unwrap();
            self.mut_lints().set_level(lint, lvlsrc);
        }
    }
}


impl<'a> EarlyContext<'a> {
    fn new(sess: &'a Session,
           krate: &'a ast::Crate) -> EarlyContext<'a> {
        // We want to own the lint store, so move it out of the session. Remember
        // to put it back later...
        let lint_store = mem::replace(&mut *sess.lint_store.borrow_mut(),
                                      LintStore::new());
        EarlyContext {
            sess: sess,
            krate: krate,
            lints: lint_store,
            level_stack: vec![],
        }
    }

    fn visit_ids<F>(&mut self, f: F)
        where F: FnOnce(&mut ast_util::IdVisitor<EarlyContext>)
    {
        let mut v = ast_util::IdVisitor {
            operation: self,
            pass_through_items: false,
            visited_outermost: false,
        };
        f(&mut v);
    }
}

impl<'a, 'tcx> LateContext<'a, 'tcx> {
    fn new(tcx: &'a ty::ctxt<'tcx>,
           krate: &'a hir::Crate,
           exported_items: &'a ExportedItems) -> LateContext<'a, 'tcx> {
        // We want to own the lint store, so move it out of the session.
        let lint_store = mem::replace(&mut *tcx.sess.lint_store.borrow_mut(),
                                      LintStore::new());

        LateContext {
            tcx: tcx,
            krate: krate,
            exported_items: exported_items,
            lints: lint_store,
            level_stack: vec![],
            node_levels: RefCell::new(FnvHashMap()),
        }
    }

    fn visit_ids<F>(&mut self, f: F)
        where F: FnOnce(&mut util::IdVisitor<LateContext>)
    {
        let mut v = util::IdVisitor {
            operation: self,
            pass_through_items: false,
            visited_outermost: false,
        };
        f(&mut v);
    }
}

impl<'a, 'tcx> LintContext for LateContext<'a, 'tcx> {
    /// Get the overall compiler `Session` object.
    fn sess(&self) -> &Session {
        &self.tcx.sess
    }

    fn lints(&self) -> &LintStore {
        &self.lints
    }

    fn mut_lints(&mut self) -> &mut LintStore {
        &mut self.lints
    }

    fn level_stack(&mut self) -> &mut Vec<(LintId, LevelSource)> {
        &mut self.level_stack
    }

    fn enter_attrs(&mut self, attrs: &[ast::Attribute]) {
        run_lints!(self, enter_lint_attrs, late_passes, attrs);
    }

    fn exit_attrs(&mut self, attrs: &[ast::Attribute]) {
        run_lints!(self, exit_lint_attrs, late_passes, attrs);
    }
}

impl<'a> LintContext for EarlyContext<'a> {
    /// Get the overall compiler `Session` object.
    fn sess(&self) -> &Session {
        &self.sess
    }

    fn lints(&self) -> &LintStore {
        &self.lints
    }

    fn mut_lints(&mut self) -> &mut LintStore {
        &mut self.lints
    }

    fn level_stack(&mut self) -> &mut Vec<(LintId, LevelSource)> {
        &mut self.level_stack
    }

    fn enter_attrs(&mut self, attrs: &[ast::Attribute]) {
        run_lints!(self, enter_lint_attrs, early_passes, attrs);
    }

    fn exit_attrs(&mut self, attrs: &[ast::Attribute]) {
        run_lints!(self, exit_lint_attrs, early_passes, attrs);
    }
}

impl<'a, 'tcx, 'v> hir_visit::Visitor<'v> for LateContext<'a, 'tcx> {
    fn visit_item(&mut self, it: &hir::Item) {
        self.with_lint_attrs(&it.attrs, |cx| {
            run_lints!(cx, check_item, late_passes, it);
            cx.visit_ids(|v| v.visit_item(it));
            hir_visit::walk_item(cx, it);
        })
    }

    fn visit_foreign_item(&mut self, it: &hir::ForeignItem) {
        self.with_lint_attrs(&it.attrs, |cx| {
            run_lints!(cx, check_foreign_item, late_passes, it);
            hir_visit::walk_foreign_item(cx, it);
        })
    }

    fn visit_pat(&mut self, p: &hir::Pat) {
        run_lints!(self, check_pat, late_passes, p);
        hir_visit::walk_pat(self, p);
    }

    fn visit_expr(&mut self, e: &hir::Expr) {
        run_lints!(self, check_expr, late_passes, e);
        hir_visit::walk_expr(self, e);
    }

    fn visit_stmt(&mut self, s: &hir::Stmt) {
        run_lints!(self, check_stmt, late_passes, s);
        hir_visit::walk_stmt(self, s);
    }

    fn visit_fn(&mut self, fk: hir_visit::FnKind<'v>, decl: &'v hir::FnDecl,
                body: &'v hir::Block, span: Span, id: ast::NodeId) {
        run_lints!(self, check_fn, late_passes, fk, decl, body, span, id);
        hir_visit::walk_fn(self, fk, decl, body, span);
    }

    fn visit_variant_data(&mut self,
                        s: &hir::VariantData,
                        name: ast::Name,
                        g: &hir::Generics,
                        item_id: ast::NodeId,
                        _: Span) {
        run_lints!(self, check_struct_def, late_passes, s, name, g, item_id);
        hir_visit::walk_struct_def(self, s);
        run_lints!(self, check_struct_def_post, late_passes, s, name, g, item_id);
    }

    fn visit_struct_field(&mut self, s: &hir::StructField) {
        self.with_lint_attrs(&s.node.attrs, |cx| {
            run_lints!(cx, check_struct_field, late_passes, s);
            hir_visit::walk_struct_field(cx, s);
        })
    }

    fn visit_variant(&mut self, v: &hir::Variant, g: &hir::Generics, item_id: ast::NodeId) {
        self.with_lint_attrs(&v.node.attrs, |cx| {
            run_lints!(cx, check_variant, late_passes, v, g);
            hir_visit::walk_variant(cx, v, g, item_id);
            run_lints!(cx, check_variant_post, late_passes, v, g);
        })
    }

    fn visit_ty(&mut self, t: &hir::Ty) {
        run_lints!(self, check_ty, late_passes, t);
        hir_visit::walk_ty(self, t);
    }

    fn visit_name(&mut self, sp: Span, name: ast::Name) {
        run_lints!(self, check_name, late_passes, sp, name);
    }

    fn visit_mod(&mut self, m: &hir::Mod, s: Span, n: ast::NodeId) {
        run_lints!(self, check_mod, late_passes, m, s, n);
        hir_visit::walk_mod(self, m);
    }

    fn visit_local(&mut self, l: &hir::Local) {
        run_lints!(self, check_local, late_passes, l);
        hir_visit::walk_local(self, l);
    }

    fn visit_block(&mut self, b: &hir::Block) {
        run_lints!(self, check_block, late_passes, b);
        hir_visit::walk_block(self, b);
    }

    fn visit_arm(&mut self, a: &hir::Arm) {
        run_lints!(self, check_arm, late_passes, a);
        hir_visit::walk_arm(self, a);
    }

    fn visit_decl(&mut self, d: &hir::Decl) {
        run_lints!(self, check_decl, late_passes, d);
        hir_visit::walk_decl(self, d);
    }

    fn visit_expr_post(&mut self, e: &hir::Expr) {
        run_lints!(self, check_expr_post, late_passes, e);
    }

    fn visit_generics(&mut self, g: &hir::Generics) {
        run_lints!(self, check_generics, late_passes, g);
        hir_visit::walk_generics(self, g);
    }

    fn visit_trait_item(&mut self, trait_item: &hir::TraitItem) {
        self.with_lint_attrs(&trait_item.attrs, |cx| {
            run_lints!(cx, check_trait_item, late_passes, trait_item);
            cx.visit_ids(|v| v.visit_trait_item(trait_item));
            hir_visit::walk_trait_item(cx, trait_item);
        });
    }

    fn visit_impl_item(&mut self, impl_item: &hir::ImplItem) {
        self.with_lint_attrs(&impl_item.attrs, |cx| {
            run_lints!(cx, check_impl_item, late_passes, impl_item);
            cx.visit_ids(|v| v.visit_impl_item(impl_item));
            hir_visit::walk_impl_item(cx, impl_item);
        });
    }

    fn visit_lifetime(&mut self, lt: &hir::Lifetime) {
        run_lints!(self, check_lifetime, late_passes, lt);
    }

    fn visit_lifetime_def(&mut self, lt: &hir::LifetimeDef) {
        run_lints!(self, check_lifetime_def, late_passes, lt);
    }

    fn visit_explicit_self(&mut self, es: &hir::ExplicitSelf) {
        run_lints!(self, check_explicit_self, late_passes, es);
        hir_visit::walk_explicit_self(self, es);
    }

    fn visit_path(&mut self, p: &hir::Path, id: ast::NodeId) {
        run_lints!(self, check_path, late_passes, p, id);
        hir_visit::walk_path(self, p);
    }

    fn visit_path_list_item(&mut self, prefix: &hir::Path, item: &hir::PathListItem) {
        run_lints!(self, check_path_list_item, late_passes, item);
        hir_visit::walk_path_list_item(self, prefix, item);
    }

    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        run_lints!(self, check_attribute, late_passes, attr);
    }
}

impl<'a, 'v> ast_visit::Visitor<'v> for EarlyContext<'a> {
    fn visit_item(&mut self, it: &ast::Item) {
        self.with_lint_attrs(&it.attrs, |cx| {
            run_lints!(cx, check_item, early_passes, it);
            cx.visit_ids(|v| v.visit_item(it));
            ast_visit::walk_item(cx, it);
        })
    }

    fn visit_foreign_item(&mut self, it: &ast::ForeignItem) {
        self.with_lint_attrs(&it.attrs, |cx| {
            run_lints!(cx, check_foreign_item, early_passes, it);
            ast_visit::walk_foreign_item(cx, it);
        })
    }

    fn visit_pat(&mut self, p: &ast::Pat) {
        run_lints!(self, check_pat, early_passes, p);
        ast_visit::walk_pat(self, p);
    }

    fn visit_expr(&mut self, e: &ast::Expr) {
        run_lints!(self, check_expr, early_passes, e);
        ast_visit::walk_expr(self, e);
    }

    fn visit_stmt(&mut self, s: &ast::Stmt) {
        run_lints!(self, check_stmt, early_passes, s);
        ast_visit::walk_stmt(self, s);
    }

    fn visit_fn(&mut self, fk: ast_visit::FnKind<'v>, decl: &'v ast::FnDecl,
                body: &'v ast::Block, span: Span, id: ast::NodeId) {
        run_lints!(self, check_fn, early_passes, fk, decl, body, span, id);
        ast_visit::walk_fn(self, fk, decl, body, span);
    }

    fn visit_variant_data(&mut self,
                        s: &ast::VariantData,
                        ident: ast::Ident,
                        g: &ast::Generics,
                        item_id: ast::NodeId,
                        _: Span) {
        run_lints!(self, check_struct_def, early_passes, s, ident, g, item_id);
        ast_visit::walk_struct_def(self, s);
        run_lints!(self, check_struct_def_post, early_passes, s, ident, g, item_id);
    }

    fn visit_struct_field(&mut self, s: &ast::StructField) {
        self.with_lint_attrs(&s.node.attrs, |cx| {
            run_lints!(cx, check_struct_field, early_passes, s);
            ast_visit::walk_struct_field(cx, s);
        })
    }

    fn visit_variant(&mut self, v: &ast::Variant, g: &ast::Generics, item_id: ast::NodeId) {
        self.with_lint_attrs(&v.node.attrs, |cx| {
            run_lints!(cx, check_variant, early_passes, v, g);
            ast_visit::walk_variant(cx, v, g, item_id);
            run_lints!(cx, check_variant_post, early_passes, v, g);
        })
    }

    fn visit_ty(&mut self, t: &ast::Ty) {
        run_lints!(self, check_ty, early_passes, t);
        ast_visit::walk_ty(self, t);
    }

    fn visit_ident(&mut self, sp: Span, id: ast::Ident) {
        run_lints!(self, check_ident, early_passes, sp, id);
    }

    fn visit_mod(&mut self, m: &ast::Mod, s: Span, n: ast::NodeId) {
        run_lints!(self, check_mod, early_passes, m, s, n);
        ast_visit::walk_mod(self, m);
    }

    fn visit_local(&mut self, l: &ast::Local) {
        run_lints!(self, check_local, early_passes, l);
        ast_visit::walk_local(self, l);
    }

    fn visit_block(&mut self, b: &ast::Block) {
        run_lints!(self, check_block, early_passes, b);
        ast_visit::walk_block(self, b);
    }

    fn visit_arm(&mut self, a: &ast::Arm) {
        run_lints!(self, check_arm, early_passes, a);
        ast_visit::walk_arm(self, a);
    }

    fn visit_decl(&mut self, d: &ast::Decl) {
        run_lints!(self, check_decl, early_passes, d);
        ast_visit::walk_decl(self, d);
    }

    fn visit_expr_post(&mut self, e: &ast::Expr) {
        run_lints!(self, check_expr_post, early_passes, e);
    }

    fn visit_generics(&mut self, g: &ast::Generics) {
        run_lints!(self, check_generics, early_passes, g);
        ast_visit::walk_generics(self, g);
    }

    fn visit_trait_item(&mut self, trait_item: &ast::TraitItem) {
        self.with_lint_attrs(&trait_item.attrs, |cx| {
            run_lints!(cx, check_trait_item, early_passes, trait_item);
            cx.visit_ids(|v| v.visit_trait_item(trait_item));
            ast_visit::walk_trait_item(cx, trait_item);
        });
    }

    fn visit_impl_item(&mut self, impl_item: &ast::ImplItem) {
        self.with_lint_attrs(&impl_item.attrs, |cx| {
            run_lints!(cx, check_impl_item, early_passes, impl_item);
            cx.visit_ids(|v| v.visit_impl_item(impl_item));
            ast_visit::walk_impl_item(cx, impl_item);
        });
    }

    fn visit_lifetime(&mut self, lt: &ast::Lifetime) {
        run_lints!(self, check_lifetime, early_passes, lt);
    }

    fn visit_lifetime_def(&mut self, lt: &ast::LifetimeDef) {
        run_lints!(self, check_lifetime_def, early_passes, lt);
    }

    fn visit_explicit_self(&mut self, es: &ast::ExplicitSelf) {
        run_lints!(self, check_explicit_self, early_passes, es);
        ast_visit::walk_explicit_self(self, es);
    }

    fn visit_path(&mut self, p: &ast::Path, id: ast::NodeId) {
        run_lints!(self, check_path, early_passes, p, id);
        ast_visit::walk_path(self, p);
    }

    fn visit_path_list_item(&mut self, prefix: &ast::Path, item: &ast::PathListItem) {
        run_lints!(self, check_path_list_item, early_passes, item);
        ast_visit::walk_path_list_item(self, prefix, item);
    }

    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        run_lints!(self, check_attribute, early_passes, attr);
    }
}

// Output any lints that were previously added to the session.
impl<'a, 'tcx> IdVisitingOperation for LateContext<'a, 'tcx> {
    fn visit_id(&mut self, id: ast::NodeId) {
        match self.sess().lints.borrow_mut().remove(&id) {
            None => {}
            Some(lints) => {
                for (lint_id, span, msg) in lints {
                    self.span_lint(lint_id.lint, span, &msg[..])
                }
            }
        }
    }
}
impl<'a> IdVisitingOperation for EarlyContext<'a> {
    fn visit_id(&mut self, id: ast::NodeId) {
        match self.sess.lints.borrow_mut().remove(&id) {
            None => {}
            Some(lints) => {
                for (lint_id, span, msg) in lints {
                    self.span_lint(lint_id.lint, span, &msg[..])
                }
            }
        }
    }
}

// This lint pass is defined here because it touches parts of the `LateContext`
// that we don't want to expose. It records the lint level at certain AST
// nodes, so that the variant size difference check in trans can call
// `raw_emit_lint`.

pub struct GatherNodeLevels;

impl LintPass for GatherNodeLevels {
    fn get_lints(&self) -> LintArray {
        lint_array!()
    }
}

impl LateLintPass for GatherNodeLevels {
    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        match it.node {
            hir::ItemEnum(..) => {
                let lint_id = LintId::of(builtin::VARIANT_SIZE_DIFFERENCES);
                let lvlsrc = cx.lints.get_level_source(lint_id);
                match lvlsrc {
                    (lvl, _) if lvl != Allow => {
                        cx.node_levels.borrow_mut()
                            .insert((it.id, lint_id), lvlsrc);
                    },
                    _ => { }
                }
            },
            _ => { }
        }
    }
}

/// Perform lint checking on a crate.
///
/// Consumes the `lint_store` field of the `Session`.
pub fn check_crate(tcx: &ty::ctxt,
                   krate: &hir::Crate,
                   exported_items: &ExportedItems) {

    let mut cx = LateContext::new(tcx, krate, exported_items);

    // Visit the whole crate.
    cx.with_lint_attrs(&krate.attrs, |cx| {
        cx.visit_id(ast::CRATE_NODE_ID);
        cx.visit_ids(|v| {
            v.visited_outermost = true;
            hir_visit::walk_crate(v, krate);
        });

        // since the root module isn't visited as an item (because it isn't an
        // item), warn for it here.
        run_lints!(cx, check_crate, late_passes, krate);

        hir_visit::walk_crate(cx, krate);
    });

    // If we missed any lints added to the session, then there's a bug somewhere
    // in the iteration code.
    for (id, v) in tcx.sess.lints.borrow().iter() {
        for &(lint, span, ref msg) in v {
            tcx.sess.span_bug(span,
                              &format!("unprocessed lint {} at {}: {}",
                                       lint.as_str(), tcx.map.node_to_string(*id), *msg))
        }
    }

    *tcx.node_lint_levels.borrow_mut() = cx.node_levels.into_inner();
}

pub fn check_ast_crate(sess: &Session, krate: &ast::Crate) {
    let mut cx = EarlyContext::new(sess, krate);

    // Visit the whole crate.
    cx.with_lint_attrs(&krate.attrs, |cx| {
        cx.visit_id(ast::CRATE_NODE_ID);
        cx.visit_ids(|v| {
            v.visited_outermost = true;
            ast_visit::walk_crate(v, krate);
        });

        // since the root module isn't visited as an item (because it isn't an
        // item), warn for it here.
        run_lints!(cx, check_crate, early_passes, krate);

        ast_visit::walk_crate(cx, krate);
    });

    // Put the lint store back in the session.
    mem::replace(&mut *sess.lint_store.borrow_mut(), cx.lints);

    // If we missed any lints added to the session, then there's a bug somewhere
    // in the iteration code.
    for (_, v) in sess.lints.borrow().iter() {
        for &(lint, span, ref msg) in v {
            sess.span_bug(span,
                          &format!("unprocessed lint {}: {}",
                                   lint.as_str(), *msg))
        }
    }
}
