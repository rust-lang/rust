// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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
use session::config::UnstableFeatures;
use lint::{Level, LevelSource, Lint, LintId, LintArray, LintPass, LintPassObject};
use lint::{Default, CommandLine, Node, Allow, Warn, Deny, Forbid, ReleaseChannel};
use lint::builtin;
use util::nodemap::FnvHashMap;

use std::cell::RefCell;
use std::mem;
use syntax::ast_util::IdVisitingOperation;
use syntax::attr::AttrMetaMethods;
use syntax::attr;
use syntax::codemap::Span;
use syntax::visit::{Visitor, FnKind};
use syntax::parse::token::InternedString;
use syntax::{ast, ast_util, visit};

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
    passes: Option<Vec<LintPassObject>>,

    /// Lints indexed by name.
    by_name: FnvHashMap<String, TargetLint>,

    /// Current levels of each lint, and where they were set.
    levels: FnvHashMap<LintId, LevelSource>,

    /// Map of registered lint groups to what lints they expand to. The bool
    /// is true if the lint group was added by a plugin.
    lint_groups: FnvHashMap<&'static str, (Vec<LintId>, bool)>,
}

/// The targed of the `by_name` map, which accounts for renaming/deprecation.
enum TargetLint {
    /// A direct lint target
    Id(LintId),

    /// Temporary renaming, used for easing migration pain; see #16545
    Renamed(String, LintId),
}

impl LintStore {
    fn get_level_source(&self, lint: LintId) -> LevelSource {
        match self.levels.get(&lint) {
            Some(&s) => s,
            None => (Allow, Default),
        }
    }

    fn set_level(&mut self, lint: LintId, lvlsrc: LevelSource) {
        if lvlsrc.0 == Allow {
            self.levels.remove(&lint);
        } else {
            self.levels.insert(lint, lvlsrc);
        }
    }

    pub fn new() -> LintStore {
        LintStore {
            lints: vec!(),
            passes: Some(vec!()),
            by_name: FnvHashMap(),
            levels: FnvHashMap(),
            lint_groups: FnvHashMap(),
        }
    }

    pub fn get_lints<'t>(&'t self) -> &'t [(&'static Lint, bool)] {
        &self.lints[]
    }

    pub fn get_lint_groups<'t>(&'t self) -> Vec<(&'static str, Vec<LintId>, bool)> {
        self.lint_groups.iter().map(|(k, v)| (*k,
                                              v.0.clone(),
                                              v.1)).collect()
    }

    pub fn register_pass(&mut self, sess: Option<&Session>,
                         from_plugin: bool, pass: LintPassObject) {
        for &lint in pass.get_lints() {
            self.lints.push((*lint, from_plugin));

            let id = LintId::of(*lint);
            if self.by_name.insert(lint.name_lower(), Id(id)).is_some() {
                let msg = format!("duplicate specification of lint {}", lint.name_lower());
                match (sess, from_plugin) {
                    // We load builtin lints first, so a duplicate is a compiler bug.
                    // Use early_error when handling -W help with no crate.
                    (None, _) => early_error(&msg[..]),
                    (Some(sess), false) => sess.bug(&msg[..]),

                    // A duplicate name from a plugin is a user error.
                    (Some(sess), true)  => sess.err(&msg[..]),
                }
            }

            if lint.default_level != Allow {
                self.levels.insert(id, (lint.default_level, Default));
            }
        }
        self.passes.as_mut().unwrap().push(pass);
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
                (None, _) => early_error(&msg[..]),
                (Some(sess), false) => sess.bug(&msg[..]),

                // A duplicate name from a plugin is a user error.
                (Some(sess), true)  => sess.err(&msg[..]),
            }
        }
    }

    fn register_renamed(&mut self, old_name: &str, new_name: &str) {
        let target = match self.by_name.get(new_name) {
            Some(&Id(lint_id)) => lint_id.clone(),
            _ => panic!("invalid lint renaming of {} to {}", old_name, new_name)
        };
        self.by_name.insert(old_name.to_string(), Renamed(new_name.to_string(), target));
    }

    pub fn register_builtin(&mut self, sess: Option<&Session>) {
        macro_rules! add_builtin {
            ($sess:ident, $($name:ident),*,) => (
                {$(
                    self.register_pass($sess, false, box builtin::$name as LintPassObject);
                )*}
            )
        }

        macro_rules! add_builtin_with_new {
            ($sess:ident, $($name:ident),*,) => (
                {$(
                    self.register_pass($sess, false, box builtin::$name::new() as LintPassObject);
                )*}
            )
        }

        macro_rules! add_lint_group {
            ($sess:ident, $name:expr, $($lint:ident),*) => (
                self.register_group($sess, false, $name, vec![$(LintId::of(builtin::$lint)),*]);
            )
        }

        add_builtin!(sess,
                     HardwiredLints,
                     WhileTrue,
                     UnusedCasts,
                     ImproperCTypes,
                     BoxPointers,
                     UnusedAttributes,
                     PathStatements,
                     UnusedResults,
                     NonCamelCaseTypes,
                     NonSnakeCase,
                     NonUpperCaseGlobals,
                     UnusedParens,
                     UnusedImportBraces,
                     NonShorthandFieldPatterns,
                     UnusedUnsafe,
                     UnsafeBlocks,
                     UnusedMut,
                     UnusedAllocation,
                     MissingCopyImplementations,
                     UnstableFeatures,
                     Stability,
                     UnconditionalRecursion,
                     InvalidNoMangleItems,
                     PluginAsLibrary,
        );

        add_builtin_with_new!(sess,
                              TypeLimits,
                              RawPointerDerive,
                              MissingDoc,
                              MissingDebugImplementations,
        );

        add_lint_group!(sess, "bad_style",
                        NON_CAMEL_CASE_TYPES, NON_SNAKE_CASE, NON_UPPER_CASE_GLOBALS);

        add_lint_group!(sess, "unused",
                        UNUSED_IMPORTS, UNUSED_VARIABLES, UNUSED_ASSIGNMENTS, DEAD_CODE,
                        UNUSED_MUT, UNREACHABLE_CODE, UNUSED_MUST_USE,
                        UNUSED_UNSAFE, PATH_STATEMENTS);

        // We have one lint pass defined in this module.
        self.register_pass(sess, false, box GatherNodeLevels as LintPassObject);

        // Insert temporary renamings for a one-time deprecation
        self.register_renamed("raw_pointer_deriving", "raw_pointer_derive");

        self.register_renamed("unknown_features", "unused_features");
    }

    #[allow(unused_variables)]
    fn find_lint(&self, lint_name: &str, sess: &Session, span: Option<Span>)
                 -> Option<LintId>
    {
        match self.by_name.get(lint_name) {
            Some(&Id(lint_id)) => Some(lint_id),
            Some(&Renamed(ref new_name, lint_id)) => {
                let warning = format!("lint {} has been renamed to {}",
                                      lint_name, new_name);
                match span {
                    Some(span) => sess.span_warn(span, &warning[..]),
                    None => sess.warn(&warning[..]),
                };
                Some(lint_id)
            }
            None => None
        }
    }

    pub fn process_command_line(&mut self, sess: &Session) {
        for &(ref lint_name, level) in &sess.opts.lint_opts {
            match self.find_lint(&lint_name[..], sess, None) {
                Some(lint_id) => self.set_level(lint_id, (level, CommandLine)),
                None => {
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
                                                 level.as_str(), lint_name)[]),
                    }
                }
            }
        }
    }

    fn maybe_stage_features(&mut self, sess: &Session) {
        let lvl = match sess.opts.unstable_features {
            UnstableFeatures::Default => return,
            UnstableFeatures::Disallow => Warn,
            UnstableFeatures::Cheat => Allow
        };
        match self.by_name.get("unstable_features") {
            Some(&Id(lint_id)) => if self.get_level_source(lint_id).0 != Forbid {
                self.set_level(lint_id, (lvl, ReleaseChannel))
            },
            Some(&Renamed(_, lint_id)) => if self.get_level_source(lint_id).0 != Forbid {
                self.set_level(lint_id, (lvl, ReleaseChannel))
            },
            None => unreachable!()
        }
    }
}

/// Context for lint checking.
pub struct Context<'a, 'tcx: 'a> {
    /// Type context we're checking in.
    pub tcx: &'a ty::ctxt<'tcx>,

    /// The crate being checked.
    pub krate: &'a ast::Crate,

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

/// Convenience macro for calling a `LintPass` method on every pass in the context.
macro_rules! run_lints { ($cx:expr, $f:ident, $($args:expr),*) => ({
    // Move the vector of passes out of `$cx` so that we can
    // iterate over it mutably while passing `$cx` to the methods.
    let mut passes = $cx.lints.passes.take().unwrap();
    for obj in &mut passes {
        obj.$f($cx, $($args),*);
    }
    $cx.lints.passes = Some(passes);
}) }

/// Parse the lint attributes into a vector, with `Err`s for malformed lint
/// attributes. Writing this as an iterator is an enormous mess.
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
    let mut note = None;
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
        ReleaseChannel => {
            let release_channel = option_env!("CFG_RELEASE_CHANNEL").unwrap_or("(unknown)");
            note = Some(format!("this feature may not be used in the {} release channel",
                                release_channel));
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

    if let Some(note) = note {
        sess.note(&note[..]);
    }

    if let Some(span) = def {
        sess.span_note(span, "lint level defined here");
    }
}

impl<'a, 'tcx> Context<'a, 'tcx> {
    fn new(tcx: &'a ty::ctxt<'tcx>,
           krate: &'a ast::Crate,
           exported_items: &'a ExportedItems) -> Context<'a, 'tcx> {
        // We want to own the lint store, so move it out of the session.
        let lint_store = mem::replace(&mut *tcx.sess.lint_store.borrow_mut(),
                                      LintStore::new());

        Context {
            tcx: tcx,
            krate: krate,
            exported_items: exported_items,
            lints: lint_store,
            level_stack: vec![],
            node_levels: RefCell::new(FnvHashMap()),
        }
    }

    /// Get the overall compiler `Session` object.
    pub fn sess(&'a self) -> &'a Session {
        &self.tcx.sess
    }

    /// Get the level of `lint` at the current position of the lint
    /// traversal.
    pub fn current_level(&self, lint: &'static Lint) -> Level {
        self.lints.levels.get(&LintId::of(lint)).map_or(Allow, |&(lvl, _)| lvl)
    }

    fn lookup_and_emit(&self, lint: &'static Lint, span: Option<Span>, msg: &str) {
        let (level, src) = match self.lints.levels.get(&LintId::of(lint)) {
            None => return,
            Some(&(Warn, src)) => {
                let lint_id = LintId::of(builtin::WARNINGS);
                (self.lints.get_level_source(lint_id).0, src)
            }
            Some(&pair) => pair,
        };

        raw_emit_lint(&self.tcx.sess, lint, (level, src), span, msg);
    }

    /// Emit a lint at the appropriate level, with no associated span.
    pub fn lint(&self, lint: &'static Lint, msg: &str) {
        self.lookup_and_emit(lint, None, msg);
    }

    /// Emit a lint at the appropriate level, for a particular span.
    pub fn span_lint(&self, lint: &'static Lint, span: Span, msg: &str) {
        self.lookup_and_emit(lint, Some(span), msg);
    }

    /// Merge the lints specified by any lint attributes into the
    /// current lint context, call the provided function, then reset the
    /// lints in effect to their previous state.
    fn with_lint_attrs<F>(&mut self,
                          attrs: &[ast::Attribute],
                          f: F) where
        F: FnOnce(&mut Context),
    {
        // Parse all of the lint attributes, and then add them all to the
        // current dictionary of lint information. Along the way, keep a history
        // of what we changed so we can roll everything back after invoking the
        // specified closure
        let mut pushed = 0;

        for result in gather_attrs(attrs) {
            let v = match result {
                Err(span) => {
                    self.tcx.sess.span_err(span, "malformed lint attribute");
                    continue;
                }
                Ok((lint_name, level, span)) => {
                    match self.lints.find_lint(&lint_name, &self.tcx.sess, Some(span)) {
                        Some(lint_id) => vec![(lint_id, level, span)],
                        None => {
                            match self.lints.lint_groups.get(&lint_name[..]) {
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
                        }
                    }
                }
            };

            for (lint_id, level, span) in v {
                let now = self.lints.get_level_source(lint_id).0;
                if now == Forbid && level != Forbid {
                    let lint_name = lint_id.as_str();
                    self.tcx.sess.span_err(span,
                                           &format!("{}({}) overruled by outer forbid({})",
                                                   level.as_str(), lint_name,
                                                   lint_name)[]);
                } else if now != level {
                    let src = self.lints.get_level_source(lint_id).1;
                    self.level_stack.push((lint_id, (now, src)));
                    pushed += 1;
                    self.lints.set_level(lint_id, (level, Node(span)));
                }
            }
        }

        run_lints!(self, enter_lint_attrs, attrs);
        f(self);
        run_lints!(self, exit_lint_attrs, attrs);

        // rollback
        for _ in 0..pushed {
            let (lint, lvlsrc) = self.level_stack.pop().unwrap();
            self.lints.set_level(lint, lvlsrc);
        }
    }

    fn visit_ids<F>(&mut self, f: F) where
        F: FnOnce(&mut ast_util::IdVisitor<Context>)
    {
        let mut v = ast_util::IdVisitor {
            operation: self,
            pass_through_items: false,
            visited_outermost: false,
        };
        f(&mut v);
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for Context<'a, 'tcx> {
    fn visit_item(&mut self, it: &ast::Item) {
        self.with_lint_attrs(&it.attrs[], |cx| {
            run_lints!(cx, check_item, it);
            cx.visit_ids(|v| v.visit_item(it));
            visit::walk_item(cx, it);
        })
    }

    fn visit_foreign_item(&mut self, it: &ast::ForeignItem) {
        self.with_lint_attrs(&it.attrs[], |cx| {
            run_lints!(cx, check_foreign_item, it);
            visit::walk_foreign_item(cx, it);
        })
    }

    fn visit_pat(&mut self, p: &ast::Pat) {
        run_lints!(self, check_pat, p);
        visit::walk_pat(self, p);
    }

    fn visit_expr(&mut self, e: &ast::Expr) {
        run_lints!(self, check_expr, e);
        visit::walk_expr(self, e);
    }

    fn visit_stmt(&mut self, s: &ast::Stmt) {
        run_lints!(self, check_stmt, s);
        visit::walk_stmt(self, s);
    }

    fn visit_fn(&mut self, fk: FnKind<'v>, decl: &'v ast::FnDecl,
                body: &'v ast::Block, span: Span, id: ast::NodeId) {
        match fk {
            visit::FkMethod(_, _, m) => {
                self.with_lint_attrs(&m.attrs[], |cx| {
                    run_lints!(cx, check_fn, fk, decl, body, span, id);
                    cx.visit_ids(|v| {
                        v.visit_fn(fk, decl, body, span, id);
                    });
                    visit::walk_fn(cx, fk, decl, body, span);
                })
            },
            _ => {
                run_lints!(self, check_fn, fk, decl, body, span, id);
                visit::walk_fn(self, fk, decl, body, span);
            }
        }
    }

    fn visit_ty_method(&mut self, t: &ast::TypeMethod) {
        self.with_lint_attrs(&t.attrs[], |cx| {
            run_lints!(cx, check_ty_method, t);
            visit::walk_ty_method(cx, t);
        })
    }

    fn visit_struct_def(&mut self,
                        s: &ast::StructDef,
                        ident: ast::Ident,
                        g: &ast::Generics,
                        id: ast::NodeId) {
        run_lints!(self, check_struct_def, s, ident, g, id);
        visit::walk_struct_def(self, s);
        run_lints!(self, check_struct_def_post, s, ident, g, id);
    }

    fn visit_struct_field(&mut self, s: &ast::StructField) {
        self.with_lint_attrs(&s.node.attrs[], |cx| {
            run_lints!(cx, check_struct_field, s);
            visit::walk_struct_field(cx, s);
        })
    }

    fn visit_variant(&mut self, v: &ast::Variant, g: &ast::Generics) {
        self.with_lint_attrs(&v.node.attrs[], |cx| {
            run_lints!(cx, check_variant, v, g);
            visit::walk_variant(cx, v, g);
            run_lints!(cx, check_variant_post, v, g);
        })
    }

    // FIXME(#10894) should continue recursing
    fn visit_ty(&mut self, t: &ast::Ty) {
        run_lints!(self, check_ty, t);
    }

    fn visit_ident(&mut self, sp: Span, id: ast::Ident) {
        run_lints!(self, check_ident, sp, id);
    }

    fn visit_mod(&mut self, m: &ast::Mod, s: Span, n: ast::NodeId) {
        run_lints!(self, check_mod, m, s, n);
        visit::walk_mod(self, m);
    }

    fn visit_local(&mut self, l: &ast::Local) {
        run_lints!(self, check_local, l);
        visit::walk_local(self, l);
    }

    fn visit_block(&mut self, b: &ast::Block) {
        run_lints!(self, check_block, b);
        visit::walk_block(self, b);
    }

    fn visit_arm(&mut self, a: &ast::Arm) {
        run_lints!(self, check_arm, a);
        visit::walk_arm(self, a);
    }

    fn visit_decl(&mut self, d: &ast::Decl) {
        run_lints!(self, check_decl, d);
        visit::walk_decl(self, d);
    }

    fn visit_expr_post(&mut self, e: &ast::Expr) {
        run_lints!(self, check_expr_post, e);
    }

    fn visit_generics(&mut self, g: &ast::Generics) {
        run_lints!(self, check_generics, g);
        visit::walk_generics(self, g);
    }

    fn visit_trait_item(&mut self, m: &ast::TraitItem) {
        run_lints!(self, check_trait_method, m);
        visit::walk_trait_item(self, m);
    }

    fn visit_opt_lifetime_ref(&mut self, sp: Span, lt: &Option<ast::Lifetime>) {
        run_lints!(self, check_opt_lifetime_ref, sp, lt);
    }

    fn visit_lifetime_ref(&mut self, lt: &ast::Lifetime) {
        run_lints!(self, check_lifetime_ref, lt);
    }

    fn visit_lifetime_def(&mut self, lt: &ast::LifetimeDef) {
        run_lints!(self, check_lifetime_def, lt);
    }

    fn visit_explicit_self(&mut self, es: &ast::ExplicitSelf) {
        run_lints!(self, check_explicit_self, es);
        visit::walk_explicit_self(self, es);
    }

    fn visit_mac(&mut self, mac: &ast::Mac) {
        run_lints!(self, check_mac, mac);
        visit::walk_mac(self, mac);
    }

    fn visit_path(&mut self, p: &ast::Path, id: ast::NodeId) {
        run_lints!(self, check_path, p, id);
        visit::walk_path(self, p);
    }

    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        run_lints!(self, check_attribute, attr);
    }
}

// Output any lints that were previously added to the session.
impl<'a, 'tcx> IdVisitingOperation for Context<'a, 'tcx> {
    fn visit_id(&mut self, id: ast::NodeId) {
        match self.tcx.sess.lints.borrow_mut().remove(&id) {
            None => {}
            Some(lints) => {
                for (lint_id, span, msg) in lints {
                    self.span_lint(lint_id.lint, span, &msg[..])
                }
            }
        }
    }
}

// This lint pass is defined here because it touches parts of the `Context`
// that we don't want to expose. It records the lint level at certain AST
// nodes, so that the variant size difference check in trans can call
// `raw_emit_lint`.

struct GatherNodeLevels;

impl LintPass for GatherNodeLevels {
    fn get_lints(&self) -> LintArray {
        lint_array!()
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        match it.node {
            ast::ItemEnum(..) => {
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
                   exported_items: &ExportedItems) {

    // If this is a feature-staged build of rustc then flip several lints to 'forbid'
    tcx.sess.lint_store.borrow_mut().maybe_stage_features(&tcx.sess);

    let krate = tcx.map.krate();
    let mut cx = Context::new(tcx, krate, exported_items);

    // Visit the whole crate.
    cx.with_lint_attrs(&krate.attrs[], |cx| {
        cx.visit_id(ast::CRATE_NODE_ID);
        cx.visit_ids(|v| {
            v.visited_outermost = true;
            visit::walk_crate(v, krate);
        });

        // since the root module isn't visited as an item (because it isn't an
        // item), warn for it here.
        run_lints!(cx, check_crate, krate);

        visit::walk_crate(cx, krate);
    });

    // If we missed any lints added to the session, then there's a bug somewhere
    // in the iteration code.
    for (id, v) in &*tcx.sess.lints.borrow() {
        for &(lint, span, ref msg) in v {
            tcx.sess.span_bug(span,
                              &format!("unprocessed lint {} at {}: {}",
                                       lint.as_str(), tcx.map.node_to_string(*id), *msg))
        }
    }

    *tcx.node_lint_levels.borrow_mut() = cx.node_levels.into_inner();
}
