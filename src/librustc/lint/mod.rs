// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A 'lint' check is a kind of miscellaneous constraint that a user _might_
//! want to enforce, but might reasonably want to permit as well, on a
//! module-by-module basis. They contrast with static constraints enforced by
//! other phases of the compiler, which are generally required to hold in order
//! to compile the program at all.
//!
//! The lint checking is all consolidated into one pass which runs just before
//! translation to LLVM bytecode. Throughout compilation, lint warnings can be
//! added via the `add_lint` method on the Session structure. This requires a
//! span and an id of the node that the lint is being added to. The lint isn't
//! actually emitted at that time because it is unknown what the actual lint
//! level at that location is.
//!
//! To actually emit lint warnings/errors, a separate pass is used just before
//! translation. A context keeps track of the current state of all lint levels.
//! Upon entering a node of the ast which can modify the lint settings, the
//! previous lint state is pushed onto a stack and the ast is then recursed
//! upon.  As the ast is traversed, this keeps track of the current lint level
//! for all lint attributes.
//!
//! Most of the lints built into `rustc` are structs implementing `LintPass`,
//! and are defined within `builtin.rs`. To add a new lint you can define such
//! a struct and add it to the `builtin_lints!` macro invocation in this file.
//! `LintPass` itself is not a subtrait of `Default`, but the `builtin_lints!`
//! macro requires `Default` (usually via `deriving`).
//!
//! Some lints are defined elsewhere in the compiler and work by calling
//! `add_lint()` on the overall `Session` object.
//!
//! If you're adding lints to the `Context` infrastructure itself, defined in
//! this file, use `span_lint` instead of `add_lint`.

#![allow(non_camel_case_types)]
#![macro_escape]

use middle::privacy::ExportedItems;
use middle::ty;
use middle::typeck::astconv::AstConv;
use middle::typeck::infer;
use driver::session::Session;
use driver::early_error;

use std::collections::HashMap;
use std::rc::Rc;
use std::gc::Gc;
use std::to_str::ToStr;
use std::cell::RefCell;
use std::default::Default;
use std::hash::Hash;
use std::tuple::Tuple2;
use std::hash;
use std::mem;
use syntax::ast_util::IdVisitingOperation;
use syntax::attr::AttrMetaMethods;
use syntax::attr;
use syntax::codemap::Span;
use syntax::visit::{Visitor, FnKind};
use syntax::{ast, ast_util, visit};

#[macro_export]
macro_rules! lint_initializer (
    ($name:ident, $level:ident, $desc:expr) => (
        ::rustc::lint::Lint {
            name: stringify!($name),
            default_level: ::rustc::lint::$level,
            desc: $desc,
        }
    )
)

#[macro_export]
macro_rules! declare_lint (
    // FIXME(#14660): deduplicate
    (pub $name:ident, $level:ident, $desc:expr) => (
        pub static $name: &'static ::rustc::lint::Lint
            = &lint_initializer!($name, $level, $desc);
    );
    ($name:ident, $level:ident, $desc:expr) => (
        static $name: &'static ::rustc::lint::Lint
            = &lint_initializer!($name, $level, $desc);
    );
)

#[macro_export]
macro_rules! lint_array ( ($( $lint:expr ),*) => (
    {
        static array: LintArray = &[ $( $lint ),* ];
        array
    }
))

pub mod builtin;

/// Specification of a single lint.
pub struct Lint {
    /// An identifier for the lint, written with underscores,
    /// e.g. "unused_imports". This identifies the lint in
    /// attributes and in command-line arguments. On the
    /// command line, underscores become dashes.
    pub name: &'static str,

    /// Default level for the lint.
    pub default_level: Level,

    /// Description of the lint or the issue it detects,
    /// e.g. "imports that are never used"
    pub desc: &'static str,
}

pub type LintArray = &'static [&'static Lint];

/// Trait for types providing lint checks. Each `check` method checks a single
/// syntax node, and should not invoke methods recursively (unlike `Visitor`).
/// By default they do nothing.
//
// FIXME: eliminate the duplication with `Visitor`. But this also
// contains a few lint-specific methods with no equivalent in `Visitor`.
pub trait LintPass {
    /// Get descriptions of the lints this `LintPass` object can emit.
    ///
    /// NB: there is no enforcement that the object only emits lints it registered.
    /// And some `rustc` internal `LintPass`es register lints to be emitted by other
    /// parts of the compiler. If you want enforced access restrictions for your
    /// `Lint`, make it a private `static` item in its own module.
    fn get_lints(&self) -> LintArray;

    fn check_crate(&mut self, _: &Context, _: &ExportedItems, _: &ast::Crate) { }
    fn check_ident(&mut self, _: &Context, _: Span, _: ast::Ident) { }
    fn check_mod(&mut self, _: &Context, _: &ast::Mod, _: Span, _: ast::NodeId) { }
    fn check_view_item(&mut self, _: &Context, _: &ast::ViewItem) { }
    fn check_foreign_item(&mut self, _: &Context, _: &ast::ForeignItem) { }
    fn check_item(&mut self, _: &Context, _: &ast::Item) { }
    fn check_local(&mut self, _: &Context, _: &ast::Local) { }
    fn check_block(&mut self, _: &Context, _: &ast::Block) { }
    fn check_stmt(&mut self, _: &Context, _: &ast::Stmt) { }
    fn check_arm(&mut self, _: &Context, _: &ast::Arm) { }
    fn check_pat(&mut self, _: &Context, _: &ast::Pat) { }
    fn check_decl(&mut self, _: &Context, _: &ast::Decl) { }
    fn check_expr(&mut self, _: &Context, _: &ast::Expr) { }
    fn check_expr_post(&mut self, _: &Context, _: &ast::Expr) { }
    fn check_ty(&mut self, _: &Context, _: &ast::Ty) { }
    fn check_generics(&mut self, _: &Context, _: &ast::Generics) { }
    fn check_fn(&mut self, _: &Context,
        _: &FnKind, _: &ast::FnDecl, _: &ast::Block, _: Span, _: ast::NodeId) { }
    fn check_ty_method(&mut self, _: &Context, _: &ast::TypeMethod) { }
    fn check_trait_method(&mut self, _: &Context, _: &ast::TraitMethod) { }
    fn check_struct_def(&mut self, _: &Context,
        _: &ast::StructDef, _: ast::Ident, _: &ast::Generics, _: ast::NodeId) { }
    fn check_struct_def_post(&mut self, _: &Context,
        _: &ast::StructDef, _: ast::Ident, _: &ast::Generics, _: ast::NodeId) { }
    fn check_struct_field(&mut self, _: &Context, _: &ast::StructField) { }
    fn check_variant(&mut self, _: &Context, _: &ast::Variant, _: &ast::Generics) { }
    fn check_opt_lifetime_ref(&mut self, _: &Context, _: Span, _: &Option<ast::Lifetime>) { }
    fn check_lifetime_ref(&mut self, _: &Context, _: &ast::Lifetime) { }
    fn check_lifetime_decl(&mut self, _: &Context, _: &ast::Lifetime) { }
    fn check_explicit_self(&mut self, _: &Context, _: &ast::ExplicitSelf) { }
    fn check_mac(&mut self, _: &Context, _: &ast::Mac) { }
    fn check_path(&mut self, _: &Context, _: &ast::Path, _: ast::NodeId) { }
    fn check_attribute(&mut self, _: &Context, _: &ast::Attribute) { }

    /// Called when entering a syntax node that can have lint attributes such
    /// as `#[allow(...)]`. Called with *all* the attributes of that node.
    fn enter_lint_attrs(&mut self, _: &Context, _: &[ast::Attribute]) { }

    /// Counterpart to `enter_lint_attrs`.
    fn exit_lint_attrs(&mut self, _: &Context, _: &[ast::Attribute]) { }
}

type LintPassObject = Box<LintPass + 'static>;

/// Identifies a lint known to the compiler.
#[deriving(Clone)]
pub struct LintId {
    // Identity is based on pointer equality of this field.
    lint: &'static Lint,
}

impl PartialEq for LintId {
    fn eq(&self, other: &LintId) -> bool {
        (self.lint as *Lint) == (other.lint as *Lint)
    }
}

impl Eq for LintId { }

impl<S: hash::Writer> Hash<S> for LintId {
    fn hash(&self, state: &mut S) {
        let ptr = self.lint as *Lint;
        ptr.hash(state);
    }
}

impl LintId {
    pub fn of(lint: &'static Lint) -> LintId {
        LintId {
            lint: lint,
        }
    }

    pub fn as_str(&self) -> &'static str {
        self.lint.name
    }
}

#[deriving(Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum Level {
    Allow, Warn, Deny, Forbid
}

impl Level {
    pub fn as_str(self) -> &'static str {
        match self {
            Allow => "allow",
            Warn => "warn",
            Deny => "deny",
            Forbid => "forbid",
        }
    }

    pub fn from_str(x: &str) -> Option<Level> {
        match x {
            "allow" => Some(Allow),
            "warn" => Some(Warn),
            "deny" => Some(Deny),
            "forbid" => Some(Forbid),
            _ => None,
        }
    }
}

// this is public for the lints that run in trans
#[deriving(PartialEq)]
pub enum LintSource {
    Node(Span),
    Default,
    CommandLine
}

pub type LevelSource = (Level, LintSource);

/// Information about the registered lints.
/// This is basically the subset of `Context` that we can
/// build early in the compile pipeline.
pub struct LintStore {
    /// Registered lints. The bool is true if the lint was
    /// added by a plugin.
    lints: Vec<(&'static Lint, bool)>,

    /// Trait objects for each lint pass.
    passes: Vec<RefCell<LintPassObject>>,

    /// Lints indexed by name.
    by_name: HashMap<&'static str, LintId>,

    /// Current levels of each lint, and where they were set.
    levels: HashMap<LintId, LevelSource>,
}

impl LintStore {
    fn get_level_source(&self, lint: LintId) -> LevelSource {
        match self.levels.find(&lint) {
            Some(&s) => s,
            None => (Allow, Default),
        }
    }

    fn set_level(&mut self, lint: LintId, lvlsrc: LevelSource) {
        if lvlsrc.val0() == Allow {
            self.levels.remove(&lint);
        } else {
            self.levels.insert(lint, lvlsrc);
        }
    }

    pub fn new() -> LintStore {
        LintStore {
            lints: vec!(),
            passes: vec!(),
            by_name: HashMap::new(),
            levels: HashMap::new(),
        }
    }

    pub fn get_lints<'t>(&'t self) -> &'t [(&'static Lint, bool)] {
        self.lints.as_slice()
    }

    pub fn register_pass(&mut self, sess: Option<&Session>,
                         from_plugin: bool, pass: LintPassObject) {
        for &lint in pass.get_lints().iter() {
            self.lints.push((lint, from_plugin));

            let id = LintId::of(lint);
            if !self.by_name.insert(lint.name, id) {
                let msg = format!("duplicate specification of lint {}", lint.name);
                match (sess, from_plugin) {
                    // We load builtin lints first, so a duplicate is a compiler bug.
                    // Use early_error when handling -W help with no crate.
                    (None, _) => early_error(msg.as_slice()),
                    (Some(sess), false) => sess.bug(msg.as_slice()),

                    // A duplicate name from a plugin is a user error.
                    (Some(sess), true)  => sess.err(msg.as_slice()),
                }
            }

            if lint.default_level != Allow {
                self.levels.insert(id, (lint.default_level, Default));
            }
        }
        self.passes.push(RefCell::new(pass));
    }

    pub fn register_builtin(&mut self, sess: Option<&Session>) {
        macro_rules! add_builtin_lints ( ( $sess:ident, $($name:ident),*, ) => (
            {$(
                {
                    let obj: builtin::$name = Default::default();
                    self.register_pass($sess, false, box obj as LintPassObject);
                };
            )*}
        ))

        add_builtin_lints!(sess,
            WhileTrue, UnusedCasts, TypeLimits, CTypes, HeapMemory,
            RawPointerDeriving, UnusedAttribute, PathStatement,
            UnusedResult, DeprecatedOwnedVector, NonCamelCaseTypes,
            NonSnakeCaseFunctions, NonUppercaseStatics,
            NonUppercasePatternStatics, UppercaseVariables,
            UnnecessaryParens, UnusedUnsafe, UnsafeBlock, UnusedMut,
            UnnecessaryAllocation, MissingDoc, Stability,

            GatherNodeLevels, HardwiredLints,
        )
    }

    pub fn process_command_line(&mut self, sess: &Session) {
        for &(ref lint_name, level) in sess.opts.lint_opts.iter() {
            match self.by_name.find_equiv(&lint_name.as_slice()) {
                Some(&lint_id) => self.set_level(lint_id, (level, CommandLine)),
                None => sess.err(format!("unknown {} flag: {}",
                    level.as_str(), lint_name).as_slice()),
            }
        }
    }
}

/// Context for lint checking.
pub struct Context<'a> {
    /// The store of registered lints.
    lints: LintStore,

    /// Context we're checking in (used to access fields like sess).
    tcx: &'a ty::ctxt,

    /// When recursing into an attributed node of the ast which modifies lint
    /// levels, this stack keeps track of the previous lint levels of whatever
    /// was modified.
    level_stack: Vec<(LintId, LevelSource)>,

    /// Level of lints for certain NodeIds, stored here because the body of
    /// the lint needs to run in trans.
    node_levels: RefCell<HashMap<(ast::NodeId, LintId), LevelSource>>,
}

/// Convenience macro for calling a `LintPass` method on every pass in the context.
macro_rules! run_lints ( ($cx:expr, $f:ident, $($args:expr),*) => (
    for obj in $cx.lints.passes.iter() {
        obj.borrow_mut().$f($cx, $($args),*);
    }
))

/// Emit a lint as a `span_warn` or `span_err` (or not at all)
/// according to `level`.  This lives outside of `Context` so
/// it can be used by checks in trans that run after the main
/// lint phase is finished.
pub fn emit_lint(sess: &Session, lint: &'static Lint,
                 lvlsrc: LevelSource, span: Span, msg: &str) {
    let (level, source) = lvlsrc;
    if level == Allow { return }

    let mut note = None;
    let msg = match source {
        Default => {
            format!("{}, #[{}({})] on by default", msg,
                level_to_str(level), lint_str)
        },
        CommandLine => {
            format!("{} [-{} {}]", msg,
                match level {
                    Warn => 'W', Deny => 'D', Forbid => 'F',
                    Allow => fail!()
                }, lint.name.replace("_", "-"))
        },
        Node(src) => {
            note = Some(src);
            msg.to_string()
        }
    };

    match level {
        Warn =>          { sess.span_warn(span, msg.as_slice()); }
        Deny | Forbid => { sess.span_err(span, msg.as_slice());  }
        Allow => fail!(),
    }

    for span in note.move_iter() {
        sess.span_note(span, "lint level defined here");
    }
}

impl<'a> Context<'a> {
    pub fn span_lint(&self, lint: &'static Lint, span: Span, msg: &str) {
        let (level, src) = match self.lints.levels.find(&LintId::of(lint)) {
            None => return,
            Some(&(Warn, src))
                => (self.lints.get_level_source(LintId::of(builtin::warnings)).val0(), src),
            Some(&pair) => pair,
        };

        emit_lint(&self.tcx.sess, lint, (level, src), span, msg);
    }

    /**
     * Merge the lints specified by any lint attributes into the
     * current lint context, call the provided function, then reset the
     * lints in effect to their previous state.
     */
    fn with_lint_attrs(&mut self,
                       attrs: &[ast::Attribute],
                       f: |&mut Context|) {
        // Parse all of the lint attributes, and then add them all to the
        // current dictionary of lint information. Along the way, keep a history
        // of what we changed so we can roll everything back after invoking the
        // specified closure
        let lint_attrs = self.gather_lint_attrs(attrs);
        let mut pushed = 0u;
        for (lint_id, level, span) in lint_attrs.move_iter() {
            let now = self.lints.get_level_source(lint_id).val0();
            if now == Forbid && level != Forbid {
                let lint_name = lint_id.as_str();
                self.tcx.sess.span_err(span,
                format!("{}({}) overruled by outer forbid({})",
                        level.as_str(), lint_name, lint_name).as_slice());
            } else if now != level {
                let src = self.lints.get_level_source(lint_id).val1();
                self.level_stack.push((lint_id, (now, src)));
                pushed += 1;
                self.lints.set_level(lint_id, (level, Node(span)));
            }
        }

        run_lints!(self, enter_lint_attrs, attrs);
        f(self);
        run_lints!(self, exit_lint_attrs, attrs);

        // rollback
        for _ in range(0, pushed) {
            let (lint, lvlsrc) = self.level_stack.pop().unwrap();
            self.lints.set_level(lint, lvlsrc);
        }
    }

    fn visit_ids(&self, f: |&mut ast_util::IdVisitor<Context>|) {
        let mut v = ast_util::IdVisitor {
            operation: self,
            pass_through_items: false,
            visited_outermost: false,
        };
        f(&mut v);
    }

    fn insert_node_level(&self, id: ast::NodeId, lint: LintId, lvlsrc: LevelSource) {
        self.node_levels.borrow_mut().insert((id, lint), lvlsrc);
    }

    fn gather_lint_attrs(&mut self, attrs: &[ast::Attribute]) -> Vec<(LintId, Level, Span)> {
        // Doing this as an iterator is messy due to multiple borrowing.
        // Allocating and copying these should be quick.
        let mut out = vec!();
        for attr in attrs.iter() {
            let level = match Level::from_str(attr.name().get()) {
                None => continue,
                Some(lvl) => lvl,
            };

            attr::mark_used(attr);

            let meta = attr.node.value;
            let metas = match meta.node {
                ast::MetaList(_, ref metas) => metas,
                _ => {
                    self.tcx.sess.span_err(meta.span, "malformed lint attribute");
                    continue;
                }
            };

            for meta in metas.iter() {
                match meta.node {
                    ast::MetaWord(ref lint_name) => {
                        match self.lints.by_name.find_equiv(lint_name) {
                            Some(lint_id) => out.push((*lint_id, level, meta.span)),

                            None => self.span_lint(builtin::unrecognized_lint,
                                meta.span,
                                format!("unknown `{}` attribute: `{}`",
                                    level.as_str(), lint_name).as_slice()),
                        }
                    }
                    _ => self.tcx.sess.span_err(meta.span, "malformed lint attribute"),
                }
            }
        }
        out
    }
}

impl<'a> AstConv for Context<'a>{
    fn tcx<'a>(&'a self) -> &'a ty::ctxt { self.tcx }

    fn get_item_ty(&self, id: ast::DefId) -> ty::ty_param_bounds_and_ty {
        ty::lookup_item_type(self.tcx, id)
    }

    fn get_trait_def(&self, id: ast::DefId) -> Rc<ty::TraitDef> {
        ty::lookup_trait_def(self.tcx, id)
    }

    fn ty_infer(&self, _span: Span) -> ty::t {
        infer::new_infer_ctxt(self.tcx).next_ty_var()
    }
}

impl<'a> Visitor<()> for Context<'a> {
    fn visit_item(&mut self, it: &ast::Item, _: ()) {
        self.with_lint_attrs(it.attrs.as_slice(), |cx| {
            run_lints!(cx, check_item, it);
            cx.visit_ids(|v| v.visit_item(it, ()));
            visit::walk_item(cx, it, ());
        })
    }

    fn visit_foreign_item(&mut self, it: &ast::ForeignItem, _: ()) {
        self.with_lint_attrs(it.attrs.as_slice(), |cx| {
            run_lints!(cx, check_foreign_item, it);
            visit::walk_foreign_item(cx, it, ());
        })
    }

    fn visit_view_item(&mut self, i: &ast::ViewItem, _: ()) {
        self.with_lint_attrs(i.attrs.as_slice(), |cx| {
            run_lints!(cx, check_view_item, i);
            cx.visit_ids(|v| v.visit_view_item(i, ()));
            visit::walk_view_item(cx, i, ());
        })
    }

    fn visit_pat(&mut self, p: &ast::Pat, _: ()) {
        run_lints!(self, check_pat, p);
        visit::walk_pat(self, p, ());
    }

    fn visit_expr(&mut self, e: &ast::Expr, _: ()) {
        run_lints!(self, check_expr, e);
        visit::walk_expr(self, e, ());
    }

    fn visit_stmt(&mut self, s: &ast::Stmt, _: ()) {
        run_lints!(self, check_stmt, s);
        visit::walk_stmt(self, s, ());
    }

    fn visit_fn(&mut self, fk: &FnKind, decl: &ast::FnDecl,
                body: &ast::Block, span: Span, id: ast::NodeId, _: ()) {
        match *fk {
            visit::FkMethod(_, _, m) => {
                self.with_lint_attrs(m.attrs.as_slice(), |cx| {
                    run_lints!(cx, check_fn, fk, decl, body, span, id);
                    cx.visit_ids(|v| {
                        v.visit_fn(fk, decl, body, span, id, ());
                    });
                    visit::walk_fn(cx, fk, decl, body, span, ());
                })
            },
            _ => {
                run_lints!(self, check_fn, fk, decl, body, span, id);
                visit::walk_fn(self, fk, decl, body, span, ());
            }
        }
    }

    fn visit_ty_method(&mut self, t: &ast::TypeMethod, _: ()) {
        self.with_lint_attrs(t.attrs.as_slice(), |cx| {
            run_lints!(cx, check_ty_method, t);
            visit::walk_ty_method(cx, t, ());
        })
    }

    fn visit_struct_def(&mut self,
                        s: &ast::StructDef,
                        ident: ast::Ident,
                        g: &ast::Generics,
                        id: ast::NodeId,
                        _: ()) {
        run_lints!(self, check_struct_def, s, ident, g, id);
        visit::walk_struct_def(self, s, ());
        run_lints!(self, check_struct_def_post, s, ident, g, id);
    }

    fn visit_struct_field(&mut self, s: &ast::StructField, _: ()) {
        self.with_lint_attrs(s.node.attrs.as_slice(), |cx| {
            run_lints!(cx, check_struct_field, s);
            visit::walk_struct_field(cx, s, ());
        })
    }

    fn visit_variant(&mut self, v: &ast::Variant, g: &ast::Generics, _: ()) {
        self.with_lint_attrs(v.node.attrs.as_slice(), |cx| {
            run_lints!(cx, check_variant, v, g);
            visit::walk_variant(cx, v, g, ());
        })
    }

    // FIXME(#10894) should continue recursing
    fn visit_ty(&mut self, t: &ast::Ty, _: ()) {
        run_lints!(self, check_ty, t);
    }

    fn visit_ident(&mut self, sp: Span, id: ast::Ident, _: ()) {
        run_lints!(self, check_ident, sp, id);
    }

    fn visit_mod(&mut self, m: &ast::Mod, s: Span, n: ast::NodeId, _: ()) {
        run_lints!(self, check_mod, m, s, n);
        visit::walk_mod(self, m, ());
    }

    fn visit_local(&mut self, l: &ast::Local, _: ()) {
        run_lints!(self, check_local, l);
        visit::walk_local(self, l, ());
    }

    fn visit_block(&mut self, b: &ast::Block, _: ()) {
        run_lints!(self, check_block, b);
        visit::walk_block(self, b, ());
    }

    fn visit_arm(&mut self, a: &ast::Arm, _: ()) {
        run_lints!(self, check_arm, a);
        visit::walk_arm(self, a, ());
    }

    fn visit_decl(&mut self, d: &ast::Decl, _: ()) {
        run_lints!(self, check_decl, d);
        visit::walk_decl(self, d, ());
    }

    fn visit_expr_post(&mut self, e: &ast::Expr, _: ()) {
        run_lints!(self, check_expr_post, e);
    }

    fn visit_generics(&mut self, g: &ast::Generics, _: ()) {
        run_lints!(self, check_generics, g);
        visit::walk_generics(self, g, ());
    }

    fn visit_trait_method(&mut self, m: &ast::TraitMethod, _: ()) {
        run_lints!(self, check_trait_method, m);
        visit::walk_trait_method(self, m, ());
    }

    fn visit_opt_lifetime_ref(&mut self, sp: Span, lt: &Option<ast::Lifetime>, _: ()) {
        run_lints!(self, check_opt_lifetime_ref, sp, lt);
    }

    fn visit_lifetime_ref(&mut self, lt: &ast::Lifetime, _: ()) {
        run_lints!(self, check_lifetime_ref, lt);
    }

    fn visit_lifetime_decl(&mut self, lt: &ast::Lifetime, _: ()) {
        run_lints!(self, check_lifetime_decl, lt);
    }

    fn visit_explicit_self(&mut self, es: &ast::ExplicitSelf, _: ()) {
        run_lints!(self, check_explicit_self, es);
        visit::walk_explicit_self(self, es, ());
    }

    fn visit_mac(&mut self, mac: &ast::Mac, _: ()) {
        run_lints!(self, check_mac, mac);
        visit::walk_mac(self, mac, ());
    }

    fn visit_path(&mut self, p: &ast::Path, id: ast::NodeId, _: ()) {
        run_lints!(self, check_path, p, id);
        visit::walk_path(self, p, ());
    }

    fn visit_attribute(&mut self, attr: &ast::Attribute, _: ()) {
        run_lints!(self, check_attribute, attr);
    }
}

// Output any lints that were previously added to the session.
impl<'a> IdVisitingOperation for Context<'a> {
    fn visit_id(&self, id: ast::NodeId) {
        match self.tcx.sess.lints.borrow_mut().pop(&id) {
            None => {}
            Some(lints) => {
                for (lint_id, span, msg) in lints.move_iter() {
                    self.span_lint(lint_id.lint, span, msg.as_slice())
                }
            }
        }
    }
}

pub fn check_crate(tcx: &ty::ctxt,
                   exported_items: &ExportedItems,
                   krate: &ast::Crate) {

    // We want to own the lint store, so move it out of the session.
    let lint_store = mem::replace(&mut *tcx.sess.lint_store.borrow_mut(),
        LintStore::new());

    let mut cx = Context {
        lints: lint_store,
        tcx: tcx,
        level_stack: Vec::new(),
        node_levels: RefCell::new(HashMap::new()),
    };

    // Visit the whole crate.
    cx.with_lint_attrs(krate.attrs.as_slice(), |cx| {
        cx.visit_id(ast::CRATE_NODE_ID);
        cx.visit_ids(|v| {
            v.visited_outermost = true;
            visit::walk_crate(v, krate, ());
        });

        // since the root module isn't visited as an item (because it isn't an
        // item), warn for it here.
        run_lints!(cx, check_crate, exported_items, krate);

        visit::walk_crate(cx, krate, ());
    });

    // If we missed any lints added to the session, then there's a bug somewhere
    // in the iteration code.
    for (id, v) in tcx.sess.lints.borrow().iter() {
        for &(lint, span, ref msg) in v.iter() {
            tcx.sess.span_bug(span,
                format!("unprocessed lint {} at {}: {}",
                    lint.as_str(), tcx.map.node_to_str(*id), *msg)
                .as_slice())
        }
    }

    tcx.sess.abort_if_errors();
    *tcx.node_lint_levels.borrow_mut() = cx.node_levels.unwrap();
}
