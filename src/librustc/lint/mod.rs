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

use driver::session;
use middle::dead::DEAD_CODE_LINT_STR;
use middle::privacy::ExportedItems;
use middle::ty;
use middle::typeck::astconv::AstConv;
use middle::typeck::infer;

use std::collections::HashMap;
use std::rc::Rc;
use std::gc::Gc;
use std::to_str::ToStr;
use std::cell::RefCell;
use std::default::Default;
use std::collections::SmallIntMap;
use syntax::ast_util::IdVisitingOperation;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use syntax::visit::{Visitor, FnKind};
use syntax::{ast, ast_util, visit};

mod builtin;

/// Trait for types providing lint checks. Each method checks a single syntax
/// node, and should not invoke methods recursively (unlike `Visitor`).  Each
/// method has a default do-nothing implementation. The trait also contains a
/// few lint-specific methods with no equivalent in `Visitor`.
//
// FIXME: eliminate the duplication with `Visitor`
trait LintPass {
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

#[deriving(Clone, Show, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum LintId {
    CTypes,
    UnusedImports,
    UnnecessaryQualification,
    WhileTrue,
    PathStatement,
    UnrecognizedLint,
    NonCamelCaseTypes,
    NonUppercaseStatics,
    NonUppercasePatternStatics,
    NonSnakeCaseFunctions,
    UppercaseVariables,
    UnnecessaryParens,
    TypeLimits,
    TypeOverflow,
    UnusedUnsafe,
    UnsafeBlock,
    UnusedAttribute,
    UnknownFeatures,
    UnknownCrateType,
    UnsignedNegate,
    VariantSizeDifference,

    ManagedHeapMemory,
    OwnedHeapMemory,
    HeapMemory,

    UnusedVariable,
    DeadAssignment,
    UnusedMut,
    UnnecessaryAllocation,
    DeadCode,
    VisiblePrivateTypes,
    UnnecessaryTypecast,

    MissingDoc,
    UnreachableCode,

    Deprecated,
    Experimental,
    Unstable,

    UnusedMustUse,
    UnusedResult,

    Warnings,

    RawPointerDeriving,
}

pub fn level_to_str(lv: Level) -> &'static str {
    match lv {
      Allow => "allow",
      Warn => "warn",
      Deny => "deny",
      Forbid => "forbid"
    }
}

#[deriving(Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum Level {
    Allow, Warn, Deny, Forbid
}

#[deriving(Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct LintSpec {
    pub default: Level,
    pub lint: LintId,
    pub desc: &'static str,
}

pub type LintDict = HashMap<&'static str, LintSpec>;

// this is public for the lints that run in trans
#[deriving(PartialEq)]
pub enum LintSource {
    Node(Span),
    Default,
    CommandLine
}

static lint_table: &'static [(&'static str, LintSpec)] = &[
    ("ctypes",
     LintSpec {
        lint: CTypes,
        desc: "proper use of libc types in foreign modules",
        default: Warn
     }),

    ("unused_imports",
     LintSpec {
        lint: UnusedImports,
        desc: "imports that are never used",
        default: Warn
     }),

    ("unnecessary_qualification",
     LintSpec {
        lint: UnnecessaryQualification,
        desc: "detects unnecessarily qualified names",
        default: Allow
     }),

    ("while_true",
     LintSpec {
        lint: WhileTrue,
        desc: "suggest using `loop { }` instead of `while true { }`",
        default: Warn
     }),

    ("path_statement",
     LintSpec {
        lint: PathStatement,
        desc: "path statements with no effect",
        default: Warn
     }),

    ("unrecognized_lint",
     LintSpec {
        lint: UnrecognizedLint,
        desc: "unrecognized lint attribute",
        default: Warn
     }),

    ("non_camel_case_types",
     LintSpec {
        lint: NonCamelCaseTypes,
        desc: "types, variants and traits should have camel case names",
        default: Warn
     }),

    ("non_uppercase_statics",
     LintSpec {
         lint: NonUppercaseStatics,
         desc: "static constants should have uppercase identifiers",
         default: Allow
     }),

    ("non_uppercase_pattern_statics",
     LintSpec {
         lint: NonUppercasePatternStatics,
         desc: "static constants in match patterns should be all caps",
         default: Warn
     }),

    ("non_snake_case_functions",
     LintSpec {
         lint: NonSnakeCaseFunctions,
         desc: "methods and functions should have snake case names",
         default: Warn
     }),

    ("uppercase_variables",
     LintSpec {
         lint: UppercaseVariables,
         desc: "variable and structure field names should start with a lowercase character",
         default: Warn
     }),

     ("unnecessary_parens",
     LintSpec {
        lint: UnnecessaryParens,
        desc: "`if`, `match`, `while` and `return` do not need parentheses",
        default: Warn
     }),

    ("managed_heap_memory",
     LintSpec {
        lint: ManagedHeapMemory,
        desc: "use of managed (@ type) heap memory",
        default: Allow
     }),

    ("owned_heap_memory",
     LintSpec {
        lint: OwnedHeapMemory,
        desc: "use of owned (Box type) heap memory",
        default: Allow
     }),

    ("heap_memory",
     LintSpec {
        lint: HeapMemory,
        desc: "use of any (Box type or @ type) heap memory",
        default: Allow
     }),

    ("type_limits",
     LintSpec {
        lint: TypeLimits,
        desc: "comparisons made useless by limits of the types involved",
        default: Warn
     }),

    ("type_overflow",
     LintSpec {
        lint: TypeOverflow,
        desc: "literal out of range for its type",
        default: Warn
     }),


    ("unused_unsafe",
     LintSpec {
        lint: UnusedUnsafe,
        desc: "unnecessary use of an `unsafe` block",
        default: Warn
    }),

    ("unsafe_block",
     LintSpec {
        lint: UnsafeBlock,
        desc: "usage of an `unsafe` block",
        default: Allow
    }),

    ("unused_attribute",
     LintSpec {
         lint: UnusedAttribute,
         desc: "detects attributes that were not used by the compiler",
         default: Warn
    }),

    ("unused_variable",
     LintSpec {
        lint: UnusedVariable,
        desc: "detect variables which are not used in any way",
        default: Warn
    }),

    ("dead_assignment",
     LintSpec {
        lint: DeadAssignment,
        desc: "detect assignments that will never be read",
        default: Warn
    }),

    ("unnecessary_typecast",
     LintSpec {
        lint: UnnecessaryTypecast,
        desc: "detects unnecessary type casts, that can be removed",
        default: Allow,
    }),

    ("unused_mut",
     LintSpec {
        lint: UnusedMut,
        desc: "detect mut variables which don't need to be mutable",
        default: Warn
    }),

    ("unnecessary_allocation",
     LintSpec {
        lint: UnnecessaryAllocation,
        desc: "detects unnecessary allocations that can be eliminated",
        default: Warn
    }),

    (DEAD_CODE_LINT_STR,
     LintSpec {
        lint: DeadCode,
        desc: "detect piece of code that will never be used",
        default: Warn
    }),
    ("visible_private_types",
     LintSpec {
        lint: VisiblePrivateTypes,
        desc: "detect use of private types in exported type signatures",
        default: Warn
    }),

    ("missing_doc",
     LintSpec {
        lint: MissingDoc,
        desc: "detects missing documentation for public members",
        default: Allow
    }),

    ("unreachable_code",
     LintSpec {
        lint: UnreachableCode,
        desc: "detects unreachable code",
        default: Warn
    }),

    ("deprecated",
     LintSpec {
        lint: Deprecated,
        desc: "detects use of #[deprecated] items",
        default: Warn
    }),

    ("experimental",
     LintSpec {
        lint: Experimental,
        desc: "detects use of #[experimental] items",
        // FIXME #6875: Change to Warn after std library stabilization is complete
        default: Allow
    }),

    ("unstable",
     LintSpec {
        lint: Unstable,
        desc: "detects use of #[unstable] items (incl. items with no stability attribute)",
        default: Allow
    }),

    ("warnings",
     LintSpec {
        lint: Warnings,
        desc: "mass-change the level for lints which produce warnings",
        default: Warn
    }),

    ("unknown_features",
     LintSpec {
        lint: UnknownFeatures,
        desc: "unknown features found in crate-level #[feature] directives",
        default: Deny,
    }),

    ("unknown_crate_type",
    LintSpec {
        lint: UnknownCrateType,
        desc: "unknown crate type found in #[crate_type] directive",
        default: Deny,
    }),

    ("unsigned_negate",
    LintSpec {
        lint: UnsignedNegate,
        desc: "using an unary minus operator on unsigned type",
        default: Warn
    }),

    ("variant_size_difference",
    LintSpec {
        lint: VariantSizeDifference,
        desc: "detects enums with widely varying variant sizes",
        default: Allow,
    }),

    ("unused_must_use",
    LintSpec {
        lint: UnusedMustUse,
        desc: "unused result of a type flagged as #[must_use]",
        default: Warn,
    }),

    ("unused_result",
    LintSpec {
        lint: UnusedResult,
        desc: "unused result of an expression in a statement",
        default: Allow,
    }),

    ("raw_pointer_deriving",
     LintSpec {
        lint: RawPointerDeriving,
        desc: "uses of #[deriving] with raw pointers are rarely correct",
        default: Warn,
    }),
];

/*
  Pass names should not contain a '-', as the compiler normalizes
  '-' to '_' in command-line flags
 */
pub fn get_lint_dict() -> LintDict {
    lint_table.iter().map(|&(k, v)| (k, v)).collect()
}

struct Context<'a> {
    /// All known lint modes (string versions)
    dict: LintDict,
    /// Current levels of each lint warning
    cur: SmallIntMap<(Level, LintSource)>,
    /// Context we're checking in (used to access fields like sess)
    tcx: &'a ty::ctxt,

    /// When recursing into an attributed node of the ast which modifies lint
    /// levels, this stack keeps track of the previous lint levels of whatever
    /// was modified.
    level_stack: Vec<(LintId, Level, LintSource)>,

    /// Level of lints for certain NodeIds, stored here because the body of
    /// the lint needs to run in trans.
    node_levels: RefCell<HashMap<(ast::NodeId, LintId), (Level, LintSource)>>,

    /// Trait objects for each lint.
    lints: Vec<RefCell<LintPassObject>>,
}

/// Convenience macro for calling a `LintPass` method on every lint in the context.
macro_rules! run_lints ( ($cx:expr, $f:ident, $($args:expr),*) => (
    for tl in $cx.lints.iter() {
        tl.borrow_mut().$f($cx, $($args),*);
    }
))

pub fn emit_lint(level: Level, src: LintSource, msg: &str, span: Span,
                 lint_str: &str, tcx: &ty::ctxt) {
    if level == Allow { return }

    let mut note = None;
    let msg = match src {
        Default => {
            format!("{}, #[{}({})] on by default", msg,
                level_to_str(level), lint_str)
        },
        CommandLine => {
            format!("{} [-{} {}]", msg,
                match level {
                    Warn => 'W', Deny => 'D', Forbid => 'F',
                    Allow => fail!()
                }, lint_str.replace("_", "-"))
        },
        Node(src) => {
            note = Some(src);
            msg.to_str()
        }
    };

    match level {
        Warn =>          { tcx.sess.span_warn(span, msg.as_slice()); }
        Deny | Forbid => { tcx.sess.span_err(span, msg.as_slice());  }
        Allow => fail!(),
    }

    for &span in note.iter() {
        tcx.sess.span_note(span, "lint level defined here");
    }
}

pub fn lint_to_str(lint: LintId) -> &'static str {
    for &(name, lspec) in lint_table.iter() {
        if lspec.lint == lint {
            return name;
        }
    }

    fail!("unrecognized lint: {}", lint);
}

impl<'a> Context<'a> {
    fn get_level(&self, lint: LintId) -> Level {
        match self.cur.find(&(lint as uint)) {
          Some(&(lvl, _)) => lvl,
          None => Allow
        }
    }

    fn get_source(&self, lint: LintId) -> LintSource {
        match self.cur.find(&(lint as uint)) {
          Some(&(_, src)) => src,
          None => Default
        }
    }

    fn set_level(&mut self, lint: LintId, level: Level, src: LintSource) {
        if level == Allow {
            self.cur.remove(&(lint as uint));
        } else {
            self.cur.insert(lint as uint, (level, src));
        }
    }

    fn lint_to_str(&self, lint: LintId) -> &'static str {
        for (k, v) in self.dict.iter() {
            if v.lint == lint {
                return *k;
            }
        }
        fail!("unregistered lint {}", lint);
    }

    fn span_lint(&self, lint: LintId, span: Span, msg: &str) {
        let (level, src) = match self.cur.find(&(lint as uint)) {
            None => { return }
            Some(&(Warn, src)) => (self.get_level(Warnings), src),
            Some(&pair) => pair,
        };

        emit_lint(level, src, msg, span, self.lint_to_str(lint), self.tcx);
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
        let mut pushed = 0u;
        each_lint(&self.tcx.sess, attrs, |meta, level, lintname| {
            match self.dict.find_equiv(&lintname) {
                None => {
                    self.span_lint(
                        UnrecognizedLint,
                        meta.span,
                        format!("unknown `{}` attribute: `{}`",
                                level_to_str(level), lintname).as_slice());
                }
                Some(lint) => {
                    let lint = lint.lint;
                    let now = self.get_level(lint);
                    if now == Forbid && level != Forbid {
                        self.tcx.sess.span_err(meta.span,
                        format!("{}({}) overruled by outer forbid({})",
                                level_to_str(level),
                                lintname,
                                lintname).as_slice());
                    } else if now != level {
                        let src = self.get_source(lint);
                        self.level_stack.push((lint, now, src));
                        pushed += 1;
                        self.set_level(lint, level, Node(meta.span));
                    }
                }
            }
            true
        });

        run_lints!(self, enter_lint_attrs, attrs);
        f(self);
        run_lints!(self, exit_lint_attrs, attrs);

        // rollback
        for _ in range(0, pushed) {
            let (lint, lvl, src) = self.level_stack.pop().unwrap();
            self.set_level(lint, lvl, src);
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

    fn insert_node_level(&self, id: ast::NodeId, lint: LintId, lvl: Level, src: LintSource) {
        self.node_levels.borrow_mut().insert((id, lint), (lvl, src));
    }
}

/// Check that every lint from the list of attributes satisfies `f`.
/// Return true if that's the case. Otherwise return false.
pub fn each_lint(sess: &session::Session,
                 attrs: &[ast::Attribute],
                 f: |Gc<ast::MetaItem>, Level, InternedString| -> bool)
                 -> bool {
    let xs = [Allow, Warn, Deny, Forbid];
    for &level in xs.iter() {
        let level_name = level_to_str(level);
        for attr in attrs.iter().filter(|m| m.check_name(level_name)) {
            let meta = attr.node.value;
            let metas = match meta.node {
                ast::MetaList(_, ref metas) => metas,
                _ => {
                    sess.span_err(meta.span, "malformed lint attribute");
                    continue;
                }
            };
            for meta in metas.iter() {
                match meta.node {
                    ast::MetaWord(ref lintname) => {
                        if !f(*meta, level, (*lintname).clone()) {
                            return false;
                        }
                    }
                    _ => {
                        sess.span_err(meta.span, "malformed lint attribute");
                    }
                }
            }
        }
    }
    true
}

/// Check from a list of attributes if it contains the appropriate
/// `#[level(lintname)]` attribute (e.g. `#[allow(dead_code)]).
pub fn contains_lint(attrs: &[ast::Attribute],
                     level: Level,
                     lintname: &'static str)
                     -> bool {
    let level_name = level_to_str(level);
    for attr in attrs.iter().filter(|m| m.name().equiv(&level_name)) {
        if attr.meta_item_list().is_none() {
            continue
        }
        let list = attr.meta_item_list().unwrap();
        for meta_item in list.iter() {
            if meta_item.name().equiv(&lintname) {
                return true;
            }
        }
    }
    false
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

impl<'a> IdVisitingOperation for Context<'a> {
    fn visit_id(&self, id: ast::NodeId) {
        match self.tcx.sess.lints.borrow_mut().pop(&id) {
            None => {}
            Some(l) => {
                for (lint, span, msg) in l.move_iter() {
                    self.span_lint(lint, span, msg.as_slice())
                }
            }
        }
    }
}

pub fn check_crate(tcx: &ty::ctxt,
                   exported_items: &ExportedItems,
                   krate: &ast::Crate) {
    macro_rules! builtin_lints (( $($name:ident),*, ) => (
        vec!($(
            {
                let obj: builtin::$name = Default::default();
                RefCell::new(box obj as LintPassObject)
            }
        ),*)
    ))

    let builtin_lints = builtin_lints!(
        GatherNodeLevels, WhileTrue, UnusedCasts, TypeLimits, CTypes,
        HeapMemory, RawPointerDeriving, UnusedAttribute,
        PathStatement, UnusedMustUse, DeprecatedOwnedVector,
        NonCamelCaseTypes, NonSnakeCaseFunctions, NonUppercaseStatics,
        UppercaseVariables, UnnecessaryParens, UnusedUnsafe, UnsafeBlock,
        UnusedMut, UnnecessaryAllocation, MissingDoc, Stability,
    );

    let mut cx = Context {
        dict: get_lint_dict(),
        cur: SmallIntMap::new(),
        tcx: tcx,
        level_stack: Vec::new(),
        node_levels: RefCell::new(HashMap::new()),
        lints: builtin_lints,
    };

    // Install default lint levels, followed by the command line levels, and
    // then actually visit the whole crate.
    for (_, spec) in cx.dict.iter() {
        if spec.default != Allow {
            cx.cur.insert(spec.lint as uint, (spec.default, Default));
        }
    }
    for &(lint, level) in tcx.sess.opts.lint_opts.iter() {
        cx.set_level(lint, level, CommandLine);
    }
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
            tcx.sess.span_bug(span, format!("unprocessed lint {} at {}: {}",
                                            lint, tcx.map.node_to_str(*id), *msg).as_slice())
        }
    }

    tcx.sess.abort_if_errors();
    *tcx.node_lint_levels.borrow_mut() = cx.node_levels.unwrap();
}
