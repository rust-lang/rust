// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::session;
use middle::ty;
use middle::pat_util;
use util::ppaux::{ty_to_str};

use std::cmp;
use std::hashmap::HashMap;
use std::i16;
use std::i32;
use std::i64;
use std::i8;
use std::u16;
use std::u32;
use std::u64;
use std::u8;
use extra::smallintmap::SmallIntMap;
use syntax::ast_map;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::span;
use syntax::codemap;
use syntax::parse::token;
use syntax::{ast, ast_util, visit};
use syntax::visit::Visitor;

/**
 * A 'lint' check is a kind of miscellaneous constraint that a user _might_
 * want to enforce, but might reasonably want to permit as well, on a
 * module-by-module basis. They contrast with static constraints enforced by
 * other phases of the compiler, which are generally required to hold in order
 * to compile the program at all.
 *
 * The lint checking is all consolidated into one pass which runs just before
 * translation to LLVM bytecode. Throughout compilation, lint warnings can be
 * added via the `add_lint` method on the Session structure. This requires a
 * span and an id of the node that the lint is being added to. The lint isn't
 * actually emitted at that time because it is unknown what the actual lint
 * level at that location is.
 *
 * To actually emit lint warnings/errors, a separate pass is used just before
 * translation. A context keeps track of the current state of all lint levels.
 * Upon entering a node of the ast which can modify the lint settings, the
 * previous lint state is pushed onto a stack and the ast is then recursed upon.
 * As the ast is traversed, this keeps track of the current lint level for all
 * lint attributes.
 *
 * At each node of the ast which can modify lint attributes, all known lint
 * passes are also applied.  Each lint pass is an oldvisit::vt<()> structure.
 * The visitors are constructed via the lint_*() functions below. There are
 * also some lint checks which operate directly on ast nodes (such as
 * @ast::item), and those are organized as check_item_*(). Each visitor added
 * to the lint context is modified to stop once it reaches a node which could
 * alter the lint levels. This means that everything is looked at once and
 * only once by every lint pass.
 *
 * With this all in place, to add a new lint warning, all you need to do is to
 * either invoke `add_lint` on the session at the appropriate time, or write a
 * lint pass in this module which is just an ast visitor. The context used when
 * traversing the ast has a `span_lint` method which only needs the span of the
 * item that's being warned about.
 */

#[deriving(Clone, Eq)]
pub enum lint {
    ctypes,
    cstack,
    unused_imports,
    unnecessary_qualification,
    while_true,
    path_statement,
    unrecognized_lint,
    non_camel_case_types,
    non_uppercase_statics,
    type_limits,
    unused_unsafe,

    managed_heap_memory,
    owned_heap_memory,
    heap_memory,

    unused_variable,
    dead_assignment,
    unused_mut,
    unnecessary_allocation,

    missing_doc,
    unreachable_code,

    warnings,
}

pub fn level_to_str(lv: level) -> &'static str {
    match lv {
      allow => "allow",
      warn => "warn",
      deny => "deny",
      forbid => "forbid"
    }
}

#[deriving(Clone, Eq, Ord)]
pub enum level {
    allow, warn, deny, forbid
}

#[deriving(Clone, Eq)]
pub struct LintSpec {
    lint: lint,
    desc: &'static str,
    default: level
}

impl Ord for LintSpec {
    fn lt(&self, other: &LintSpec) -> bool { self.default < other.default }
}

pub type LintDict = HashMap<&'static str, LintSpec>;

enum AttributedNode<'self> {
    Item(@ast::item),
    Method(&'self ast::method),
    Crate(@ast::Crate),
}

#[deriving(Eq)]
enum LintSource {
    Node(span),
    Default,
    CommandLine
}

static lint_table: &'static [(&'static str, LintSpec)] = &[
    ("ctypes",
     LintSpec {
        lint: ctypes,
        desc: "proper use of std::libc types in foreign modules",
        default: warn
     }),

    ("cstack",
     LintSpec {
        lint: cstack,
        desc: "only invoke foreign functions from fixedstacksegment fns",
        default: deny
     }),

    ("unused_imports",
     LintSpec {
        lint: unused_imports,
        desc: "imports that are never used",
        default: warn
     }),

    ("unnecessary_qualification",
     LintSpec {
        lint: unnecessary_qualification,
        desc: "detects unnecessarily qualified names",
        default: allow
     }),

    ("while_true",
     LintSpec {
        lint: while_true,
        desc: "suggest using loop { } instead of while(true) { }",
        default: warn
     }),

    ("path_statement",
     LintSpec {
        lint: path_statement,
        desc: "path statements with no effect",
        default: warn
     }),

    ("unrecognized_lint",
     LintSpec {
        lint: unrecognized_lint,
        desc: "unrecognized lint attribute",
        default: warn
     }),

    ("non_camel_case_types",
     LintSpec {
        lint: non_camel_case_types,
        desc: "types, variants and traits should have camel case names",
        default: allow
     }),

    ("non_uppercase_statics",
     LintSpec {
         lint: non_uppercase_statics,
         desc: "static constants should have uppercase identifiers",
         default: allow
     }),

    ("managed_heap_memory",
     LintSpec {
        lint: managed_heap_memory,
        desc: "use of managed (@ type) heap memory",
        default: allow
     }),

    ("owned_heap_memory",
     LintSpec {
        lint: owned_heap_memory,
        desc: "use of owned (~ type) heap memory",
        default: allow
     }),

    ("heap_memory",
     LintSpec {
        lint: heap_memory,
        desc: "use of any (~ type or @ type) heap memory",
        default: allow
     }),

    ("type_limits",
     LintSpec {
        lint: type_limits,
        desc: "comparisons made useless by limits of the types involved",
        default: warn
     }),

    ("unused_unsafe",
     LintSpec {
        lint: unused_unsafe,
        desc: "unnecessary use of an `unsafe` block",
        default: warn
    }),

    ("unused_variable",
     LintSpec {
        lint: unused_variable,
        desc: "detect variables which are not used in any way",
        default: warn
    }),

    ("dead_assignment",
     LintSpec {
        lint: dead_assignment,
        desc: "detect assignments that will never be read",
        default: warn
    }),

    ("unused_mut",
     LintSpec {
        lint: unused_mut,
        desc: "detect mut variables which don't need to be mutable",
        default: warn
    }),

    ("unnecessary_allocation",
     LintSpec {
        lint: unnecessary_allocation,
        desc: "detects unnecessary allocations that can be eliminated",
        default: warn
    }),

    ("missing_doc",
     LintSpec {
        lint: missing_doc,
        desc: "detects missing documentation for public members",
        default: allow
    }),

    ("unreachable_code",
     LintSpec {
        lint: unreachable_code,
        desc: "detects unreachable code",
        default: warn
    }),

    ("warnings",
     LintSpec {
        lint: warnings,
        desc: "mass-change the level for lints which produce warnings",
        default: warn
    }),
];

/*
  Pass names should not contain a '-', as the compiler normalizes
  '-' to '_' in command-line flags
 */
pub fn get_lint_dict() -> LintDict {
    let mut map = HashMap::new();
    for &(k, v) in lint_table.iter() {
        map.insert(k, v);
    }
    return map;
}

trait OuterLint {
    fn process_item(@mut self, i:@ast::item, e:@mut Context);
    fn process_fn(@mut self, fk:&visit::fn_kind, fd:&ast::fn_decl,
                  b:&ast::Block, s:span, n:ast::NodeId, e:@mut Context);

    // Returned inner variant will not proceed past subitems.
    // Supports decomposition of simple lints into subitem-traversing
    // outer lint visitor and subitem-stopping inner lint visitor.
    fn inner_variant(@mut self) -> @mut InnerLint;
}

trait InnerLint {
    fn descend_item(@mut self, i:&ast::item, e:@mut Context);
    fn descend_crate(@mut self, crate: &ast::Crate, env: @mut Context);
    fn descend_fn(@mut self,
                  function_kind: &visit::fn_kind,
                  function_declaration: &ast::fn_decl,
                  function_body: &ast::Block,
                  sp: span,
                  id: ast::NodeId,
                  env: @mut Context);
}

impl<V:Visitor<@mut Context>> InnerLint for V {
    fn descend_item(@mut self, i:&ast::item, e:@mut Context) {
        visit::walk_item(self, i, e);
    }
    fn descend_crate(@mut self, crate: &ast::Crate, env: @mut Context) {
        visit::walk_crate(self, crate, env);
    }
    fn descend_fn(@mut self, fk: &visit::fn_kind, fd: &ast::fn_decl, fb: &ast::Block,
                  sp: span, id: ast::NodeId, env: @mut Context) {
        visit::walk_fn(self, fk, fd, fb, sp, id, env);
    }
}

enum AnyVisitor {
    // This is a pair so every visitor can visit every node. When a lint pass
    // is registered, another visitor is created which stops at all items
    // which can alter the attributes of the ast. This "item stopping visitor"
    // is the second element of the pair, while the original visitor is the
    // first element. This means that when visiting a node, the original
    // recursive call can use the original visitor's method, although the
    // recursing visitor supplied to the method is the item stopping visitor.
    OldVisitor(@mut OuterLint, @mut InnerLint),
    NewVisitor(@mut visit::Visitor<()>),
}

type VCObj = @mut Visitor<@mut Context>;

struct Context {
    // All known lint modes (string versions)
    dict: @LintDict,
    // Current levels of each lint warning
    curr: SmallIntMap<(level, LintSource)>,
    // context we're checking in (used to access fields like sess)
    tcx: ty::ctxt,
    // Just a simple flag if we're currently recursing into a trait
    // implementation. This is only used by the lint_missing_doc() pass
    in_trait_impl: bool,
    // Another flag for doc lint emissions. Does some parent of the current node
    // have the doc(hidden) attribute? Treating this as allow(missing_doc) would
    // play badly with forbid(missing_doc) when it shouldn't.
    doc_hidden: bool,
    // When recursing into an attributed node of the ast which modifies lint
    // levels, this stack keeps track of the previous lint levels of whatever
    // was modified.
    lint_stack: ~[(lint, level, LintSource)],
    // Each of these visitors represents a lint pass. A number of the lint
    // attributes are registered by adding a visitor to iterate over the ast.
    // Others operate directly on @ast::item structures (or similar). Finally,
    // others still are added to the Session object via `add_lint`, and these
    // are all passed with the lint_session visitor.
    visitors: ~[AnyVisitor],
}

impl Context {
    fn get_level(&self, lint: lint) -> level {
        match self.curr.find(&(lint as uint)) {
          Some(&(lvl, _)) => lvl,
          None => allow
        }
    }

    fn get_source(&self, lint: lint) -> LintSource {
        match self.curr.find(&(lint as uint)) {
          Some(&(_, src)) => src,
          None => Default
        }
    }

    fn set_level(&mut self, lint: lint, level: level, src: LintSource) {
        if level == allow {
            self.curr.remove(&(lint as uint));
        } else {
            self.curr.insert(lint as uint, (level, src));
        }
    }

    fn lint_to_str(&self, lint: lint) -> &'static str {
        for (k, v) in self.dict.iter() {
            if v.lint == lint {
                return *k;
            }
        }
        fail!("unregistered lint %?", lint);
    }

    fn span_lint(&self, lint: lint, span: span, msg: &str) {
        let (level, src) = match self.curr.find(&(lint as uint)) {
            None => { return }
            Some(&(warn, src)) => (self.get_level(warnings), src),
            Some(&pair) => pair,
        };
        if level == allow { return }

        let mut note = None;
        let msg = match src {
            Default | CommandLine => {
                fmt!("%s [-%c %s%s]", msg, match level {
                        warn => 'W', deny => 'D', forbid => 'F',
                        allow => fail!()
                    }, self.lint_to_str(lint).replace("_", "-"),
                    if src == Default { " (default)" } else { "" })
            },
            Node(src) => {
                note = Some(src);
                msg.to_str()
            }
        };
        match level {
            warn =>          { self.tcx.sess.span_warn(span, msg); }
            deny | forbid => { self.tcx.sess.span_err(span, msg);  }
            allow => fail!(),
        }

        for &span in note.iter() {
            self.tcx.sess.span_note(span, "lint level defined here");
        }
    }

    /**
     * Merge the lints specified by any lint attributes into the
     * current lint context, call the provided function, then reset the
     * lints in effect to their previous state.
     */
    fn with_lint_attrs(@mut self, attrs: &[ast::Attribute], f: &fn()) {
        // Parse all of the lint attributes, and then add them all to the
        // current dictionary of lint information. Along the way, keep a history
        // of what we changed so we can roll everything back after invoking the
        // specified closure
        let mut pushed = 0u;
        do each_lint(self.tcx.sess, attrs) |meta, level, lintname| {
            match self.dict.find_equiv(&lintname) {
                None => {
                    self.span_lint(
                        unrecognized_lint,
                        meta.span,
                        fmt!("unknown `%s` attribute: `%s`",
                        level_to_str(level), lintname));
                }
                Some(lint) => {
                    let lint = lint.lint;
                    let now = self.get_level(lint);
                    if now == forbid && level != forbid {
                        self.tcx.sess.span_err(meta.span,
                        fmt!("%s(%s) overruled by outer forbid(%s)",
                        level_to_str(level),
                        lintname, lintname));
                    } else if now != level {
                        let src = self.get_source(lint);
                        self.lint_stack.push((lint, now, src));
                        pushed += 1;
                        self.set_level(lint, level, Node(meta.span));
                    }
                }
            }
            true
        };

        // detect doc(hidden)
        let mut doc_hidden = do attrs.iter().any |attr| {
            "doc" == attr.name() &&
                match attr.meta_item_list() {
                    Some(l) => attr::contains_name(l, "hidden"),
                    None    => false // not of the form #[doc(...)]
                }
        };

        if doc_hidden && !self.doc_hidden {
            self.doc_hidden = true;
        } else {
            doc_hidden = false;
        }

        f();

        // rollback
        if doc_hidden && self.doc_hidden {
            self.doc_hidden = false;
        }
        do pushed.times {
            let (lint, lvl, src) = self.lint_stack.pop();
            self.set_level(lint, lvl, src);
        }
    }

    fn add_oldvisit_lint(&mut self, v: @mut OuterLint) {
        self.visitors.push(OldVisitor(v, v.inner_variant()));
    }

    fn add_lint(&mut self, v: @mut visit::Visitor<()>) {
        self.visitors.push(NewVisitor(v));
    }

    fn process(@mut self, n: AttributedNode) {
        // see comment of the `visitors` field in the struct for why there's a
        // pair instead of just one visitor.
        match n {
            Item(it) => {
                for visitor in self.visitors.iter() {
                    match *visitor {
                        OldVisitor(orig, stopping) => {
                            orig.process_item(it, self);
                            stopping.descend_item(it, self);
                        }
                        NewVisitor(new_visitor) => {
                            let new_visitor = new_visitor;
                            new_visitor.visit_item(it, ());
                        }
                    }
                }
            }
            Crate(c) => {
                for visitor in self.visitors.iter() {
                    match *visitor {
                        OldVisitor(_, stopping) => {
                            stopping.descend_crate(c, self)
                        }
                        NewVisitor(new_visitor) => {
                            let mut new_visitor = new_visitor;
                            visit::walk_crate(&mut new_visitor, c, ())
                        }
                    }
                }
            }
            // Can't use oldvisit::visit_method_helper because the
            // item_stopping_visitor has overridden visit_fn(&fk_method(... ))
            // to be a no-op, so manually invoke visit_fn.
            Method(m) => {
                for visitor in self.visitors.iter() {
                    match *visitor {
                        OldVisitor(orig, stopping) => {
                            let fk = visit::fk_method(m.ident, &m.generics, m);
                            orig.process_fn(&fk, &m.decl, &m.body, m.span, m.id, self);
                            stopping.descend_fn(&fk, &m.decl, &m.body, m.span, m.id, self);
                        }
                        NewVisitor(new_visitor) => {
                            let fk = visit::fk_method(m.ident,
                                                      &m.generics,
                                                      m);
                            let new_visitor = new_visitor;
                            new_visitor.visit_fn(&fk,
                                                 &m.decl,
                                                 &m.body,
                                                 m.span,
                                                 m.id,
                                                 ())
                        }
                    }
                }
            }
        }
    }
}

pub fn each_lint(sess: session::Session,
                 attrs: &[ast::Attribute],
                 f: &fn(@ast::MetaItem, level, @str) -> bool) -> bool {
    let xs = [allow, warn, deny, forbid];
    for &level in xs.iter() {
        let level_name = level_to_str(level);
        for attr in attrs.iter().filter(|m| level_name == m.name()) {
            let meta = attr.node.value;
            let metas = match meta.node {
                ast::MetaList(_, ref metas) => metas,
                _ => {
                    sess.span_err(meta.span, "malformed lint attribute");
                    loop;
                }
            };
            for meta in metas.iter() {
                match meta.node {
                    ast::MetaWord(lintname) => {
                        if !f(*meta, level, lintname) {
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

trait SubitemStoppableVisitor : Visitor<@mut Context> {
    fn is_running_on_items(&mut self) -> bool;

    fn visit_item_action(&mut self, _i:@ast::item, _e:@mut Context) {
        // fill in with particular action without recursion if desired
    }

    fn visit_fn_action(&mut self, _fk:&visit::fn_kind, _fd:&ast::fn_decl,
                       _b:&ast::Block, _s:span, _n:ast::NodeId, _e:@mut Context) {
        // fill in with particular action without recursion if desired
    }

    // The two OVERRIDE methods:
    //
    //   OVERRIDE_visit_item
    //   OVERRIDE_visit_fn
    //
    // *must* be included as initial reimplementations of the standard
    // default behavior of visit_item and visit_fn for every impl of
    // Visitor, in order to recreate the effect of having two variant
    // Outer/Inner behaviors of lint visitors.  (See earlier versions
    // of this module to see what the original encoding was of this
    // emulated behavior.)

    fn OVERRIDE_visit_item(&mut self, i:@ast::item, e:@mut Context) {
        if self.is_running_on_items() {
            self.visit_item_action(i, e);
            visit::walk_item(self, i, e);
        }
    }

    fn OVERRIDE_visit_fn(&mut self, fk:&visit::fn_kind, fd:&ast::fn_decl,
                b:&ast::Block, s:span, n:ast::NodeId, e:@mut Context) {
        if self.is_running_on_items() {
            self.visit_fn_action(fk, fd, b, s, n, e);
            visit::walk_fn(self, fk, fd, b, s, n, e);
        } else {
            match *fk {
                visit::fk_method(*) => {}
                _ => {
                    self.visit_fn_action(fk, fd, b, s, n, e);
                    visit::walk_fn(self, fk, fd, b, s, n, e);
                }
            }
        }
    }
}

struct WhileTrueLintVisitor { stopping_on_items: bool }


impl SubitemStoppableVisitor for WhileTrueLintVisitor {
    fn is_running_on_items(&mut self) -> bool { !self.stopping_on_items }
}

impl Visitor<@mut Context> for WhileTrueLintVisitor {
    fn visit_item(&mut self, i:@ast::item, e:@mut Context) {
        self.OVERRIDE_visit_item(i, e);
    }

    fn visit_fn(&mut self, fk:&visit::fn_kind, fd:&ast::fn_decl,
                b:&ast::Block, s:span, n:ast::NodeId, e:@mut Context) {
        self.OVERRIDE_visit_fn(fk, fd, b, s, n, e);
    }

    fn visit_expr(&mut self, e:@ast::expr, cx:@mut Context) {

            match e.node {
                ast::expr_while(cond, _) => {
                    match cond.node {
                        ast::expr_lit(@codemap::spanned {
                            node: ast::lit_bool(true), _}) =>
                        {
                            cx.span_lint(while_true, e.span,
                                         "denote infinite loops with \
                                          loop { ... }");
                        }
                        _ => ()
                    }
                }
                _ => ()
            }
            visit::walk_expr(self, e, cx);
    }
}

macro_rules! outer_lint_boilerplate_impl(
    ($Visitor:ident) =>
    (
        impl OuterLint for $Visitor {
            fn process_item(@mut self, i:@ast::item, e:@mut Context) {
                self.visit_item_action(i, e);
            }
            fn process_fn(@mut self, fk:&visit::fn_kind, fd:&ast::fn_decl,
                          b:&ast::Block, s:span, n:ast::NodeId, e:@mut Context) {
                self.visit_fn_action(fk, fd, b, s, n, e);
            }
            fn inner_variant(@mut self) -> @mut InnerLint {
                @mut $Visitor { stopping_on_items: true } as @mut InnerLint
            }
        }
    ))

outer_lint_boilerplate_impl!(WhileTrueLintVisitor)

fn lint_while_true() -> @mut OuterLint {
    @mut WhileTrueLintVisitor{ stopping_on_items: false } as @mut OuterLint
}

struct TypeLimitsLintVisitor { stopping_on_items: bool }

impl SubitemStoppableVisitor for TypeLimitsLintVisitor {
    fn is_running_on_items(&mut self) -> bool { !self.stopping_on_items }
}

impl TypeLimitsLintVisitor {
    fn is_valid<T:cmp::Ord>(&mut self, binop: ast::binop, v: T,
            min: T, max: T) -> bool {
        match binop {
            ast::lt => v <= max,
            ast::le => v < max,
            ast::gt => v >= min,
            ast::ge => v > min,
            ast::eq | ast::ne => v >= min && v <= max,
            _ => fail!()
        }
    }

    fn rev_binop(&mut self, binop: ast::binop) -> ast::binop {
        match binop {
            ast::lt => ast::gt,
            ast::le => ast::ge,
            ast::gt => ast::lt,
            ast::ge => ast::le,
            _ => binop
        }
    }

    // for int & uint, be conservative with the warnings, so that the
    // warnings are consistent between 32- and 64-bit platforms
    fn int_ty_range(&mut self, int_ty: ast::int_ty) -> (i64, i64) {
        match int_ty {
            ast::ty_i =>    (i64::min_value,        i64::max_value),
            ast::ty_char => (u32::min_value as i64, u32::max_value as i64),
            ast::ty_i8 =>   (i8::min_value  as i64, i8::max_value  as i64),
            ast::ty_i16 =>  (i16::min_value as i64, i16::max_value as i64),
            ast::ty_i32 =>  (i32::min_value as i64, i32::max_value as i64),
            ast::ty_i64 =>  (i64::min_value,        i64::max_value)
        }
    }

    fn uint_ty_range(&mut self, uint_ty: ast::uint_ty) -> (u64, u64) {
        match uint_ty {
            ast::ty_u =>   (u64::min_value,         u64::max_value),
            ast::ty_u8 =>  (u8::min_value   as u64, u8::max_value   as u64),
            ast::ty_u16 => (u16::min_value  as u64, u16::max_value  as u64),
            ast::ty_u32 => (u32::min_value  as u64, u32::max_value  as u64),
            ast::ty_u64 => (u64::min_value,         u64::max_value)
        }
    }

    fn check_limits(&mut self,
                    cx: &Context,
                    binop: ast::binop,
                    l: @ast::expr,
                    r: @ast::expr)
                    -> bool {
        let (lit, expr, swap) = match (&l.node, &r.node) {
            (&ast::expr_lit(_), _) => (l, r, true),
            (_, &ast::expr_lit(_)) => (r, l, false),
            _ => return true
        };
        // Normalize the binop so that the literal is always on the RHS in
        // the comparison
        let norm_binop = if swap {
            self.rev_binop(binop)
        } else {
            binop
        };
        match ty::get(ty::expr_ty(cx.tcx, expr)).sty {
            ty::ty_int(int_ty) => {
                let (min, max) = self.int_ty_range(int_ty);
                let lit_val: i64 = match lit.node {
                    ast::expr_lit(@li) => match li.node {
                        ast::lit_int(v, _) => v,
                        ast::lit_uint(v, _) => v as i64,
                        ast::lit_int_unsuffixed(v) => v,
                        _ => return true
                    },
                    _ => fail!()
                };
                self.is_valid(norm_binop, lit_val, min, max)
            }
            ty::ty_uint(uint_ty) => {
                let (min, max): (u64, u64) = self.uint_ty_range(uint_ty);
                let lit_val: u64 = match lit.node {
                    ast::expr_lit(@li) => match li.node {
                        ast::lit_int(v, _) => v as u64,
                        ast::lit_uint(v, _) => v,
                        ast::lit_int_unsuffixed(v) => v as u64,
                        _ => return true
                    },
                    _ => fail!()
                };
                self.is_valid(norm_binop, lit_val, min, max)
            }
            _ => true
        }
    }

    fn is_comparison(&mut self, binop: ast::binop) -> bool {
        match binop {
            ast::eq | ast::lt | ast::le |
            ast::ne | ast::ge | ast::gt => true,
            _ => false
        }
    }
}

impl Visitor<@mut Context> for TypeLimitsLintVisitor {
    fn visit_item(&mut self, i:@ast::item, e:@mut Context) {
        self.OVERRIDE_visit_item(i, e);
    }

    fn visit_fn(&mut self, fk:&visit::fn_kind, fd:&ast::fn_decl,
                b:&ast::Block, s:span, n:ast::NodeId, e:@mut Context) {
        self.OVERRIDE_visit_fn(fk, fd, b, s, n, e);
    }

    fn visit_expr(&mut self, e:@ast::expr, cx:@mut Context) {

            match e.node {
                ast::expr_binary(_, ref binop, l, r) => {
                    if self.is_comparison(*binop)
                        && !self.check_limits(cx, *binop, l, r) {
                        cx.span_lint(type_limits, e.span,
                                     "comparison is useless due to type limits");
                    }
                }
                _ => ()
            }
            visit::walk_expr(self, e, cx);
    }
}

outer_lint_boilerplate_impl!(TypeLimitsLintVisitor)

fn lint_type_limits() -> @mut OuterLint {
    @mut TypeLimitsLintVisitor{ stopping_on_items: false } as @mut OuterLint
}

fn check_item_ctypes(cx: &Context, it: &ast::item) {
    fn check_ty(cx: &Context, ty: &ast::Ty) {
        match ty.node {
            ast::ty_path(_, _, id) => {
                match cx.tcx.def_map.get_copy(&id) {
                    ast::def_prim_ty(ast::ty_int(ast::ty_i)) => {
                        cx.span_lint(ctypes, ty.span,
                                "found rust type `int` in foreign module, while \
                                libc::c_int or libc::c_long should be used");
                    }
                    ast::def_prim_ty(ast::ty_uint(ast::ty_u)) => {
                        cx.span_lint(ctypes, ty.span,
                                "found rust type `uint` in foreign module, while \
                                libc::c_uint or libc::c_ulong should be used");
                    }
                    _ => ()
                }
            }
            ast::ty_ptr(ref mt) => { check_ty(cx, mt.ty) }
            _ => ()
        }
    }

    fn check_foreign_fn(cx: &Context, decl: &ast::fn_decl) {
        for input in decl.inputs.iter() {
            check_ty(cx, &input.ty);
        }
        check_ty(cx, &decl.output)
    }

    match it.node {
      ast::item_foreign_mod(ref nmod) if !nmod.abis.is_intrinsic() => {
        for ni in nmod.items.iter() {
            match ni.node {
                ast::foreign_item_fn(ref decl, _) => {
                    check_foreign_fn(cx, decl);
                }
                ast::foreign_item_static(ref t, _) => { check_ty(cx, t); }
            }
        }
      }
      _ => {/* nothing to do */ }
    }
}

fn check_type_for_lint(cx: &Context, lint: lint, span: span, ty: ty::t) {
    if cx.get_level(lint) == allow { return }

    let mut n_box = 0;
    let mut n_uniq = 0;
    ty::fold_ty(cx.tcx, ty, |t| {
        match ty::get(t).sty {
          ty::ty_box(_) => n_box += 1,
          ty::ty_uniq(_) => n_uniq += 1,
          _ => ()
        };
        t
    });

    if n_uniq > 0 && lint != managed_heap_memory {
        let s = ty_to_str(cx.tcx, ty);
        let m = ~"type uses owned (~ type) pointers: " + s;
        cx.span_lint(lint, span, m);
    }

    if n_box > 0 && lint != owned_heap_memory {
        let s = ty_to_str(cx.tcx, ty);
        let m = ~"type uses managed (@ type) pointers: " + s;
        cx.span_lint(lint, span, m);
    }
}

fn check_type(cx: &Context, span: span, ty: ty::t) {
    let xs = [managed_heap_memory, owned_heap_memory, heap_memory];
    for lint in xs.iter() {
        check_type_for_lint(cx, *lint, span, ty);
    }
}

fn check_item_heap(cx: &Context, it: &ast::item) {
    match it.node {
      ast::item_fn(*) |
      ast::item_ty(*) |
      ast::item_enum(*) |
      ast::item_struct(*) => check_type(cx, it.span,
                                        ty::node_id_to_type(cx.tcx,
                                                            it.id)),
      _ => ()
    }

    // If it's a struct, we also have to check the fields' types
    match it.node {
        ast::item_struct(struct_def, _) => {
            for struct_field in struct_def.fields.iter() {
                check_type(cx, struct_field.span,
                           ty::node_id_to_type(cx.tcx,
                                               struct_field.node.id));
            }
        }
        _ => ()
    }
}

struct HeapLintVisitor { stopping_on_items: bool }

impl SubitemStoppableVisitor for HeapLintVisitor {
    fn is_running_on_items(&mut self) -> bool { !self.stopping_on_items }
}

impl Visitor<@mut Context> for HeapLintVisitor {
    fn visit_item(&mut self, i:@ast::item, e:@mut Context) {
        self.OVERRIDE_visit_item(i, e);
    }

    fn visit_fn(&mut self, fk:&visit::fn_kind, fd:&ast::fn_decl,
                b:&ast::Block, s:span, n:ast::NodeId, e:@mut Context) {
        self.OVERRIDE_visit_fn(fk, fd, b, s, n, e);
    }

    fn visit_expr(&mut self, e:@ast::expr, cx:@mut Context) {
            let ty = ty::expr_ty(cx.tcx, e);
            check_type(cx, e.span, ty);
            visit::walk_expr(self, e, cx);
    }
}

outer_lint_boilerplate_impl!(HeapLintVisitor)

fn lint_heap() -> @mut OuterLint {
    @mut HeapLintVisitor { stopping_on_items: false } as @mut OuterLint
}

struct PathStatementLintVisitor {
    stopping_on_items: bool
}

impl SubitemStoppableVisitor for PathStatementLintVisitor {
    fn is_running_on_items(&mut self) -> bool { !self.stopping_on_items }
}

impl Visitor<@mut Context> for PathStatementLintVisitor {
    fn visit_item(&mut self, i:@ast::item, e:@mut Context) {
        self.OVERRIDE_visit_item(i, e);
    }

    fn visit_fn(&mut self, fk:&visit::fn_kind, fd:&ast::fn_decl,
                b:&ast::Block, s:span, n:ast::NodeId, e:@mut Context) {
        self.OVERRIDE_visit_fn(fk, fd, b, s, n, e);
    }

    fn visit_stmt(&mut self, s:@ast::stmt, cx:@mut Context) {
            match s.node {
                ast::stmt_semi(
                    @ast::expr { node: ast::expr_path(_), _ },
                    _
                ) => {
                    cx.span_lint(path_statement, s.span,
                                 "path statement with no effect");
                }
                _ => ()
            }
            visit::walk_stmt(self, s, cx);

    }
}

outer_lint_boilerplate_impl!(PathStatementLintVisitor)

fn lint_path_statement() -> @mut OuterLint {
    @mut PathStatementLintVisitor{ stopping_on_items: false } as @mut OuterLint
}

fn check_item_non_camel_case_types(cx: &Context, it: &ast::item) {
    fn is_camel_case(cx: ty::ctxt, ident: ast::ident) -> bool {
        let ident = cx.sess.str_of(ident);
        assert!(!ident.is_empty());
        let ident = ident.trim_chars(&'_');

        // start with a non-lowercase letter rather than non-uppercase
        // ones (some scripts don't have a concept of upper/lowercase)
        !ident.char_at(0).is_lowercase() &&
            !ident.contains_char('_')
    }

    fn check_case(cx: &Context, sort: &str, ident: ast::ident, span: span) {
        if !is_camel_case(cx.tcx, ident) {
            cx.span_lint(
                non_camel_case_types, span,
                fmt!("%s `%s` should have a camel case identifier",
                    sort, cx.tcx.sess.str_of(ident)));
        }
    }

    match it.node {
        ast::item_ty(*) | ast::item_struct(*) => {
            check_case(cx, "type", it.ident, it.span)
        }
        ast::item_trait(*) => {
            check_case(cx, "trait", it.ident, it.span)
        }
        ast::item_enum(ref enum_definition, _) => {
            check_case(cx, "type", it.ident, it.span);
            for variant in enum_definition.variants.iter() {
                check_case(cx, "variant", variant.node.name, variant.span);
            }
        }
        _ => ()
    }
}

fn check_item_non_uppercase_statics(cx: &Context, it: &ast::item) {
    match it.node {
        // only check static constants
        ast::item_static(_, ast::m_imm, _) => {
            let s = cx.tcx.sess.str_of(it.ident);
            // check for lowercase letters rather than non-uppercase
            // ones (some scripts don't have a concept of
            // upper/lowercase)
            if s.iter().any(|c| c.is_lowercase()) {
                cx.span_lint(non_uppercase_statics, it.span,
                             "static constant should have an uppercase identifier");
            }
        }
        _ => {}
    }
}

struct UnusedUnsafeLintVisitor { stopping_on_items: bool }

impl SubitemStoppableVisitor for UnusedUnsafeLintVisitor {
    fn is_running_on_items(&mut self) -> bool { !self.stopping_on_items }
}

impl Visitor<@mut Context> for UnusedUnsafeLintVisitor {
    fn visit_item(&mut self, i:@ast::item, e:@mut Context) {
        self.OVERRIDE_visit_item(i, e);
    }

    fn visit_fn(&mut self, fk:&visit::fn_kind, fd:&ast::fn_decl,
                b:&ast::Block, s:span, n:ast::NodeId, e:@mut Context) {
        self.OVERRIDE_visit_fn(fk, fd, b, s, n, e);
    }

    fn visit_expr(&mut self, e:@ast::expr, cx:@mut Context) {

            match e.node {
                ast::expr_block(ref blk) if blk.rules == ast::UnsafeBlock => {
                    if !cx.tcx.used_unsafe.contains(&blk.id) {
                        cx.span_lint(unused_unsafe, blk.span,
                                     "unnecessary `unsafe` block");
                    }
                }
                _ => ()
            }
            visit::walk_expr(self, e, cx);
    }
}

outer_lint_boilerplate_impl!(UnusedUnsafeLintVisitor)

fn lint_unused_unsafe() -> @mut OuterLint {
    @mut UnusedUnsafeLintVisitor{ stopping_on_items: false } as @mut OuterLint
}

struct UnusedMutLintVisitor { stopping_on_items: bool }

impl UnusedMutLintVisitor {
    fn check_pat(&mut self, cx: &Context, p: @ast::pat) {
        let mut used = false;
        let mut bindings = 0;
        do pat_util::pat_bindings(cx.tcx.def_map, p) |_, id, _, _| {
            used = used || cx.tcx.used_mut_nodes.contains(&id);
            bindings += 1;
        }
        if !used {
            let msg = if bindings == 1 {
                "variable does not need to be mutable"
            } else {
                "variables do not need to be mutable"
            };
            cx.span_lint(unused_mut, p.span, msg);
        }
    }

    fn visit_fn_decl(&mut self, cx: &Context, fd: &ast::fn_decl) {
        for arg in fd.inputs.iter() {
            if arg.is_mutbl {
                self.check_pat(cx, arg.pat);
            }
        }
    }
}

impl SubitemStoppableVisitor for UnusedMutLintVisitor {
    fn is_running_on_items(&mut self) -> bool { !self.stopping_on_items }

    fn visit_fn_action(&mut self, _a:&visit::fn_kind, fd:&ast::fn_decl,
                       _b:&ast::Block, _c:span, _d:ast::NodeId, cx:@mut Context) {
            self.visit_fn_decl(cx, fd);
    }
}

impl Visitor<@mut Context> for UnusedMutLintVisitor {
    fn visit_item(&mut self, i:@ast::item, e:@mut Context) {
        self.OVERRIDE_visit_item(i, e);
    }

    fn visit_fn(&mut self, fk:&visit::fn_kind, fd:&ast::fn_decl,
                b:&ast::Block, s:span, n:ast::NodeId, e:@mut Context) {
        self.OVERRIDE_visit_fn(fk, fd, b, s, n, e);
    }


    fn visit_local(&mut self, l:@ast::Local, cx:@mut Context) {
            if l.is_mutbl {
                self.check_pat(cx, l.pat);
            }
            visit::walk_local(self, l, cx);
    }

    fn visit_ty_method(&mut self, tm:&ast::TypeMethod, cx:@mut Context) {
            self.visit_fn_decl(cx, &tm.decl);
            visit::walk_ty_method(self, tm, cx);
    }

    fn visit_trait_method(&mut self, tm:&ast::trait_method, cx:@mut Context) {
            match *tm {
                ast::required(ref tm) => self.visit_fn_decl(cx, &tm.decl),
                ast::provided(m) => self.visit_fn_decl(cx, &m.decl)
            }
            visit::walk_trait_method(self, tm, cx);
    }
}

outer_lint_boilerplate_impl!(UnusedMutLintVisitor)

fn lint_unused_mut() -> @mut OuterLint {
    @mut UnusedMutLintVisitor{ stopping_on_items: false } as @mut OuterLint
}

fn lint_session(cx: @mut Context) -> @mut visit::Visitor<()> {
    ast_util::id_visitor(|id| {
        match cx.tcx.sess.lints.pop(&id) {
            None => {},
            Some(l) => {
                for (lint, span, msg) in l.move_iter() {
                    cx.span_lint(lint, span, msg)
                }
            }
        }
    }, false)
}

struct UnnecessaryAllocationLintVisitor { stopping_on_items: bool }

impl SubitemStoppableVisitor for UnnecessaryAllocationLintVisitor {
    fn is_running_on_items(&mut self) -> bool { !self.stopping_on_items }
}

impl Visitor<@mut Context> for UnnecessaryAllocationLintVisitor {
    fn visit_item(&mut self, i:@ast::item, e:@mut Context) {
        self.OVERRIDE_visit_item(i, e);
    }

    fn visit_fn(&mut self, fk:&visit::fn_kind, fd:&ast::fn_decl,
                b:&ast::Block, s:span, n:ast::NodeId, e:@mut Context) {
        self.OVERRIDE_visit_fn(fk, fd, b, s, n, e);
    }

    fn visit_expr(&mut self, e:@ast::expr, cx:@mut Context) {
            self.check(cx, e);
            visit::walk_expr(self, e, cx);
    }
}

impl UnnecessaryAllocationLintVisitor {
    // Warn if string and vector literals with sigils are immediately borrowed.
    // Those can have the sigil removed.
    fn check(&mut self, cx: &Context, e: &ast::expr) {
        match e.node {
            ast::expr_vstore(e2, ast::expr_vstore_uniq) |
            ast::expr_vstore(e2, ast::expr_vstore_box) => {
                match e2.node {
                    ast::expr_lit(@codemap::spanned{
                            node: ast::lit_str(*), _}) |
                    ast::expr_vec(*) => {}
                    _ => return
                }
            }

            _ => return
        }

        match cx.tcx.adjustments.find_copy(&e.id) {
            Some(@ty::AutoDerefRef(ty::AutoDerefRef {
                autoref: Some(ty::AutoBorrowVec(*)), _ })) => {
                cx.span_lint(unnecessary_allocation,
                             e.span, "unnecessary allocation, the sigil can be \
                                      removed");
            }

            _ => ()
        }
    }
}

outer_lint_boilerplate_impl!(UnnecessaryAllocationLintVisitor)

fn lint_unnecessary_allocations() -> @mut OuterLint {
    @mut UnnecessaryAllocationLintVisitor{ stopping_on_items: false } as @mut OuterLint
}

struct MissingDocLintVisitor { stopping_on_items: bool }

impl MissingDocLintVisitor {
    fn check_attrs(&mut self,
                   cx: @mut Context,
                   attrs: &[ast::Attribute],
                   sp: span,
                   msg: &str) {
        // If we're building a test harness, then warning about documentation is
        // probably not really relevant right now
        if cx.tcx.sess.opts.test { return }
        // If we have doc(hidden), nothing to do
        if cx.doc_hidden { return }
        // If we're documented, nothing to do
        if attrs.iter().any(|a| a.node.is_sugared_doc) { return }

        // otherwise, warn!
        cx.span_lint(missing_doc, sp, msg);
    }
}

impl Visitor<@mut Context> for MissingDocLintVisitor {
    fn visit_item(&mut self, i:@ast::item, e:@mut Context) {
        self.OVERRIDE_visit_item(i, e);
    }

    fn visit_fn(&mut self, fk:&visit::fn_kind, fd:&ast::fn_decl,
                b:&ast::Block, s:span, n:ast::NodeId, e:@mut Context) {
        self.OVERRIDE_visit_fn(fk, fd, b, s, n, e);
    }

    fn visit_ty_method(&mut self, m:&ast::TypeMethod, cx:@mut Context) {
            // All ty_method objects are linted about because they're part of a
            // trait (no visibility)
            self.check_attrs(cx, m.attrs, m.span,
                        "missing documentation for a method");
            visit::walk_ty_method(self, m, cx);
    }
}

impl SubitemStoppableVisitor for MissingDocLintVisitor {
    fn is_running_on_items(&mut self) -> bool { !self.stopping_on_items }

    fn visit_fn_action(&mut self, fk:&visit::fn_kind, _d:&ast::fn_decl,
                       _b:&ast::Block, sp:span, _id:ast::NodeId, cx:@mut Context) {

            // Only warn about explicitly public methods. Soon implicit
            // public-ness will hopefully be going away.
            match *fk {
                visit::fk_method(_, _, m) if m.vis == ast::public => {
                    // If we're in a trait implementation, no need to duplicate
                    // documentation
                    if !cx.in_trait_impl {
                        self.check_attrs(cx, m.attrs, sp,
                                    "missing documentation for a method");
                    }
                }

                _ => {}
            }
    }

    fn visit_item_action(&mut self, it:@ast::item, cx:@mut Context) {

            match it.node {
                // Go ahead and match the fields here instead of using
                // visit_struct_field while we have access to the enclosing
                // struct's visibility
                ast::item_struct(sdef, _) if it.vis == ast::public => {
                    self.check_attrs(cx, it.attrs, it.span,
                                "missing documentation for a struct");
                    for field in sdef.fields.iter() {
                        match field.node.kind {
                            ast::named_field(_, vis) if vis != ast::private => {
                                self.check_attrs(cx, field.node.attrs, field.span,
                                            "missing documentation for a field");
                            }
                            ast::unnamed_field | ast::named_field(*) => {}
                        }
                    }
                }

                ast::item_trait(*) if it.vis == ast::public => {
                    self.check_attrs(cx, it.attrs, it.span,
                                "missing documentation for a trait");
                }

                ast::item_fn(*) if it.vis == ast::public => {
                    self.check_attrs(cx, it.attrs, it.span,
                                "missing documentation for a function");
                }

                _ => {}
            }
    }
}

outer_lint_boilerplate_impl!(MissingDocLintVisitor)

fn lint_missing_doc() -> @mut OuterLint {
    @mut MissingDocLintVisitor { stopping_on_items: false } as @mut OuterLint
}

struct LintCheckVisitor;

impl Visitor<@mut Context> for LintCheckVisitor {

    fn visit_item(&mut self, it:@ast::item, cx: @mut Context) {

                do cx.with_lint_attrs(it.attrs) {
                    match it.node {
                        ast::item_impl(_, Some(*), _, _) => {
                            cx.in_trait_impl = true;
                        }
                        _ => {}
                    }
                    check_item_ctypes(cx, it);
                    check_item_non_camel_case_types(cx, it);
                    check_item_non_uppercase_statics(cx, it);
                    check_item_heap(cx, it);

                    cx.process(Item(it));
                    visit::walk_item(self, it, cx);
                    cx.in_trait_impl = false;
                }
    }

    fn visit_fn(&mut self, fk:&visit::fn_kind, decl:&ast::fn_decl,
                body:&ast::Block, span:span, id:ast::NodeId, cx:@mut Context) {

                match *fk {
                    visit::fk_method(_, _, m) => {
                        do cx.with_lint_attrs(m.attrs) {
                            cx.process(Method(m));
                            visit::walk_fn(self,
                                               fk,
                                               decl,
                                               body,
                                               span,
                                               id,
                                               cx);
                        }
                    }
                    _ => {
                        visit::walk_fn(self,
                                           fk,
                                           decl,
                                           body,
                                           span,
                                           id,
                                           cx);
                    }
                }
    }
}

pub fn check_crate(tcx: ty::ctxt, crate: @ast::Crate) {
    let cx = @mut Context {
        dict: @get_lint_dict(),
        curr: SmallIntMap::new(),
        tcx: tcx,
        lint_stack: ~[],
        visitors: ~[],
        in_trait_impl: false,
        doc_hidden: false,
    };

    // Install defaults.
    for (_, spec) in cx.dict.iter() {
        cx.set_level(spec.lint, spec.default, Default);
    }

    // Install command-line options, overriding defaults.
    for &(lint, level) in tcx.sess.opts.lint_opts.iter() {
        cx.set_level(lint, level, CommandLine);
    }

    // Register each of the lint passes with the context
    cx.add_oldvisit_lint(lint_while_true());
    cx.add_oldvisit_lint(lint_path_statement());
    cx.add_oldvisit_lint(lint_heap());
    cx.add_oldvisit_lint(lint_type_limits());
    cx.add_oldvisit_lint(lint_unused_unsafe());
    cx.add_oldvisit_lint(lint_unused_mut());
    cx.add_oldvisit_lint(lint_unnecessary_allocations());
    cx.add_oldvisit_lint(lint_missing_doc());
    cx.add_lint(lint_session(cx));

    // Actually perform the lint checks (iterating the ast)
    do cx.with_lint_attrs(crate.attrs) {
        cx.process(Crate(crate));

        let mut visitor = LintCheckVisitor;

        visit::walk_crate(&mut visitor, crate, cx);
    }

    // If we missed any lints added to the session, then there's a bug somewhere
    // in the iteration code.
    for (id, v) in tcx.sess.lints.iter() {
        for t in v.iter() {
            match *t {
                (lint, span, ref msg) =>
                    tcx.sess.span_bug(span, fmt!("unprocessed lint %? at %s: \
                                                  %s",
                                                 lint,
                                                 ast_map::node_id_to_str(
                                                 tcx.items,
                                                 *id,
                                                 token::get_ident_interner()),
                                                 *msg))
            }
        }
    }

    tcx.sess.abort_if_errors();
}
