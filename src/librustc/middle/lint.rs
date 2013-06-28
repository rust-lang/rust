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
use syntax::attr;
use syntax::codemap::span;
use syntax::codemap;
use syntax::{ast, visit, ast_util};

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
 * passes are also applied.  Each lint pass is a visit::vt<()> structure. These
 * visitors are constructed via the lint_*() functions below. There are also
 * some lint checks which operate directly on ast nodes (such as @ast::item),
 * and those are organized as check_item_*(). Each visitor added to the lint
 * context is modified to stop once it reaches a node which could alter the lint
 * levels. This means that everything is looked at once and only once by every
 * lint pass.
 *
 * With this all in place, to add a new lint warning, all you need to do is to
 * either invoke `add_lint` on the session at the appropriate time, or write a
 * lint pass in this module which is just an ast visitor. The context used when
 * traversing the ast has a `span_lint` method which only needs the span of the
 * item that's being warned about.
 */

#[deriving(Eq)]
pub enum lint {
    ctypes,
    unused_imports,
    unnecessary_qualification,
    while_true,
    path_statement,
    implicit_copies,
    unrecognized_lint,
    non_implicitly_copyable_typarams,
    deprecated_pattern,
    non_camel_case_types,
    non_uppercase_statics,
    type_limits,
    default_methods,
    unused_unsafe,
    copy_implicitly_copyable,

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

#[deriving(Eq, Ord)]
pub enum level {
    allow, warn, deny, forbid
}

#[deriving(Eq)]
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
    Crate(@ast::crate),
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

    ("non_implicitly_copyable_typarams",
     LintSpec {
        lint: non_implicitly_copyable_typarams,
        desc: "passing non implicitly copyable types as copy type params",
        default: warn
     }),

    ("implicit_copies",
     LintSpec {
        lint: implicit_copies,
        desc: "implicit copies of non implicitly copyable data",
        default: warn
     }),

    ("deprecated_pattern",
     LintSpec {
        lint: deprecated_pattern,
        desc: "warn about deprecated uses of pattern bindings",
        default: allow
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

    ("default_methods",
     LintSpec {
        lint: default_methods,
        desc: "allow default methods",
        default: allow
     }),

    ("unused_unsafe",
     LintSpec {
        lint: unused_unsafe,
        desc: "unnecessary use of an `unsafe` block",
        default: warn
    }),

    ("copy_implicitly_copyable",
     LintSpec {
        lint: copy_implicitly_copyable,
        desc: "detect unnecessary uses of `copy` on implicitly copyable \
               values",
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
    for lint_table.iter().advance |&(k, v)| {
        map.insert(k, v);
    }
    return map;
}

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
    //
    // This is a pair so every visitor can visit every node. When a lint pass is
    // registered, another visitor is created which stops at all items which can
    // alter the attributes of the ast. This "item stopping visitor" is the
    // second element of the pair, while the original visitor is the first
    // element. This means that when visiting a node, the original recursive
    // call can used the original visitor's method, although the recursing
    // visitor supplied to the method is the item stopping visitor.
    visitors: ~[(visit::vt<@mut Context>, visit::vt<@mut Context>)],
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
        for self.dict.iter().advance |(k, v)| {
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

        for note.iter().advance |&span| {
            self.tcx.sess.span_note(span, "lint level defined here");
        }
    }

    /**
     * Merge the lints specified by any lint attributes into the
     * current lint context, call the provided function, then reset the
     * lints in effect to their previous state.
     */
    fn with_lint_attrs(@mut self, attrs: &[ast::attribute], f: &fn()) {
        // Parse all of the lint attributes, and then add them all to the
        // current dictionary of lint information. Along the way, keep a history
        // of what we changed so we can roll everything back after invoking the
        // specified closure
        let mut pushed = 0u;
        for each_lint(self.tcx.sess, attrs) |meta, level, lintname| {
            let lint = match self.dict.find_equiv(&lintname) {
              None => {
                self.span_lint(
                    unrecognized_lint,
                    meta.span,
                    fmt!("unknown `%s` attribute: `%s`",
                         level_to_str(level), lintname));
                loop
              }
              Some(lint) => { lint.lint }
            };

            let now = self.get_level(lint);
            if now == forbid && level != forbid {
                self.tcx.sess.span_err(meta.span,
                    fmt!("%s(%s) overruled by outer forbid(%s)",
                         level_to_str(level),
                         lintname, lintname));
                loop;
            }

            if now != level {
                let src = self.get_source(lint);
                self.lint_stack.push((lint, now, src));
                pushed += 1;
                self.set_level(lint, level, Node(meta.span));
            }
        }

        // detect doc(hidden)
        let mut doc_hidden = false;
        let r = attr::find_attrs_by_name(attrs, "doc");
        for r.iter().advance |attr| {
            match attr::get_meta_item_list(attr.node.value) {
                Some(s) => {
                    if attr::find_meta_items_by_name(s, "hidden").len() > 0 {
                        doc_hidden = true;
                    }
                }
                None => {}
            }
        }
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
        for pushed.times {
            let (lint, lvl, src) = self.lint_stack.pop();
            self.set_level(lint, lvl, src);
        }
    }

    fn add_lint(&mut self, v: visit::vt<@mut Context>) {
        self.visitors.push((v, item_stopping_visitor(v)));
    }

    fn process(@mut self, n: AttributedNode) {
        // see comment of the `visitors` field in the struct for why there's a
        // pair instead of just one visitor.
        match n {
            Item(it) => {
                for self.visitors.iter().advance |&(orig, stopping)| {
                    (orig.visit_item)(it, (self, stopping));
                }
            }
            Crate(c) => {
                for self.visitors.iter().advance |&(_, stopping)| {
                    visit::visit_crate(c, (self, stopping));
                }
            }
            // Can't use visit::visit_method_helper because the
            // item_stopping_visitor has overridden visit_fn(&fk_method(... ))
            // to be a no-op, so manually invoke visit_fn.
            Method(m) => {
                let fk = visit::fk_method(m.ident, &m.generics, m);
                for self.visitors.iter().advance |&(orig, stopping)| {
                    (orig.visit_fn)(&fk, &m.decl, &m.body, m.span, m.id,
                                    (self, stopping));
                }
            }
        }
    }
}

pub fn each_lint(sess: session::Session,
                 attrs: &[ast::attribute],
                 f: &fn(@ast::meta_item, level, @str) -> bool) -> bool {
    let xs = [allow, warn, deny, forbid];
    for xs.iter().advance |&level| {
        let level_name = level_to_str(level);
        let attrs = attr::find_attrs_by_name(attrs, level_name);
        for attrs.iter().advance |attr| {
            let meta = attr.node.value;
            let metas = match meta.node {
                ast::meta_list(_, ref metas) => metas,
                _ => {
                    sess.span_err(meta.span, "malformed lint attribute");
                    loop;
                }
            };
            for metas.iter().advance |meta| {
                match meta.node {
                    ast::meta_word(lintname) => {
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

// Take a visitor, and modify it so that it will not proceed past subitems.
// This is used to make the simple visitors used for the lint passes
// not traverse into subitems, since that is handled by the outer
// lint visitor.
fn item_stopping_visitor<E: Copy>(outer: visit::vt<E>) -> visit::vt<E> {
    visit::mk_vt(@visit::Visitor {
        visit_item: |_i, (_e, _v)| { },
        visit_fn: |fk, fd, b, s, id, (e, v)| {
            match *fk {
                visit::fk_method(*) => {}
                _ => (outer.visit_fn)(fk, fd, b, s, id, (e, v))
            }
        },
    .. **outer})
}

fn lint_while_true() -> visit::vt<@mut Context> {
    visit::mk_vt(@visit::Visitor {
        visit_expr: |e, (cx, vt): (@mut Context, visit::vt<@mut Context>)| {
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
            visit::visit_expr(e, (cx, vt));
        },
        .. *visit::default_visitor()
    })
}

fn lint_type_limits() -> visit::vt<@mut Context> {
    fn is_valid<T:cmp::Ord>(binop: ast::binop, v: T,
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

    fn rev_binop(binop: ast::binop) -> ast::binop {
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
    fn int_ty_range(int_ty: ast::int_ty) -> (i64, i64) {
        match int_ty {
            ast::ty_i =>    (i64::min_value,        i64::max_value),
            ast::ty_char => (u32::min_value as i64, u32::max_value as i64),
            ast::ty_i8 =>   (i8::min_value  as i64, i8::max_value  as i64),
            ast::ty_i16 =>  (i16::min_value as i64, i16::max_value as i64),
            ast::ty_i32 =>  (i32::min_value as i64, i32::max_value as i64),
            ast::ty_i64 =>  (i64::min_value,        i64::max_value)
        }
    }

    fn uint_ty_range(uint_ty: ast::uint_ty) -> (u64, u64) {
        match uint_ty {
            ast::ty_u =>   (u64::min_value,         u64::max_value),
            ast::ty_u8 =>  (u8::min_value   as u64, u8::max_value   as u64),
            ast::ty_u16 => (u16::min_value  as u64, u16::max_value  as u64),
            ast::ty_u32 => (u32::min_value  as u64, u32::max_value  as u64),
            ast::ty_u64 => (u64::min_value,         u64::max_value)
        }
    }

    fn check_limits(cx: &Context, binop: ast::binop, l: &ast::expr,
                    r: &ast::expr) -> bool {
        let (lit, expr, swap) = match (&l.node, &r.node) {
            (&ast::expr_lit(_), _) => (l, r, true),
            (_, &ast::expr_lit(_)) => (r, l, false),
            _ => return true
        };
        // Normalize the binop so that the literal is always on the RHS in
        // the comparison
        let norm_binop = if swap {
            rev_binop(binop)
        } else {
            binop
        };
        match ty::get(ty::expr_ty(cx.tcx, @/*bad*/copy *expr)).sty {
            ty::ty_int(int_ty) => {
                let (min, max) = int_ty_range(int_ty);
                let lit_val: i64 = match lit.node {
                    ast::expr_lit(@li) => match li.node {
                        ast::lit_int(v, _) => v,
                        ast::lit_uint(v, _) => v as i64,
                        ast::lit_int_unsuffixed(v) => v,
                        _ => return true
                    },
                    _ => fail!()
                };
                is_valid(norm_binop, lit_val, min, max)
            }
            ty::ty_uint(uint_ty) => {
                let (min, max): (u64, u64) = uint_ty_range(uint_ty);
                let lit_val: u64 = match lit.node {
                    ast::expr_lit(@li) => match li.node {
                        ast::lit_int(v, _) => v as u64,
                        ast::lit_uint(v, _) => v,
                        ast::lit_int_unsuffixed(v) => v as u64,
                        _ => return true
                    },
                    _ => fail!()
                };
                is_valid(norm_binop, lit_val, min, max)
            }
            _ => true
        }
    }

    fn is_comparison(binop: ast::binop) -> bool {
        match binop {
            ast::eq | ast::lt | ast::le |
            ast::ne | ast::ge | ast::gt => true,
            _ => false
        }
    }

    visit::mk_vt(@visit::Visitor {
        visit_expr: |e, (cx, vt): (@mut Context, visit::vt<@mut Context>)| {
            match e.node {
                ast::expr_binary(_, ref binop, @ref l, @ref r) => {
                    if is_comparison(*binop)
                        && !check_limits(cx, *binop, l, r) {
                        cx.span_lint(type_limits, e.span,
                                     "comparison is useless due to type limits");
                    }
                }
                _ => ()
            }
            visit::visit_expr(e, (cx, vt));
        },

        .. *visit::default_visitor()
    })
}

fn check_item_default_methods(cx: &Context, item: &ast::item) {
    match item.node {
        ast::item_trait(_, _, ref methods) => {
            for methods.iter().advance |method| {
                match *method {
                    ast::required(*) => {}
                    ast::provided(*) => {
                        cx.span_lint(default_methods, item.span,
                                     "default methods are experimental");
                    }
                }
            }
        }
        _ => {}
    }
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
        for decl.inputs.iter().advance |in| {
            check_ty(cx, &in.ty);
        }
        check_ty(cx, &decl.output)
    }

    match it.node {
      ast::item_foreign_mod(ref nmod) if !nmod.abis.is_intrinsic() => {
        for nmod.items.iter().advance |ni| {
            match ni.node {
                ast::foreign_item_fn(ref decl, _, _) => {
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
    for xs.iter().advance |lint| {
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
            for struct_def.fields.iter().advance |struct_field| {
                check_type(cx, struct_field.span,
                           ty::node_id_to_type(cx.tcx,
                                               struct_field.node.id));
            }
        }
        _ => ()
    }
}

fn lint_heap() -> visit::vt<@mut Context> {
    visit::mk_vt(@visit::Visitor {
        visit_expr: |e, (cx, vt): (@mut Context, visit::vt<@mut Context>)| {
            let ty = ty::expr_ty(cx.tcx, e);
            check_type(cx, e.span, ty);
            visit::visit_expr(e, (cx, vt));
        },
        .. *visit::default_visitor()
    })
}

fn lint_path_statement() -> visit::vt<@mut Context> {
    visit::mk_vt(@visit::Visitor {
        visit_stmt: |s, (cx, vt): (@mut Context, visit::vt<@mut Context>)| {
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
            visit::visit_stmt(s, (cx, vt));
        },
        .. *visit::default_visitor()
    })
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

    fn check_case(cx: &Context, ident: ast::ident, span: span) {
        if !is_camel_case(cx.tcx, ident) {
            cx.span_lint(non_camel_case_types, span,
                         "type, variant, or trait should have \
                          a camel case identifier");
        }
    }

    match it.node {
        ast::item_ty(*) | ast::item_struct(*) |
        ast::item_trait(*) => {
            check_case(cx, it.ident, it.span)
        }
        ast::item_enum(ref enum_definition, _) => {
            check_case(cx, it.ident, it.span);
            for enum_definition.variants.iter().advance |variant| {
                check_case(cx, variant.node.name, variant.span);
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

fn lint_unused_unsafe() -> visit::vt<@mut Context> {
    visit::mk_vt(@visit::Visitor {
        visit_expr: |e, (cx, vt): (@mut Context, visit::vt<@mut Context>)| {
            match e.node {
                ast::expr_block(ref blk) if blk.rules == ast::unsafe_blk => {
                    if !cx.tcx.used_unsafe.contains(&blk.id) {
                        cx.span_lint(unused_unsafe, blk.span,
                                     "unnecessary `unsafe` block");
                    }
                }
                _ => ()
            }
            visit::visit_expr(e, (cx, vt));
        },
        .. *visit::default_visitor()
    })
}

fn lint_copy_implicitly_copyable() -> visit::vt<@mut Context> {
    visit::mk_vt(@visit::Visitor {
        visit_expr: |e, (cx, vt): (@mut Context, visit::vt<@mut Context>)| {
            match e.node {
                ast::expr_copy(subexpr) => {
                    let ty = ty::expr_ty(cx.tcx, subexpr);
                    if !ty::type_moves_by_default(cx.tcx, ty) {
                        cx.span_lint(copy_implicitly_copyable,
                                     e.span,
                                     "unnecessary `copy`; this value is implicitly copyable");
                    }
                }
                _ => ()
            }
            visit::visit_expr(e, (cx, vt));
        },
        .. *visit::default_visitor()
    })
}

fn lint_unused_mut() -> visit::vt<@mut Context> {
    fn check_pat(cx: &Context, p: @ast::pat) {
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

    fn visit_fn_decl(cx: &Context, fd: &ast::fn_decl) {
        for fd.inputs.iter().advance |arg| {
            if arg.is_mutbl {
                check_pat(cx, arg.pat);
            }
        }
    }

    visit::mk_vt(@visit::Visitor {
        visit_local: |l, (cx, vt): (@mut Context, visit::vt<@mut Context>)| {
            if l.node.is_mutbl {
                check_pat(cx, l.node.pat);
            }
            visit::visit_local(l, (cx, vt));
        },
        visit_fn: |a, fd, b, c, d, (cx, vt)| {
            visit_fn_decl(cx, fd);
            visit::visit_fn(a, fd, b, c, d, (cx, vt));
        },
        visit_ty_method: |tm, (cx, vt)| {
            visit_fn_decl(cx, &tm.decl);
            visit::visit_ty_method(tm, (cx, vt));
        },
        visit_trait_method: |tm, (cx, vt)| {
            match *tm {
                ast::required(ref tm) => visit_fn_decl(cx, &tm.decl),
                ast::provided(m) => visit_fn_decl(cx, &m.decl)
            }
            visit::visit_trait_method(tm, (cx, vt));
        },
        .. *visit::default_visitor()
    })
}

fn lint_session() -> visit::vt<@mut Context> {
    ast_util::id_visitor(|id, cx: @mut Context| {
        match cx.tcx.sess.lints.pop(&id) {
            None => {},
            Some(l) => {
                for l.consume_iter().advance |(lint, span, msg)| {
                    cx.span_lint(lint, span, msg)
                }
            }
        }
    })
}

fn lint_unnecessary_allocations() -> visit::vt<@mut Context> {
    // Warn if string and vector literals with sigils are immediately borrowed.
    // Those can have the sigil removed.
    fn check(cx: &Context, e: &ast::expr) {
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

    visit::mk_vt(@visit::Visitor {
        visit_expr: |e, (cx, vt): (@mut Context, visit::vt<@mut Context>)| {
            check(cx, e);
            visit::visit_expr(e, (cx, vt));
        },
        .. *visit::default_visitor()
    })
}

fn lint_missing_doc() -> visit::vt<@mut Context> {
    fn check_attrs(cx: @mut Context, attrs: &[ast::attribute],
                   sp: span, msg: &str) {
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

    visit::mk_vt(@visit::Visitor {
        visit_ty_method: |m, (cx, vt)| {
            // All ty_method objects are linted about because they're part of a
            // trait (no visibility)
            check_attrs(cx, m.attrs, m.span,
                        "missing documentation for a method");
            visit::visit_ty_method(m, (cx, vt));
        },

        visit_fn: |fk, d, b, sp, id, (cx, vt)| {
            // Only warn about explicitly public methods. Soon implicit
            // public-ness will hopefully be going away.
            match *fk {
                visit::fk_method(_, _, m) if m.vis == ast::public => {
                    // If we're in a trait implementation, no need to duplicate
                    // documentation
                    if !cx.in_trait_impl {
                        check_attrs(cx, m.attrs, sp,
                                    "missing documentation for a method");
                    }
                }

                _ => {}
            }
            visit::visit_fn(fk, d, b, sp, id, (cx, vt));
        },

        visit_item: |it, (cx, vt)| {
            match it.node {
                // Go ahead and match the fields here instead of using
                // visit_struct_field while we have access to the enclosing
                // struct's visibility
                ast::item_struct(sdef, _) if it.vis == ast::public => {
                    check_attrs(cx, it.attrs, it.span,
                                "missing documentation for a struct");
                    for sdef.fields.iter().advance |field| {
                        match field.node.kind {
                            ast::named_field(_, vis) if vis != ast::private => {
                                check_attrs(cx, field.node.attrs, field.span,
                                            "missing documentation for a field");
                            }
                            ast::unnamed_field | ast::named_field(*) => {}
                        }
                    }
                }

                ast::item_trait(*) if it.vis == ast::public => {
                    check_attrs(cx, it.attrs, it.span,
                                "missing documentation for a trait");
                }

                ast::item_fn(*) if it.vis == ast::public => {
                    check_attrs(cx, it.attrs, it.span,
                                "missing documentation for a function");
                }

                _ => {}
            };

            visit::visit_item(it, (cx, vt));
        },

        .. *visit::default_visitor()
    })
}

pub fn check_crate(tcx: ty::ctxt, crate: @ast::crate) {
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
    for cx.dict.each_value |spec| {
        cx.set_level(spec.lint, spec.default, Default);
    }

    // Install command-line options, overriding defaults.
    for tcx.sess.opts.lint_opts.iter().advance |&(lint, level)| {
        cx.set_level(lint, level, CommandLine);
    }

    // Register each of the lint passes with the context
    cx.add_lint(lint_while_true());
    cx.add_lint(lint_path_statement());
    cx.add_lint(lint_heap());
    cx.add_lint(lint_type_limits());
    cx.add_lint(lint_unused_unsafe());
    cx.add_lint(lint_copy_implicitly_copyable());
    cx.add_lint(lint_unused_mut());
    cx.add_lint(lint_session());
    cx.add_lint(lint_unnecessary_allocations());
    cx.add_lint(lint_missing_doc());

    // Actually perform the lint checks (iterating the ast)
    do cx.with_lint_attrs(crate.node.attrs) {
        cx.process(Crate(crate));

        visit::visit_crate(crate, (cx, visit::mk_vt(@visit::Visitor {
            visit_item: |it, (cx, vt): (@mut Context, visit::vt<@mut Context>)| {
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
                    check_item_default_methods(cx, it);
                    check_item_heap(cx, it);

                    cx.process(Item(it));
                    visit::visit_item(it, (cx, vt));
                    cx.in_trait_impl = false;
                }
            },
            visit_fn: |fk, decl, body, span, id, (cx, vt)| {
                match *fk {
                    visit::fk_method(_, _, m) => {
                        do cx.with_lint_attrs(m.attrs) {
                            cx.process(Method(m));
                            visit::visit_fn(fk, decl, body, span, id, (cx, vt));
                        }
                    }
                    _ => {
                        visit::visit_fn(fk, decl, body, span, id, (cx, vt));
                    }
                }
            },
            .. *visit::default_visitor()
        })));
    }

    // If we missed any lints added to the session, then there's a bug somewhere
    // in the iteration code.
    for tcx.sess.lints.iter().advance |(_, v)| {
        for v.iter().advance |t| {
            match *t {
                (lint, span, ref msg) =>
                    tcx.sess.span_bug(span, fmt!("unprocessed lint %?: %s",
                                                 lint, *msg))
            }
        }
    }

    tcx.sess.abort_if_errors();
}
