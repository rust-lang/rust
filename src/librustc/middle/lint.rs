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

use core::hashmap::HashMap;
use std::smallintmap::SmallIntMap;
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
    while_true,
    path_statement,
    implicit_copies,
    unrecognized_lint,
    non_implicitly_copyable_typarams,
    deprecated_pattern,
    non_camel_case_types,
    type_limits,
    default_methods,
    unused_unsafe,

    managed_heap_memory,
    owned_heap_memory,
    heap_memory,

    unused_variable,
    dead_assignment,
    unused_mut,
}

pub fn level_to_str(lv: level) -> &'static str {
    match lv {
      allow => "allow",
      warn => "warn",
      deny => "deny",
      forbid => "forbid"
    }
}

#[deriving(Eq)]
pub enum level {
    allow, warn, deny, forbid
}

struct LintSpec {
    lint: lint,
    desc: &'static str,
    default: level
}

pub type LintDict = HashMap<~str, LintSpec>;

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
        desc: "proper use of core::libc types in foreign modules",
        default: warn
     }),

    ("unused_imports",
     LintSpec {
        lint: unused_imports,
        desc: "imports that are never used",
        default: warn
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
        default: deny
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
];

/*
  Pass names should not contain a '-', as the compiler normalizes
  '-' to '_' in command-line flags
 */
pub fn get_lint_dict() -> LintDict {
    let mut map = HashMap::new();
    for lint_table.each|&(k, v)| {
        map.insert(k.to_str(), v);
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
    // When recursing into an attributed node of the ast which modifies lint
    // levels, this stack keeps track of the previous lint levels of whatever
    // was modified.
    lint_stack: ~[(lint, level, LintSource)],
    // Each of these visitors represents a lint pass. A number of the lint
    // attributes are registered by adding a visitor to iterate over the ast.
    // Others operate directly on @ast::item structures (or similar). Finally,
    // others still are added to the Session object via `add_lint`, and these
    // are all passed with the lint_session visitor.
    visitors: ~[visit::vt<()>],
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

    fn lint_to_str(&self, lint: lint) -> ~str {
        for self.dict.each |k, v| {
            if v.lint == lint {
                return copy *k;
            }
        }
        fail!("unregistered lint %?", lint);
    }

    fn span_lint(&self, lint: lint, span: span, msg: &str) {
        let (level, src) = match self.curr.find(&(lint as uint)) {
            Some(&pair) => pair,
            None => { return; }
        };
        if level == allow { return; }

        let mut note = None;
        let msg = match src {
            Default | CommandLine => {
                fmt!("%s [-%c %s%s]", msg, match level {
                        warn => 'W', deny => 'D', forbid => 'F',
                        allow => fail!()
                    }, str::replace(self.lint_to_str(lint), "_", "-"),
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

        for note.each |&span| {
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
            let lint = match self.dict.find(lintname) {
              None => {
                self.span_lint(
                    unrecognized_lint,
                    meta.span,
                    fmt!("unknown `%s` attribute: `%s`",
                         level_to_str(level), *lintname));
                loop
              }
              Some(lint) => { lint.lint }
            };

            let now = self.get_level(lint);
            if now == forbid && level != forbid {
                self.tcx.sess.span_err(meta.span,
                    fmt!("%s(%s) overruled by outer forbid(%s)",
                         level_to_str(level),
                         *lintname, *lintname));
                loop;
            }

            if now != level {
                let src = self.get_source(lint);
                self.lint_stack.push((lint, now, src));
                pushed += 1;
                self.set_level(lint, level, Node(meta.span));
            }
        }

        f();

        // rollback
        for pushed.times {
            let (lint, lvl, src) = self.lint_stack.pop();
            self.set_level(lint, lvl, src);
        }
    }

    fn add_lint(&mut self, v: visit::vt<()>) {
        self.visitors.push(item_stopping_visitor(v));
    }

    fn process(&self, n: AttributedNode) {
        match n {
            Item(it) => {
                for self.visitors.each |v| {
                    visit::visit_item(it, (), *v);
                }
            }
            Crate(c) => {
                for self.visitors.each |v| {
                    visit::visit_crate(c, (), *v);
                }
            }
            // Can't use visit::visit_method_helper because the
            // item_stopping_visitor has overridden visit_fn(&fk_method(... ))
            // to be a no-op, so manually invoke visit_fn.
            Method(m) => {
                let fk = visit::fk_method(copy m.ident, &m.generics, m);
                for self.visitors.each |v| {
                    visit::visit_fn(&fk, &m.decl, &m.body, m.span, m.id,
                                    (), *v);
                }
            }
        }
    }
}

#[cfg(stage0)]
pub fn each_lint(sess: session::Session,
                 attrs: &[ast::attribute],
                 f: &fn(@ast::meta_item, level, &~str) -> bool)
{
    for [allow, warn, deny, forbid].each |&level| {
        let level_name = level_to_str(level);
        let attrs = attr::find_attrs_by_name(attrs, level_name);
        for attrs.each |attr| {
            let meta = attr.node.value;
            let metas = match meta.node {
                ast::meta_list(_, ref metas) => metas,
                _ => {
                    sess.span_err(meta.span, ~"malformed lint attribute");
                    loop;
                }
            };
            for metas.each |meta| {
                match meta.node {
                    ast::meta_word(lintname) => {
                        if !f(*meta, level, lintname) {
                            return;
                        }
                    }
                    _ => {
                        sess.span_err(meta.span, ~"malformed lint attribute");
                    }
                }
            }
        }
    }
}
#[cfg(not(stage0))]
pub fn each_lint(sess: session::Session,
                 attrs: &[ast::attribute],
                 f: &fn(@ast::meta_item, level, &~str) -> bool) -> bool
{
    for [allow, warn, deny, forbid].each |&level| {
        let level_name = level_to_str(level);
        let attrs = attr::find_attrs_by_name(attrs, level_name);
        for attrs.each |attr| {
            let meta = attr.node.value;
            let metas = match meta.node {
                ast::meta_list(_, ref metas) => metas,
                _ => {
                    sess.span_err(meta.span, ~"malformed lint attribute");
                    loop;
                }
            };
            for metas.each |meta| {
                match meta.node {
                    ast::meta_word(lintname) => {
                        if !f(*meta, level, lintname) {
                            return false;
                        }
                    }
                    _ => {
                        sess.span_err(meta.span, ~"malformed lint attribute");
                    }
                }
            }
        }
    }
    return true;
}

// Take a visitor, and modify it so that it will not proceed past subitems.
// This is used to make the simple visitors used for the lint passes
// not traverse into subitems, since that is handled by the outer
// lint visitor.
fn item_stopping_visitor<E: Copy>(v: visit::vt<E>) -> visit::vt<E> {
    visit::mk_vt(@visit::Visitor {
        visit_item: |_i, _e, _v| { },
        visit_fn: |fk, fd, b, s, id, e, v| {
            match *fk {
                visit::fk_method(*) => {}
                _ => visit::visit_fn(fk, fd, b, s, id, e, v)
            }
        },
    .. **(ty_stopping_visitor(v))})
}

fn ty_stopping_visitor<E>(v: visit::vt<E>) -> visit::vt<E> {
    visit::mk_vt(@visit::Visitor {visit_ty: |_t, _e, _v| { },.. **v})
}

fn lint_while_true(cx: @mut Context) -> visit::vt<()> {
    visit::mk_simple_visitor(@visit::SimpleVisitor {
        visit_expr: |e: @ast::expr| {
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
        },
        .. *visit::default_simple_visitor()
    })
}

fn lint_type_limits(cx: @mut Context) -> visit::vt<()> {
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

    fn check_limits(cx: @mut Context, binop: ast::binop, l: &ast::expr,
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

    let visit_expr: @fn(@ast::expr) = |e| {
        match e.node {
            ast::expr_binary(ref binop, @ref l, @ref r) => {
                if is_comparison(*binop)
                    && !check_limits(cx, *binop, l, r) {
                    cx.span_lint(type_limits, e.span,
                                 "comparison is useless due to type limits");
                }
            }
            _ => ()
        }
    };

    visit::mk_simple_visitor(@visit::SimpleVisitor {
        visit_expr: visit_expr,
        .. *visit::default_simple_visitor()
    })
}

fn check_item_default_methods(cx: @mut Context, item: @ast::item) {
    match item.node {
        ast::item_trait(_, _, ref methods) => {
            for methods.each |method| {
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

fn check_item_ctypes(cx: @mut Context, it: @ast::item) {

    fn check_foreign_fn(cx: @mut Context, decl: &ast::fn_decl) {
        let tys = vec::map(decl.inputs, |a| a.ty );
        for vec::each(vec::append_one(tys, decl.output)) |ty| {
            match ty.node {
              ast::ty_path(_, id) => {
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
              _ => ()
            }
        }
    }

    match it.node {
      ast::item_foreign_mod(ref nmod) if !nmod.abis.is_intrinsic() => {
        for nmod.items.each |ni| {
            match ni.node {
              ast::foreign_item_fn(ref decl, _, _) => {
                check_foreign_fn(cx, decl);
              }
              // FIXME #4622: Not implemented.
              ast::foreign_item_const(*) => {}
            }
        }
      }
      _ => {/* nothing to do */ }
    }
}

fn check_type_for_lint(cx: @mut Context, lint: lint, span: span, ty: ty::t) {
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

fn check_type(cx: @mut Context, span: span, ty: ty::t) {
    for [managed_heap_memory, owned_heap_memory, heap_memory].each |lint| {
        check_type_for_lint(cx, *lint, span, ty);
    }
}

fn check_item_heap(cx: @mut Context, it: @ast::item) {
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
            for struct_def.fields.each |struct_field| {
                check_type(cx, struct_field.span,
                           ty::node_id_to_type(cx.tcx,
                                               struct_field.node.id));
            }
        }
        _ => ()
    }
}

fn lint_heap(cx: @mut Context) -> visit::vt<()> {
    visit::mk_simple_visitor(@visit::SimpleVisitor {
        visit_expr: |e| {
            let ty = ty::expr_ty(cx.tcx, e);
            check_type(cx, e.span, ty);
        },
        .. *visit::default_simple_visitor()
    })
}

fn lint_path_statement(cx: @mut Context) -> visit::vt<()> {
    visit::mk_simple_visitor(@visit::SimpleVisitor {
        visit_stmt: |s| {
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
        },
        .. *visit::default_simple_visitor()
    })
}

fn check_item_non_camel_case_types(cx: @mut Context, it: @ast::item) {
    fn is_camel_case(cx: ty::ctxt, ident: ast::ident) -> bool {
        let ident = cx.sess.str_of(ident);
        assert!(!ident.is_empty());
        let ident = ident_without_trailing_underscores(*ident);
        let ident = ident_without_leading_underscores(ident);
        char::is_uppercase(str::char_at(ident, 0)) &&
            !ident.contains_char('_')
    }

    fn ident_without_trailing_underscores<'r>(ident: &'r str) -> &'r str {
        match str::rfind(ident, |c| c != '_') {
            Some(idx) => str::slice(ident, 0, idx + 1),
            None => ident, // all underscores
        }
    }

    fn ident_without_leading_underscores<'r>(ident: &'r str) -> &'r str {
        match str::find(ident, |c| c != '_') {
            Some(idx) => str::slice(ident, idx, ident.len()),
            None => ident // all underscores
        }
    }

    fn check_case(cx: @mut Context, ident: ast::ident, span: span) {
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
            for enum_definition.variants.each |variant| {
                check_case(cx, variant.node.name, variant.span);
            }
        }
        _ => ()
    }
}

fn lint_unused_unsafe(cx: @mut Context) -> visit::vt<()> {
    let visit_expr: @fn(@ast::expr) = |e| {
        match e.node {
            ast::expr_block(ref blk) if blk.node.rules == ast::unsafe_blk => {
                if !cx.tcx.used_unsafe.contains(&blk.node.id) {
                    cx.span_lint(unused_unsafe, blk.span,
                                 "unnecessary `unsafe` block");
                }
            }
            _ => ()
        }
    };

    visit::mk_simple_visitor(@visit::SimpleVisitor {
        visit_expr: visit_expr,
        .. *visit::default_simple_visitor()
    })
}

fn lint_unused_mut(cx: @mut Context) -> visit::vt<()> {
    let check_pat: @fn(@ast::pat) = |p| {
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
    };

    let visit_fn_decl: @fn(&ast::fn_decl) = |fd| {
        for fd.inputs.each |arg| {
            if arg.is_mutbl {
                check_pat(arg.pat);
            }
        }
    };

    visit::mk_simple_visitor(@visit::SimpleVisitor {
        visit_local: |l| {
            if l.node.is_mutbl {
                check_pat(l.node.pat);
            }
        },
        visit_fn: |_, fd, _, _, _| visit_fn_decl(fd),
        visit_ty_method: |tm| visit_fn_decl(&tm.decl),
        visit_struct_method: |sm| visit_fn_decl(&sm.decl),
        visit_trait_method: |tm| {
            match *tm {
                ast::required(ref tm) => visit_fn_decl(&tm.decl),
                ast::provided(m) => visit_fn_decl(&m.decl),
            }
        },
        .. *visit::default_simple_visitor()
    })
}

fn lint_session(cx: @mut Context) -> visit::vt<()> {
    ast_util::id_visitor(|id| {
        match cx.tcx.sess.lints.pop(&id) {
            None => {},
            Some(l) => {
                do vec::consume(l) |_, (lint, span, msg)| {
                    cx.span_lint(lint, span, msg)
                }
            }
        }
    })
}

pub fn check_crate(tcx: ty::ctxt, crate: @ast::crate) {
    let cx = @mut Context {
        dict: @get_lint_dict(),
        curr: SmallIntMap::new(),
        tcx: tcx,
        lint_stack: ~[],
        visitors: ~[],
    };

    // Install defaults.
    for cx.dict.each_value |spec| {
        cx.set_level(spec.lint, spec.default, Default);
    }

    // Install command-line options, overriding defaults.
    for tcx.sess.opts.lint_opts.each |&(lint, level)| {
        cx.set_level(lint, level, CommandLine);
    }

    // Register each of the lint passes with the context
    cx.add_lint(lint_while_true(cx));
    cx.add_lint(lint_path_statement(cx));
    cx.add_lint(lint_heap(cx));
    cx.add_lint(lint_type_limits(cx));
    cx.add_lint(lint_unused_unsafe(cx));
    cx.add_lint(lint_unused_mut(cx));
    cx.add_lint(lint_session(cx));

    // type inference doesn't like this being declared below, we need to tell it
    // what the type of this first function is...
    let visit_item:
        @fn(@ast::item, @mut Context, visit::vt<@mut Context>) =
    |it, cx, vt| {
        do cx.with_lint_attrs(it.attrs) {
            check_item_ctypes(cx, it);
            check_item_non_camel_case_types(cx, it);
            check_item_default_methods(cx, it);
            check_item_heap(cx, it);

            cx.process(Item(it));
            visit::visit_item(it, cx, vt);
        }
    };

    // Actually perform the lint checks (iterating the ast)
    do cx.with_lint_attrs(crate.node.attrs) {
        cx.process(Crate(crate));

        visit::visit_crate(crate, cx, visit::mk_vt(@visit::Visitor {
            visit_item: visit_item,
            visit_fn: |fk, decl, body, span, id, cx, vt| {
                match *fk {
                    visit::fk_method(_, _, m) => {
                        do cx.with_lint_attrs(m.attrs) {
                            cx.process(Method(m));
                            visit::visit_fn(fk, decl, body, span, id, cx, vt);
                        }
                    }
                    _ => {
                        visit::visit_fn(fk, decl, body, span, id, cx, vt);
                    }
                }
            },
            .. *visit::default_visitor()
        }));
    }

    // If we missed any lints added to the session, then there's a bug somewhere
    // in the iteration code.
    for tcx.sess.lints.each |_, v| {
        for v.each |t| {
            match *t {
                (lint, span, ref msg) =>
                    tcx.sess.span_bug(span, fmt!("unprocessed lint %?: %s",
                                                 lint, *msg))
            }
        }
    }

    tcx.sess.abort_if_errors();
}
