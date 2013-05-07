// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use driver::session::Session;
use driver::session;
use middle::ty;
use middle::pat_util;
use util::ppaux::{ty_to_str};

use core::hashmap::HashMap;
use std::smallintmap::SmallIntMap;
use syntax::attr;
use syntax::codemap::span;
use syntax::codemap;
use syntax::{ast, visit};

/**
 * A 'lint' check is a kind of miscellaneous constraint that a user _might_
 * want to enforce, but might reasonably want to permit as well, on a
 * module-by-module basis. They contrast with static constraints enforced by
 * other phases of the compiler, which are generally required to hold in order
 * to compile the program at all.
 *
 * We also build up a table containing information about lint settings, in
 * order to allow other passes to take advantage of the lint attribute
 * infrastructure. To save space, the table is keyed by the id of /items/, not
 * of every expression. When an item has the default settings, the entry will
 * be omitted. If we start allowing lint attributes on expressions, we will
 * start having entries for expressions that do not share their enclosing
 * items settings.
 *
 * This module then, exports two passes: one that populates the lint
 * settings table in the session and is run early in the compile process, and
 * one that does a variety of lint checks, and is run late in the compile
 * process.
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
    vecs_implicitly_copyable,
    deprecated_pattern,
    non_camel_case_types,
    type_limits,
    default_methods,
    deprecated_mutable_fields,
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

pub type LintDict = @HashMap<~str, LintSpec>;

/*
  Pass names should not contain a '-', as the compiler normalizes
  '-' to '_' in command-line flags
 */
pub fn get_lint_dict() -> LintDict {
    let v = ~[
        (~"ctypes",
         LintSpec {
            lint: ctypes,
            desc: "proper use of core::libc types in foreign modules",
            default: warn
         }),

        (~"unused_imports",
         LintSpec {
            lint: unused_imports,
            desc: "imports that are never used",
            default: warn
         }),

        (~"while_true",
         LintSpec {
            lint: while_true,
            desc: "suggest using loop { } instead of while(true) { }",
            default: warn
         }),

        (~"path_statement",
         LintSpec {
            lint: path_statement,
            desc: "path statements with no effect",
            default: warn
         }),

        (~"unrecognized_lint",
         LintSpec {
            lint: unrecognized_lint,
            desc: "unrecognized lint attribute",
            default: warn
         }),

        (~"non_implicitly_copyable_typarams",
         LintSpec {
            lint: non_implicitly_copyable_typarams,
            desc: "passing non implicitly copyable types as copy type params",
            default: warn
         }),

        (~"vecs_implicitly_copyable",
         LintSpec {
            lint: vecs_implicitly_copyable,
            desc: "make vecs and strs not implicitly copyable \
                  (only checked at top level)",
            default: warn
         }),

        (~"implicit_copies",
         LintSpec {
            lint: implicit_copies,
            desc: "implicit copies of non implicitly copyable data",
            default: warn
         }),

        (~"deprecated_pattern",
         LintSpec {
            lint: deprecated_pattern,
            desc: "warn about deprecated uses of pattern bindings",
            default: allow
         }),

        (~"non_camel_case_types",
         LintSpec {
            lint: non_camel_case_types,
            desc: "types, variants and traits should have camel case names",
            default: allow
         }),

        (~"managed_heap_memory",
         LintSpec {
            lint: managed_heap_memory,
            desc: "use of managed (@ type) heap memory",
            default: allow
         }),

        (~"owned_heap_memory",
         LintSpec {
            lint: owned_heap_memory,
            desc: "use of owned (~ type) heap memory",
            default: allow
         }),

        (~"heap_memory",
         LintSpec {
            lint: heap_memory,
            desc: "use of any (~ type or @ type) heap memory",
            default: allow
         }),

        (~"type_limits",
         LintSpec {
            lint: type_limits,
            desc: "comparisons made useless by limits of the types involved",
            default: warn
         }),

        (~"default_methods",
         LintSpec {
            lint: default_methods,
            desc: "allow default methods",
            default: deny
         }),

        (~"deprecated_mutable_fields",
         LintSpec {
            lint: deprecated_mutable_fields,
            desc: "deprecated mutable fields in structures",
            default: deny
        }),

        (~"unused_unsafe",
         LintSpec {
            lint: unused_unsafe,
            desc: "unnecessary use of an `unsafe` block",
            default: warn
        }),

        (~"unused_variable",
         LintSpec {
            lint: unused_variable,
            desc: "detect variables which are not used in any way",
            default: warn
        }),

        (~"dead_assignment",
         LintSpec {
            lint: dead_assignment,
            desc: "detect assignments that will never be read",
            default: warn
        }),

        (~"unused_mut",
         LintSpec {
            lint: unused_mut,
            desc: "detect mut variables which don't need to be mutable",
            default: warn
        }),
    ];
    let mut map = HashMap::new();
    do vec::consume(v) |_, (k, v)| {
        map.insert(k, v);
    }
    return @map;
}

// This is a highly not-optimal set of data structure decisions.
type LintModes = @mut SmallIntMap<level>;
type LintModeMap = @mut HashMap<ast::node_id, LintModes>;

// settings_map maps node ids of items with non-default lint settings
// to their settings; default_settings contains the settings for everything
// not in the map.
pub struct LintSettings {
    default_settings: LintModes,
    settings_map: LintModeMap
}

pub fn mk_lint_settings() -> LintSettings {
    LintSettings {
        default_settings: @mut SmallIntMap::new(),
        settings_map: @mut HashMap::new()
    }
}

pub fn get_lint_level(modes: LintModes, lint: lint) -> level {
    match modes.find(&(lint as uint)) {
      Some(&c) => c,
      None => allow
    }
}

pub fn get_lint_settings_level(settings: LintSettings,
                               lint_mode: lint,
                               _expr_id: ast::node_id,
                               item_id: ast::node_id)
                            -> level {
    match settings.settings_map.find(&item_id) {
      Some(&modes) => get_lint_level(modes, lint_mode),
      None => get_lint_level(settings.default_settings, lint_mode)
    }
}

// This is kind of unfortunate. It should be somewhere else, or we should use
// a persistent data structure...
fn clone_lint_modes(modes: LintModes) -> LintModes {
    @mut (copy *modes)
}

struct Context {
    dict: LintDict,
    curr: LintModes,
    is_default: bool,
    sess: Session
}

pub impl Context {
    fn get_level(&self, lint: lint) -> level {
        get_lint_level(self.curr, lint)
    }

    fn set_level(&self, lint: lint, level: level) {
        if level == allow {
            self.curr.remove(&(lint as uint));
        } else {
            self.curr.insert(lint as uint, level);
        }
    }

    fn span_lint(&self, level: level, span: span, msg: ~str) {
        self.sess.span_lint_level(level, span, msg);
    }

    /**
     * Merge the lints specified by any lint attributes into the
     * current lint context, call the provided function, then reset the
     * lints in effect to their previous state.
     */
    fn with_lint_attrs(&self, attrs: ~[ast::attribute], f: &fn(Context)) {

        let mut new_ctxt = *self;
        let mut triples = ~[];

        for [allow, warn, deny, forbid].each |level| {
            let level_name = level_to_str(*level);
            let metas =
                attr::attr_metas(attr::find_attrs_by_name(attrs, level_name));
            for metas.each |meta| {
                match meta.node {
                  ast::meta_list(_, ref metas) => {
                    for metas.each |meta| {
                        match meta.node {
                          ast::meta_word(ref lintname) => {
                            triples.push((*meta,
                                          *level,
                                          /*bad*/copy *lintname));
                          }
                          _ => {
                            self.sess.span_err(
                                meta.span,
                                "malformed lint attribute");
                          }
                        }
                    }
                  }
                  _  => {
                    self.sess.span_err(meta.span,
                                       "malformed lint attribute");
                  }
                }
            }
        }

        for triples.each |triple| {
            // FIXME(#3874): it would be nicer to write this...
            // let (meta, level, lintname) = /*bad*/copy *pair;
            let (meta, level, lintname) = match *triple {
                (ref meta, level, lintname) => (meta, level, lintname)
            };

            match self.dict.find(lintname) {
              None => {
                self.span_lint(
                    new_ctxt.get_level(unrecognized_lint),
                    meta.span,
                    fmt!("unknown `%s` attribute: `%s`",
                         level_to_str(level), *lintname));
              }
              Some(lint) => {

                if new_ctxt.get_level(lint.lint) == forbid &&
                    level != forbid {
                    self.span_lint(
                        forbid,
                        meta.span,
                        fmt!("%s(%s) overruled by outer forbid(%s)",
                             level_to_str(level),
                             *lintname, *lintname));
                }

                // we do multiple unneeded copies of the
                // map if many attributes are set, but
                // this shouldn't actually be a problem...

                let c = clone_lint_modes(new_ctxt.curr);
                new_ctxt = Context {
                    is_default: false,
                    curr: c,
                    .. new_ctxt
                };
                new_ctxt.set_level(lint.lint, level);
              }
            }
        }
        f(new_ctxt);
    }
}


fn build_settings_item(i: @ast::item, cx: Context, v: visit::vt<Context>) {
    do cx.with_lint_attrs(/*bad*/copy i.attrs) |cx| {
        if !cx.is_default {
            cx.sess.lint_settings.settings_map.insert(i.id, cx.curr);
        }
        visit::visit_item(i, cx, v);
    }
}

pub fn build_settings_crate(sess: session::Session, crate: @ast::crate) {
    let cx = Context {
        dict: get_lint_dict(),
        curr: @mut SmallIntMap::new(),
        is_default: true,
        sess: sess
    };

    // Install defaults.
    for cx.dict.each_value |&spec| {
        cx.set_level(spec.lint, spec.default);
    }

    // Install command-line options, overriding defaults.
    for sess.opts.lint_opts.each |pair| {
        let (lint,level) = *pair;
        cx.set_level(lint, level);
    }

    do cx.with_lint_attrs(/*bad*/copy crate.node.attrs) |cx| {
        // Copy out the default settings
        for cx.curr.each |&k, &v| {
            sess.lint_settings.default_settings.insert(k, v);
        }

        let cx = Context {
            is_default: true,
            .. cx
        };

        let visit = visit::mk_vt(@visit::Visitor {
            visit_item: build_settings_item,
            .. *visit::default_visitor()
        });
        visit::visit_crate(crate, cx, visit);
    }

    sess.abort_if_errors();
}

fn check_item(i: @ast::item, cx: ty::ctxt) {
    check_item_ctypes(cx, i);
    check_item_while_true(cx, i);
    check_item_path_statement(cx, i);
    check_item_non_camel_case_types(cx, i);
    check_item_heap(cx, i);
    check_item_type_limits(cx, i);
    check_item_default_methods(cx, i);
    check_item_deprecated_mutable_fields(cx, i);
    check_item_unused_unsafe(cx, i);
    check_item_unused_mut(cx, i);
}

// Take a visitor, and modify it so that it will not proceed past subitems.
// This is used to make the simple visitors used for the lint passes
// not traverse into subitems, since that is handled by the outer
// lint visitor.
fn item_stopping_visitor<E>(v: visit::vt<E>) -> visit::vt<E> {
    visit::mk_vt(@visit::Visitor {visit_item: |_i, _e, _v| { },
        .. **(ty_stopping_visitor(v))})
}

fn ty_stopping_visitor<E>(v: visit::vt<E>) -> visit::vt<E> {
    visit::mk_vt(@visit::Visitor {visit_ty: |_t, _e, _v| { },.. **v})
}

fn check_item_while_true(cx: ty::ctxt, it: @ast::item) {
    let visit = item_stopping_visitor(
        visit::mk_simple_visitor(@visit::SimpleVisitor {
            visit_expr: |e: @ast::expr| {
                match e.node {
                    ast::expr_while(cond, _) => {
                        match cond.node {
                            ast::expr_lit(@codemap::spanned {
                                node: ast::lit_bool(true), _}) =>
                            {
                                cx.sess.span_lint(
                                    while_true, e.id, it.id,
                                    e.span,
                                    "denote infinite loops \
                                     with loop { ... }");
                            }
                            _ => ()
                        }
                    }
                    _ => ()
                }
            },
            .. *visit::default_simple_visitor()
        }));
    visit::visit_item(it, (), visit);
}

fn check_item_type_limits(cx: ty::ctxt, it: @ast::item) {
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

    fn int_ty_range(int_ty: ast::int_ty) -> (i64, i64) {
        match int_ty {
            ast::ty_i =>    (int::min_value as i64, int::max_value as i64),
            ast::ty_char => (u32::min_value as i64, u32::max_value as i64),
            ast::ty_i8 =>   (i8::min_value  as i64, i8::max_value  as i64),
            ast::ty_i16 =>  (i16::min_value as i64, i16::max_value as i64),
            ast::ty_i32 =>  (i32::min_value as i64, i32::max_value as i64),
            ast::ty_i64 =>  (i64::min_value,        i64::max_value)
        }
    }

    fn uint_ty_range(uint_ty: ast::uint_ty) -> (u64, u64) {
        match uint_ty {
            ast::ty_u =>   (uint::min_value as u64, uint::max_value as u64),
            ast::ty_u8 =>  (u8::min_value   as u64, u8::max_value   as u64),
            ast::ty_u16 => (u16::min_value  as u64, u16::max_value  as u64),
            ast::ty_u32 => (u32::min_value  as u64, u32::max_value  as u64),
            ast::ty_u64 => (u64::min_value,         u64::max_value)
        }
    }

    fn check_limits(cx: ty::ctxt, binop: ast::binop, l: &ast::expr,
                    r: &ast::expr) -> bool {
        let (lit, expr, swap) = match (&l.node, &r.node) {
            (&ast::expr_lit(_), _) => (l, r, true),
            (_, &ast::expr_lit(_)) => (r, l, false),
            _ => return true
        };
        // Normalize the binop so that the literal is always on the RHS in
        // the comparison
        let norm_binop = if (swap) {
            rev_binop(binop)
        } else {
            binop
        };
        match ty::get(ty::expr_ty(cx, @/*bad*/copy *expr)).sty {
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
                    cx.sess.span_lint(
                        type_limits, e.id, it.id, e.span,
                        "comparison is useless due to type limits");
                }
            }
            _ => ()
        }
    };

    let visit = item_stopping_visitor(
        visit::mk_simple_visitor(@visit::SimpleVisitor {
            visit_expr: visit_expr,
            .. *visit::default_simple_visitor()
        }));
    visit::visit_item(it, (), visit);
}

fn check_item_default_methods(cx: ty::ctxt, item: @ast::item) {
    match item.node {
        ast::item_trait(_, _, ref methods) => {
            for methods.each |method| {
                match *method {
                    ast::required(*) => {}
                    ast::provided(*) => {
                        cx.sess.span_lint(
                            default_methods,
                            item.id,
                            item.id,
                            item.span,
                            "default methods are experimental");
                    }
                }
            }
        }
        _ => {}
    }
}

fn check_item_deprecated_mutable_fields(cx: ty::ctxt, item: @ast::item) {
    match item.node {
        ast::item_struct(struct_def, _) => {
            for struct_def.fields.each |field| {
                match field.node.kind {
                    ast::named_field(_, ast::struct_mutable, _) => {
                        cx.sess.span_lint(deprecated_mutable_fields,
                                          item.id,
                                          item.id,
                                          field.span,
                                          "mutable fields are deprecated");
                    }
                    ast::named_field(*) | ast::unnamed_field => {}
                }
            }
        }
        _ => {}
    }
}

fn check_item_ctypes(cx: ty::ctxt, it: @ast::item) {

    fn check_foreign_fn(cx: ty::ctxt, fn_id: ast::node_id,
                        decl: &ast::fn_decl) {
        let tys = vec::map(decl.inputs, |a| a.ty );
        for vec::each(vec::append_one(tys, decl.output)) |ty| {
            match ty.node {
              ast::ty_path(_, id) => {
                match cx.def_map.get_copy(&id) {
                  ast::def_prim_ty(ast::ty_int(ast::ty_i)) => {
                    cx.sess.span_lint(
                        ctypes, id, fn_id,
                        ty.span,
                        "found rust type `int` in foreign module, while \
                         libc::c_int or libc::c_long should be used");
                  }
                  ast::def_prim_ty(ast::ty_uint(ast::ty_u)) => {
                    cx.sess.span_lint(
                        ctypes, id, fn_id,
                        ty.span,
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
                check_foreign_fn(cx, it.id, decl);
              }
              // FIXME #4622: Not implemented.
              ast::foreign_item_const(*) => {}
            }
        }
      }
      _ => {/* nothing to do */ }
    }
}

fn check_item_heap(cx: ty::ctxt, it: @ast::item) {

    fn check_type_for_lint(cx: ty::ctxt, lint: lint,
                           node: ast::node_id,
                           item: ast::node_id,
                           span: span, ty: ty::t) {

        if get_lint_settings_level(cx.sess.lint_settings,
                                   lint, node, item) != allow {
            let mut n_box = 0;
            let mut n_uniq = 0;
            ty::fold_ty(cx, ty, |t| {
                match ty::get(t).sty {
                  ty::ty_box(_) => n_box += 1,
                  ty::ty_uniq(_) => n_uniq += 1,
                  _ => ()
                };
                t
            });

            if (n_uniq > 0 && lint != managed_heap_memory) {
                let s = ty_to_str(cx, ty);
                let m = ~"type uses owned (~ type) pointers: " + s;
                cx.sess.span_lint(lint, node, item, span, m);
            }

            if (n_box > 0 && lint != owned_heap_memory) {
                let s = ty_to_str(cx, ty);
                let m = ~"type uses managed (@ type) pointers: " + s;
                cx.sess.span_lint(lint, node, item, span, m);
            }
        }
    }

    fn check_type(cx: ty::ctxt,
                  node: ast::node_id,
                  item: ast::node_id,
                  span: span, ty: ty::t) {
            for [managed_heap_memory,
                 owned_heap_memory,
                 heap_memory].each |lint| {
                check_type_for_lint(cx, *lint, node, item, span, ty);
            }
    }

    match it.node {
      ast::item_fn(*) |
      ast::item_ty(*) |
      ast::item_enum(*) |
      ast::item_struct(*) => check_type(cx, it.id, it.id, it.span,
                                       ty::node_id_to_type(cx, it.id)),
      _ => ()
    }

    // If it's a struct, we also have to check the fields' types
    match it.node {
        ast::item_struct(struct_def, _) => {
            for struct_def.fields.each |struct_field| {
                check_type(cx, struct_field.node.id, it.id,
                           struct_field.span,
                           ty::node_id_to_type(cx, struct_field.node.id));
            }
        }
        _ => ()
    }

    let visit = item_stopping_visitor(
        visit::mk_simple_visitor(@visit::SimpleVisitor {
            visit_expr: |e: @ast::expr| {
                let ty = ty::expr_ty(cx, e);
                check_type(cx, e.id, it.id, e.span, ty);
            },
            .. *visit::default_simple_visitor()
        }));
    visit::visit_item(it, (), visit);
}

fn check_item_path_statement(cx: ty::ctxt, it: @ast::item) {
    let visit = item_stopping_visitor(
        visit::mk_simple_visitor(@visit::SimpleVisitor {
            visit_stmt: |s: @ast::stmt| {
                match s.node {
                    ast::stmt_semi(
                        @ast::expr { id: id, node: ast::expr_path(_), _ },
                        _
                    ) => {
                        cx.sess.span_lint(
                            path_statement, id, it.id,
                            s.span,
                            "path statement with no effect");
                    }
                    _ => ()
                }
            },
            .. *visit::default_simple_visitor()
        }));
    visit::visit_item(it, (), visit);
}

fn check_item_non_camel_case_types(cx: ty::ctxt, it: @ast::item) {
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

    fn check_case(cx: ty::ctxt, ident: ast::ident,
                  expr_id: ast::node_id, item_id: ast::node_id,
                  span: span) {
        if !is_camel_case(cx, ident) {
            cx.sess.span_lint(
                non_camel_case_types, expr_id, item_id, span,
                "type, variant, or trait should have \
                 a camel case identifier");
        }
    }

    match it.node {
        ast::item_ty(*) | ast::item_struct(*) |
        ast::item_trait(*) => {
            check_case(cx, it.ident, it.id, it.id, it.span)
        }
        ast::item_enum(ref enum_definition, _) => {
            check_case(cx, it.ident, it.id, it.id, it.span);
            for enum_definition.variants.each |variant| {
                check_case(cx, variant.node.name,
                           variant.node.id, it.id, variant.span);
            }
        }
        _ => ()
    }
}

fn check_item_unused_unsafe(cx: ty::ctxt, it: @ast::item) {
    let visit_expr: @fn(@ast::expr) = |e| {
        match e.node {
            ast::expr_block(ref blk) if blk.node.rules == ast::unsafe_blk => {
                if !cx.used_unsafe.contains(&blk.node.id) {
                    cx.sess.span_lint(unused_unsafe, blk.node.id, it.id,
                                      blk.span,
                                      "unnecessary `unsafe` block");
                }
            }
            _ => ()
        }
    };

    let visit = item_stopping_visitor(
        visit::mk_simple_visitor(@visit::SimpleVisitor {
            visit_expr: visit_expr,
            .. *visit::default_simple_visitor()
        }));
    visit::visit_item(it, (), visit);
}

fn check_item_unused_mut(tcx: ty::ctxt, it: @ast::item) {
    let check_pat: @fn(@ast::pat) = |p| {
        let mut used = false;
        let mut bindings = 0;
        do pat_util::pat_bindings(tcx.def_map, p) |_, id, _, _| {
            used = used || tcx.used_mut_nodes.contains(&id);
            bindings += 1;
        }
        if !used {
            let msg = if bindings == 1 {
                "variable does not need to be mutable"
            } else {
                "variables do not need to be mutable"
            };
            tcx.sess.span_lint(unused_mut, p.id, it.id, p.span, msg);
        }
    };

    let visit_fn_decl: @fn(&ast::fn_decl) = |fd| {
        for fd.inputs.each |arg| {
            if arg.is_mutbl {
                check_pat(arg.pat);
            }
        }
    };

    let visit = item_stopping_visitor(
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
        }));
    visit::visit_item(it, (), visit);
}

fn check_fn(_: ty::ctxt,
            fk: &visit::fn_kind,
            _: &ast::fn_decl,
            _: &ast::blk,
            _: span,
            id: ast::node_id) {
    debug!("lint check_fn fk=%? id=%?", fk, id);
}

pub fn check_crate(tcx: ty::ctxt, crate: @ast::crate) {
    let v = visit::mk_simple_visitor(@visit::SimpleVisitor {
        visit_item: |it|
            check_item(it, tcx),
        visit_fn: |fk, decl, body, span, id|
            check_fn(tcx, fk, decl, body, span, id),
        .. *visit::default_simple_visitor()
    });
    visit::visit_crate(crate, (), v);

    tcx.sess.abort_if_errors();
}
