import driver::session;
import driver::session::session;
import middle::ty;
import syntax::{ast, visit};
import syntax::attr;
import syntax::codemap::span;
import std::map::{map,hashmap,int_hash,hash_from_strs};
import std::smallintmap::{map,smallintmap};
import io::writer_util;
import util::ppaux::{ty_to_str};
import syntax::print::pprust::{expr_to_str, mode_to_str};
export lint, ctypes, unused_imports, while_true, path_statement, old_vecs;
export unrecognized_lint, non_implicitly_copyable_typarams;
export vecs_implicitly_copyable, implicit_copies;
export level, allow, warn, deny, forbid;
export lint_dict, get_lint_dict, level_to_str;
export get_lint_level, get_lint_settings_level;
export check_crate, build_settings_crate, mk_lint_settings;
export lint_settings;

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

enum lint {
    ctypes,
    unused_imports,
    while_true,
    path_statement,
    implicit_copies,
    unrecognized_lint,
    non_implicitly_copyable_typarams,
    vecs_implicitly_copyable,
    deprecated_mode,
    non_camel_case_types
}

// This is pretty unfortunate. We really want some sort of "deriving Enum"
// type of thing.
fn int_to_lint(i: int) -> lint {
    alt check i {
      0 { ctypes }
      1 { unused_imports }
      2 { while_true }
      3 { path_statement }
      4 { implicit_copies }
      5 { unrecognized_lint }
      6 { non_implicitly_copyable_typarams }
      7 { vecs_implicitly_copyable }
      8 { deprecated_mode }
      9 { non_camel_case_types }
    }
}

fn level_to_str(lv: level) -> ~str {
    alt lv {
      allow { ~"allow" }
      warn { ~"warn" }
      deny { ~"deny" }
      forbid { ~"forbid" }
    }
}

enum level {
    allow, warn, deny, forbid
}

type lint_spec = @{lint: lint,
                   desc: ~str,
                   default: level};

type lint_dict = hashmap<~str,lint_spec>;

/*
  Pass names should not contain a '-', as the compiler normalizes
  '-' to '_' in command-line flags
 */
fn get_lint_dict() -> lint_dict {
    let v = ~[
        (~"ctypes",
         @{lint: ctypes,
           desc: ~"proper use of core::libc types in foreign modules",
           default: warn}),

        (~"unused_imports",
         @{lint: unused_imports,
           desc: ~"imports that are never used",
           default: allow}),

        (~"while_true",
         @{lint: while_true,
           desc: ~"suggest using loop { } instead of while(true) { }",
           default: warn}),

        (~"path_statement",
         @{lint: path_statement,
           desc: ~"path statements with no effect",
           default: warn}),

        (~"unrecognized_lint",
         @{lint: unrecognized_lint,
           desc: ~"unrecognized lint attribute",
           default: warn}),

        (~"non_implicitly_copyable_typarams",
         @{lint: non_implicitly_copyable_typarams,
           desc: ~"passing non implicitly copyable types as copy type params",
           default: warn}),

        (~"vecs_implicitly_copyable",
         @{lint: vecs_implicitly_copyable,
           desc: ~"make vecs and strs not implicitly copyable \
                  (only checked at top level)",
           default: warn}),

        (~"implicit_copies",
         @{lint: implicit_copies,
           desc: ~"implicit copies of non implicitly copyable data",
           default: warn}),

        (~"deprecated_mode",
         @{lint: deprecated_mode,
           desc: ~"warn about deprecated uses of modes",
           default: allow}),

        (~"non_camel_case_types",
         @{lint: non_camel_case_types,
           desc: ~"types, variants and traits must have camel case names",
           default: allow})
    ];
    hash_from_strs(v)
}

// This is a highly not-optimal set of data structure decisions.
type lint_modes = smallintmap<level>;
type lint_mode_map = hashmap<ast::node_id, lint_modes>;

// settings_map maps node ids of items with non-default lint settings
// to their settings; default_settings contains the settings for everything
// not in the map.
type lint_settings = {
    default_settings: lint_modes,
    settings_map: lint_mode_map
};

fn mk_lint_settings() -> lint_settings {
    {default_settings: std::smallintmap::mk(),
     settings_map: int_hash()}
}

fn get_lint_level(modes: lint_modes, lint: lint) -> level {
    alt modes.find(lint as uint) {
      some(c) { c }
      none { allow }
    }
}

fn get_lint_settings_level(settings: lint_settings,
                              lint_mode: lint,
                              _expr_id: ast::node_id,
                              item_id: ast::node_id) -> level {
    alt settings.settings_map.find(item_id) {
      some(modes) { get_lint_level(modes, lint_mode) }
      none { get_lint_level(settings.default_settings, lint_mode) }
    }
}

// This is kind of unfortunate. It should be somewhere else, or we should use
// a persistent data structure...
fn clone_lint_modes(modes: lint_modes) -> lint_modes {
    std::smallintmap::smallintmap_(@{v: copy modes.v})
}

type ctxt_ = {dict: lint_dict,
              curr: lint_modes,
              is_default: bool,
              sess: session};

enum ctxt {
    ctxt_(ctxt_)
}

impl methods for ctxt {
    fn get_level(lint: lint) -> level {
        get_lint_level(self.curr, lint)
    }

    fn set_level(lint: lint, level: level) {
        if level == allow {
            self.curr.remove(lint as uint);
        } else {
            self.curr.insert(lint as uint, level);
        }
    }

    fn span_lint(level: level, span: span, msg: ~str) {
        self.sess.span_lint_level(level, span, msg);
    }

    /**
     * Merge the lints specified by any lint attributes into the
     * current lint context, call the provided function, then reset the
     * lints in effect to their previous state.
     */
    fn with_lint_attrs(attrs: ~[ast::attribute], f: fn(ctxt)) {

        let mut new_ctxt = self;
        let mut triples = ~[];

        for [allow, warn, deny, forbid].each |level| {
            let level_name = level_to_str(level);
            let metas =
                attr::attr_metas(attr::find_attrs_by_name(attrs,
                                                          level_name));
            for metas.each |meta| {
                alt meta.node {
                  ast::meta_list(_, metas) {
                    for metas.each |meta| {
                        alt meta.node {
                          ast::meta_word(lintname) {
                            vec::push(triples, (meta, level, lintname));
                          }
                          _ {
                            self.sess.span_err(
                                meta.span,
                                ~"malformed lint attribute");
                          }
                        }
                    }
                  }
                  _  {
                    self.sess.span_err(meta.span,
                                       ~"malformed lint attribute");
                  }
                }
            }
        }

        for triples.each |pair| {
            let (meta, level, lintname) = pair;
            alt self.dict.find(*lintname) {
              none {
                self.span_lint(
                    new_ctxt.get_level(unrecognized_lint),
                    meta.span,
                    fmt!{"unknown `%s` attribute: `%s`",
                         level_to_str(level), *lintname});
              }
              some(lint) {

                if new_ctxt.get_level(lint.lint) == forbid &&
                    level != forbid {
                    self.span_lint(
                        forbid,
                        meta.span,
                        fmt!{"%s(%s) overruled by outer forbid(%s)",
                             level_to_str(level),
                             *lintname, *lintname});
                }

                // we do multiple unneeded copies of the
                // map if many attributes are set, but
                // this shouldn't actually be a problem...

                let c = clone_lint_modes(new_ctxt.curr);
                new_ctxt =
                    ctxt_({is_default: false,
                           curr: c,
                           with *new_ctxt});
                new_ctxt.set_level(lint.lint, level);
              }
            }
        }
        f(new_ctxt);
    }
}


fn build_settings_item(i: @ast::item, &&cx: ctxt, v: visit::vt<ctxt>) {
    do cx.with_lint_attrs(i.attrs) |cx| {
        if !cx.is_default {
            cx.sess.lint_settings.settings_map.insert(i.id, cx.curr);
        }
        visit::visit_item(i, cx, v);
    }
}

fn build_settings_crate(sess: session::session, crate: @ast::crate) {

    let cx = ctxt_({dict: get_lint_dict(),
                    curr: std::smallintmap::mk(),
                    is_default: true,
                    sess: sess});

    // Install defaults.
    for cx.dict.each |_k, spec| { cx.set_level(spec.lint, spec.default); }

    // Install command-line options, overriding defaults.
    for sess.opts.lint_opts.each |pair| {
        let (lint,level) = pair;
        cx.set_level(lint, level);
    }

    do cx.with_lint_attrs(crate.node.attrs) |cx| {
        // Copy out the default settings
        for cx.curr.each |k, v| {
            sess.lint_settings.default_settings.insert(k, v);
        }

        let cx = ctxt_({is_default: true with *cx});

        let visit = visit::mk_vt(@{
            visit_item: build_settings_item
            with *visit::default_visitor()
        });
        visit::visit_crate(*crate, cx, visit);
    }

    sess.abort_if_errors();
}

fn check_item(i: @ast::item, cx: ty::ctxt) {
    check_item_ctypes(cx, i);
    check_item_while_true(cx, i);
    check_item_path_statement(cx, i);
    check_item_non_camel_case_types(cx, i);
}

// Take a visitor, and modify it so that it will not proceed past subitems.
// This is used to make the simple visitors used for the lint passes
// not traverse into subitems, since that is handled by the outer
// lint visitor.
fn item_stopping_visitor<E>(v: visit::vt<E>) -> visit::vt<E> {
    visit::mk_vt(@{visit_item: |_i, _e, _v| { } with **v})
}

fn check_item_while_true(cx: ty::ctxt, it: @ast::item) {
    let visit = item_stopping_visitor(visit::mk_simple_visitor(@{
        visit_expr: fn@(e: @ast::expr) {
           alt e.node {
             ast::expr_while(cond, _) {
                alt cond.node {
                    ast::expr_lit(@{node: ast::lit_bool(true),_}) {
                            cx.sess.span_lint(
                                while_true, e.id, it.id,
                                e.span,
                                ~"denote infinite loops with loop { ... }");
                    }
                    _ {}
                }
             }
             _ {}
          }
        }
        with *visit::default_simple_visitor()
    }));
    visit::visit_item(it, (), visit);
}

fn check_item_ctypes(cx: ty::ctxt, it: @ast::item) {

    fn check_foreign_fn(cx: ty::ctxt, fn_id: ast::node_id,
                       decl: ast::fn_decl) {
        let tys = vec::map(decl.inputs, |a| a.ty );
        for vec::each(vec::append_one(tys, decl.output)) |ty| {
            alt ty.node {
              ast::ty_path(_, id) {
                alt cx.def_map.get(id) {
                  ast::def_prim_ty(ast::ty_int(ast::ty_i)) {
                    cx.sess.span_lint(
                        ctypes, id, fn_id,
                        ty.span,
                        ~"found rust type `int` in foreign module, while \
                         libc::c_int or libc::c_long should be used");
                  }
                  ast::def_prim_ty(ast::ty_uint(ast::ty_u)) {
                    cx.sess.span_lint(
                        ctypes, id, fn_id,
                        ty.span,
                        ~"found rust type `uint` in foreign module, while \
                         libc::c_uint or libc::c_ulong should be used");
                  }
                  _ { }
                }
              }
              _ { }
            }
        }
    }

    alt it.node {
      ast::item_foreign_mod(nmod) if attr::foreign_abi(it.attrs) !=
      either::right(ast::foreign_abi_rust_intrinsic) {
        for nmod.items.each |ni| {
            alt ni.node {
              ast::foreign_item_fn(decl, tps) {
                check_foreign_fn(cx, it.id, decl);
              }
            }
        }
      }
      _ {/* nothing to do */ }
    }
}

fn check_item_path_statement(cx: ty::ctxt, it: @ast::item) {
    let visit = item_stopping_visitor(visit::mk_simple_visitor(@{
        visit_stmt: fn@(s: @ast::stmt) {
            alt s.node {
              ast::stmt_semi(@{id: id,
                               callee_id: _,
                               node: ast::expr_path(@path),
                               span: _}, _) {
                cx.sess.span_lint(
                    path_statement, id, it.id,
                    s.span,
                    ~"path statement with no effect");
              }
              _ {}
            }
        }
        with *visit::default_simple_visitor()
    }));
    visit::visit_item(it, (), visit);
}

fn check_item_non_camel_case_types(cx: ty::ctxt, it: @ast::item) {
    fn is_camel_case(ident: ast::ident) -> bool {
        assert ident.is_not_empty();
        char::is_uppercase(str::char_at(*ident, 0)) &&
            !ident.contains_char('_')
    }

    fn check_case(cx: ty::ctxt, ident: ast::ident,
                  expr_id: ast::node_id, item_id: ast::node_id,
                  span: span) {
        if !is_camel_case(ident) {
            cx.sess.span_lint(
                non_camel_case_types, expr_id, item_id, span,
                ~"type, variant, or trait must be camel case");
        }
    }

    alt it.node {
      ast::item_ty(*) | ast::item_class(*) |
      ast::item_trait(*) | ast::item_impl(*) {
        check_case(cx, it.ident, it.id, it.id, it.span)
      }
      ast::item_enum(variants, _) {
        check_case(cx, it.ident, it.id, it.id, it.span);
        for variants.each |variant| {
            check_case(cx, variant.node.name,
                       variant.node.id, it.id, variant.span);
        }
      }
      _ { }
    }
}

fn check_fn(tcx: ty::ctxt, fk: visit::fn_kind, decl: ast::fn_decl,
            _body: ast::blk, span: span, id: ast::node_id) {
    debug!{"lint check_fn fk=%? id=%?", fk, id};

    // don't complain about blocks, since they tend to get their modes
    // specified from the outside
    alt fk {
      visit::fk_fn_block(*) => { ret; }
      _ => {}
    }

    let fn_ty = ty::node_id_to_type(tcx, id);
    alt check ty::get(fn_ty).struct {
      ty::ty_fn(fn_ty) {
        let mut counter = 0;
        do vec::iter2(fn_ty.inputs, decl.inputs) |arg_ty, arg_ast| {
            counter += 1;
            debug!{"arg %d, ty=%s, mode=%s",
                   counter,
                   ty_to_str(tcx, arg_ty.ty),
                   mode_to_str(arg_ast.mode)};
            alt arg_ast.mode {
              ast::expl(ast::by_copy) => {
                /* always allow by-copy */
              }

              ast::expl(_) => {
                tcx.sess.span_lint(
                    deprecated_mode, id, id,
                    span,
                    fmt!{"argument %d uses an explicit mode", counter});
              }

              ast::infer(_) {
                let kind = ty::type_kind(tcx, arg_ty.ty);
                if !ty::kind_is_safe_for_default_mode(kind) {
                    tcx.sess.span_lint(
                        deprecated_mode, id, id,
                        span,
                        fmt!{"argument %d uses the default mode \
                              but shouldn't",
                             counter});
                }
              }
            }
        }
      }
    }
}

fn check_crate(tcx: ty::ctxt, crate: @ast::crate) {

    let v = visit::mk_simple_visitor(@{
        visit_item:
            |it| check_item(it, tcx),
        visit_fn:
            |fk, decl, body, span, id| check_fn(tcx, fk, decl, body,
                                                span, id),
        with *visit::default_simple_visitor()
    });
    visit::visit_crate(*crate, (), v);

    tcx.sess.abort_if_errors();
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
