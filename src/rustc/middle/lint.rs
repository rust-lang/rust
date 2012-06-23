import driver::session;
import driver::session::session;
import middle::ty;
import syntax::{ast, visit};
import syntax::attr;
import syntax::codemap::span;
import std::map::{map,hashmap,int_hash,hash_from_strs};
import std::smallintmap::{map,smallintmap};
import io::writer_util;
import syntax::print::pprust::expr_to_str;
export lint, ctypes, unused_imports;
export level, ignore, warn, error;
export lookup_lint, lint_dict, get_lint_dict;
export get_warning_level, get_warning_settings_level;
export check_crate, build_settings_crate, mk_warning_settings;
export warning_settings;

#[doc="

A 'lint' check is a kind of miscellaneous constraint that a user _might_ want
to enforce, but might reasonably want to permit as well, on a module-by-module
basis. They contrast with static constraints enforced by other phases of the
compiler, which are generally required to hold in order to compile the program
at all.

We also build up a table containing information about lint settings, in order
to allow other passes to take advantage of the warning attribute
infrastructure. To save space, the table is keyed by the id of /items/, not of
every expression. When an item has the default settings, the entry will be
omitted. If we start allowing warn attributes on expressions, we will start
having entries for expressions that do not share their enclosing items
settings.

This module then, exports two passes: one that populates the warning settings
table in the session and is run early in the compile process, and one that
does a variety of lint checks, and is run late in the compile process.

"]

enum lint {
    ctypes,
    unused_imports,
    while_true,
    path_statement,
    old_vecs,
    unrecognized_warning,
    non_implicitly_copyable_typarams,
    vecs_not_implicitly_copyable,
    implicit_copies,
    old_strs,
}

// This is pretty unfortunate. We really want some sort of "deriving Enum"
// type of thing.
fn int_to_lint(i: int) -> lint {
    alt check i {
      0 { ctypes }
      1 { unused_imports }
      2 { while_true }
      3 { path_statement }
      4 { old_vecs }
      5 { unrecognized_warning }
      6 { non_implicitly_copyable_typarams }
      7 { vecs_not_implicitly_copyable }
      8 { implicit_copies }
      9 { old_strs }
    }
}

enum level {
    ignore, warn, error
}

type lint_spec = @{lint: lint,
                   desc: str,
                   default: level};

type lint_dict = hashmap<str,lint_spec>;

/*
  Pass names should not contain a '-', as the compiler normalizes
  '-' to '_' in command-line flags
 */
fn get_lint_dict() -> lint_dict {
    let v = [
        ("ctypes",
         @{lint: ctypes,
           desc: "proper use of core::libc types in native modules",
           default: warn}),

        ("unused_imports",
         @{lint: unused_imports,
           desc: "imports that are never used",
           default: ignore}),

        ("while_true",
         @{lint: while_true,
           desc: "suggest using loop { } instead of while(true) { }",
           default: warn}),

        ("path_statement",
         @{lint: path_statement,
           desc: "path statements with no effect",
           default: warn}),

        ("old_vecs",
         @{lint: old_vecs,
           desc: "old (deprecated) vectors",
           default: warn}),

        ("old_strs",
         @{lint: old_strs,
           desc: "old (deprecated) strings",
           default: ignore}),

        ("unrecognized_warning",
         @{lint: unrecognized_warning,
           desc: "unrecognized warning attribute",
           default: warn}),

        ("non_implicitly_copyable_typarams",
         @{lint: non_implicitly_copyable_typarams,
           desc: "passing non implicitly copyable types as copy type params",
           default: warn}),

        ("vecs_not_implicitly_copyable",
         @{lint: vecs_not_implicitly_copyable,
           desc: "make vecs and strs not implicitly copyable\
                  ('err' is ignored; only checked at top level",
           default: warn}),

        ("implicit_copies",
         @{lint: implicit_copies,
           desc: "implicit copies of non implicitly copyable data",
           default: warn})

    ];
    hash_from_strs(v)
}

// This is a highly not-optimal set of data structure decisions.
type lint_modes = smallintmap<level>;
type lint_mode_map = hashmap<ast::node_id, lint_modes>;

// settings_map maps node ids of items with non-default warning settings
// to their settings; default_settings contains the settings for everything
// not in the map.
type warning_settings = {
    default_settings: lint_modes,
    settings_map: lint_mode_map
};

fn mk_warning_settings() -> warning_settings {
    {default_settings: std::smallintmap::mk(),
     settings_map: int_hash()}
}

fn get_warning_level(modes: lint_modes, lint: lint) -> level {
    alt modes.find(lint as uint) {
      some(c) { c }
      none { ignore }
    }
}

fn get_warning_settings_level(settings: warning_settings,
                              lint_mode: lint,
                              _expr_id: ast::node_id,
                              item_id: ast::node_id) -> level {
    alt settings.settings_map.find(item_id) {
      some(modes) { get_warning_level(modes, lint_mode) }
      none { get_warning_level(settings.default_settings, lint_mode) }
    }
}

// This is kind of unfortunate. It should be somewhere else, or we should use
// a persistent data structure...
fn clone_lint_modes(modes: lint_modes) -> lint_modes {
    @{v: copy modes.v}
}

type ctxt = {dict: lint_dict,
             curr: lint_modes,
             is_default: bool,
             sess: session};


impl methods for ctxt {
    fn get_level(lint: lint) -> level {
        get_warning_level(self.curr, lint)
    }

    fn set_level(lint: lint, level: level) {
        if level == ignore {
            self.curr.remove(lint as uint);
        } else {
            self.curr.insert(lint as uint, level);
        }
    }

    fn span_lint(level: level, span: span, msg: str) {
        self.sess.span_lint_level(level, span, msg);
    }

    #[doc="
          Merge the warnings specified by any `warn(...)` attributes into the
          current lint context, call the provided function, then reset the
          warnings in effect to their previous state.
    "]
    fn with_warn_attrs(attrs: [ast::attribute], f: fn(ctxt)) {

        let mut new_ctxt = self;

        let metas = attr::attr_metas(attr::find_attrs_by_name(attrs, "warn"));
        for metas.each {|meta|
            alt meta.node {
              ast::meta_list(_, metas) {
                for metas.each {|meta|
                    alt meta.node {
                      ast::meta_word(lintname) {
                        alt lookup_lint(self.dict, *lintname) {
                          (name, none) {
                            self.span_lint(
                                self.get_level(unrecognized_warning),
                                meta.span,
                                #fmt("unknown warning: '%s'", name));
                          }
                          (_, some((lint, new_level))) {
                            // we do multiple unneeded copies of the map
                            // if many attributes are set, but this shouldn't
                            // actually be a problem...
                            new_ctxt = {is_default: false,
                                        curr: clone_lint_modes(new_ctxt.curr)
                                        with new_ctxt};
                            new_ctxt.set_level(lint, new_level);
                          }
                        }
                      }
                      _ {
                        self.sess.span_err(
                            meta.span,
                            "malformed warning attribute");
                      }
                    }
                }
              }
              _ {
                self.sess.span_err(meta.span,
                                   "malformed warning attribute");
              }
            }
        }

        f(new_ctxt);
    }
}


fn lookup_lint(dict: lint_dict, s: str)
    -> (str, option<(lint, level)>) {
    let s = str::replace(s, "-", "_");
    let (name, level) = if s.starts_with("no_") {
        (s.substr(3u, s.len() - 3u), ignore)
    } else if s.starts_with("err_") {
        (s.substr(4u, s.len() - 4u), error)
    } else {
        (s, warn)
    };
    (name,
     alt dict.find(name) {
         none { none }
         some(spec) { some((spec.lint, level)) }
     })
}

fn build_settings_item(i: @ast::item, &&cx: ctxt, v: visit::vt<ctxt>) {
    cx.with_warn_attrs(i.attrs) {|cx|
        if !cx.is_default {
            cx.sess.warning_settings.settings_map.insert(i.id, cx.curr);
        }
        visit::visit_item(i, cx, v);
    }
}

fn build_settings_crate(sess: session::session, crate: @ast::crate) {

    let cx = {dict: get_lint_dict(),
              curr: std::smallintmap::mk(),
              is_default: true,
              sess: sess};

    // Install defaults.
    for cx.dict.each {|_k, spec| cx.set_level(spec.lint, spec.default); }

    // Install command-line options, overriding defaults.
    for sess.opts.lint_opts.each {|pair|
        let (lint,level) = pair;
        cx.set_level(lint, level);
    }

    cx.with_warn_attrs(crate.node.attrs) {|cx|
        // Copy out the default settings
        for cx.curr.each {|k, v|
            sess.warning_settings.default_settings.insert(k, v);
        }

        let cx = {is_default: true with cx};

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
    check_item_old_vecs(cx, i);
}

// Take a visitor, and modify it so that it will not proceed past subitems.
// This is used to make the simple visitors used for the lint passes
// not traverse into subitems, since that is handled by the outer
// lint visitor.
fn item_stopping_visitor<E>(v: visit::vt<E>) -> visit::vt<E> {
    visit::mk_vt(@{visit_item: {|_i, _e, _v| } with **v})
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
                                "denote infinite loops with loop { ... }");
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

    fn check_native_fn(cx: ty::ctxt, fn_id: ast::node_id,
                       decl: ast::fn_decl) {
        let tys = vec::map(decl.inputs) {|a| a.ty };
        for vec::each(tys + [decl.output]) {|ty|
            alt ty.node {
              ast::ty_path(_, id) {
                alt cx.def_map.get(id) {
                  ast::def_prim_ty(ast::ty_int(ast::ty_i)) {
                    cx.sess.span_lint(
                        ctypes, id, fn_id,
                        ty.span,
                        "found rust type `int` in native module, while \
                         libc::c_int or libc::c_long should be used");
                  }
                  ast::def_prim_ty(ast::ty_uint(ast::ty_u)) {
                    cx.sess.span_lint(
                        ctypes, id, fn_id,
                        ty.span,
                        "found rust type `uint` in native module, while \
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
      ast::item_native_mod(nmod) if attr::native_abi(it.attrs) !=
      either::right(ast::native_abi_rust_intrinsic) {
        for nmod.items.each {|ni|
            alt ni.node {
              ast::native_item_fn(decl, tps) {
                check_native_fn(cx, it.id, decl);
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
                               node: ast::expr_path(@path),
                               span: _}, _) {
                cx.sess.span_lint(
                    path_statement, id, it.id,
                    s.span,
                    "path statement with no effect");
              }
              _ {}
            }
        }
        with *visit::default_simple_visitor()
    }));
    visit::visit_item(it, (), visit);
}

fn check_item_old_vecs(cx: ty::ctxt, it: @ast::item) {
    let uses_vstore = int_hash();

    let visit = item_stopping_visitor(visit::mk_simple_visitor(@{

        visit_expr:fn@(e: @ast::expr) {
            alt e.node {
              ast::expr_vec(_, _)
              if ! uses_vstore.contains_key(e.id) {
                cx.sess.span_lint(
                    old_vecs, e.id, it.id,
                    e.span, "deprecated vec expr");
              }
              ast::expr_lit(@{node: ast::lit_str(_), span:_})
              if ! uses_vstore.contains_key(e.id) {
                cx.sess.span_lint(
                    old_strs, e.id, it.id,
                    e.span, "deprecated str expr");
              }

              ast::expr_vstore(@inner, _) {
                uses_vstore.insert(inner.id, true);
              }
              _ { }
            }
        },

        visit_ty: fn@(t: @ast::ty) {
            alt t.node {
              ast::ty_vec(_)
              if ! uses_vstore.contains_key(t.id) {
                cx.sess.span_lint(
                    old_vecs, t.id, it.id,
                    t.span, "deprecated vec type");
              }
              ast::ty_path(@{span: _, global: _, idents: ids,
                             rp: none, types: _}, _)
              if ids == [@"str"] && (! uses_vstore.contains_key(t.id)) {
                cx.sess.span_lint(
                    old_strs, t.id, it.id,
                    t.span, "deprecated str type");
              }
              ast::ty_vstore(inner, _) {
                uses_vstore.insert(inner.id, true);
              }
              _ { }
            }
        }

        with *visit::default_simple_visitor()
    }));
    visit::visit_item(it, (), visit);
}

fn check_crate(tcx: ty::ctxt, crate: @ast::crate) {

    let v = visit::mk_simple_visitor(@{
        visit_item: fn@(it: @ast::item) { check_item(it, tcx); }
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
