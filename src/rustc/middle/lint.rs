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
export lookup_lint, lint_dict, get_lint_dict, check_crate;

#[doc="

A 'lint' check is a kind of miscellaneous constraint that a user _might_ want
to enforce, but might reasonably want to permit as well, on a module-by-module
basis. They contrast with static constraints enforced by other phases of the
compiler, which are generally required to hold in order to compile the program
at all.

"]

enum lint {
    ctypes,
    unused_imports,
    while_true,
    path_statement,
    old_vecs,
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
           desc: "old (deprecated) vectors and strings",
           default: ignore})

    ];
    hash_from_strs(v)
}

type ctxt = @{dict: lint_dict,
              curr: smallintmap<level>,
              tcx: ty::ctxt};

impl methods for ctxt {
    fn get_level(lint: lint) -> level {
        alt self.curr.find(lint as uint) {
          some(c) { c }
          none { ignore }
        }
    }

    fn set_level(lint: lint, level: level) {
        if level == ignore {
            self.curr.remove(lint as uint);
        } else {
            self.curr.insert(lint as uint, level);
        }
    }

    fn span_lint(level: level, span: span, msg: str) {
        alt level {
          ignore { }
          warn { self.tcx.sess.span_warn(span, msg); }
          error { self.tcx.sess.span_err(span, msg); }
        }
    }

    #[doc="
          Merge the warnings specified by any `warn(...)` attributes into the
          current lint context, call the provided function, then reset the
          warnings in effect to their previous state.
    "]
    fn with_warn_attrs(attrs: [ast::attribute], f: fn(ctxt)) {

        let mut undo = [];

        let metas = attr::attr_metas(attr::find_attrs_by_name(attrs, "warn"));
        for metas.each {|meta|
            alt meta.node {
              ast::meta_list(_, metas) {
                for metas.each {|meta|
                    alt meta.node {
                      ast::meta_word(lintname) {
                        alt lookup_lint(self.dict, lintname) {
                          none {
                            self.tcx.sess.span_err(
                                meta.span,
                                #fmt("unknown warning: '%s'", lintname));
                          }
                          some((lint, new_level)) {
                            let old_level = self.get_level(lint);
                            self.set_level(lint, new_level);
                            undo += [(lint, old_level)]
                          }
                        }
                      }
                      _ {
                        self.tcx.sess.span_err(
                            meta.span,
                            "malformed warning attribute");
                      }
                    }
                }
              }
              _ {
                self.tcx.sess.span_err(meta.span,
                                       "malformed warning attribute");
              }
            }
        }

        f(self);

        for undo.each {|pair|
            let (lint,old_level) = pair;
            self.set_level(lint, old_level);
        }
    }
}


fn lookup_lint(dict: lint_dict, s: str)
    -> option<(lint, level)> {
    let s = str::replace(s, "-", "_");
    let (name, level) = if s.starts_with("no_") {
        (s.substr(3u, s.len() - 3u), ignore)
    } else if s.starts_with("err_") {
        (s.substr(4u, s.len() - 4u), error)
    } else {
        (s, warn)
    };
    alt dict.find(name) {
      none { none }
      some(spec) { some((spec.lint, level)) }
    }
}


// FIXME: Copied from driver.rs, to work around a bug(#1566)
fn time(do_it: bool, what: str, thunk: fn()) {
    if !do_it{ ret thunk(); }
    let start = std::time::precise_time_s();
    thunk();
    let end = std::time::precise_time_s();
    io::stdout().write_str(#fmt("time: %3.3f s\t%s\n",
                                end - start, what));
}

fn check_item(cx: ctxt, i: @ast::item) {
    cx.with_warn_attrs(i.attrs) {|cx|
        for cx.curr.each {|lint, level|
            alt int_to_lint(lint as int) {
              ctypes { check_item_ctypes(cx, level, i); }
              unused_imports { check_item_unused_imports(cx, level, i); }
              while_true { check_item_while_true(cx, level, i); }
              path_statement { check_item_path_statement(cx, level, i); }
              old_vecs { check_item_old_vecs(cx, level, i); }
            }
        }
    }
}

fn check_item_while_true(cx: ctxt, level: level, it: @ast::item) {
    let visit = visit::mk_simple_visitor(@{
        visit_expr: fn@(e: @ast::expr) {
           alt e.node {
             ast::expr_while(cond, _) {
                alt cond.node {
                    ast::expr_lit(@{node: ast::lit_bool(true),_}) {
                            cx.span_lint(
                              level, e.span,
                              "denote infinite loops with loop { ... }");
                    }
                    _ {}
                }
             }
             _ {}
          }
        }
        with *visit::default_simple_visitor()
    });
    visit::visit_item(it, (), visit);
}

fn check_item_unused_imports(_cx: ctxt, _level: level, _it: @ast::item) {
    // FIXME: Don't know how to check this in lint yet, it's currently being
    // done over in resolve. When resolve is rewritten, do it here instead.
}

fn check_item_ctypes(cx: ctxt, level: level, it: @ast::item) {

    fn check_native_fn(cx: ctxt, level: level, decl: ast::fn_decl) {
        let tys = vec::map(decl.inputs) {|a| a.ty };
        for vec::each(tys + [decl.output]) {|ty|
            alt ty.node {
              ast::ty_path(_, id) {
                alt cx.tcx.def_map.get(id) {
                  ast::def_prim_ty(ast::ty_int(ast::ty_i)) {
                    cx.span_lint(
                        level, ty.span,
                        "found rust type `int` in native module, while \
                         libc::c_int or libc::c_long should be used");
                  }
                  ast::def_prim_ty(ast::ty_uint(ast::ty_u)) {
                    cx.span_lint(
                        level, ty.span,
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
                check_native_fn(cx, level, decl);
              }
            }
        }
      }
      _ {/* nothing to do */ }
    }
}

fn check_item_path_statement(cx: ctxt, level: level, it: @ast::item) {
    let visit = visit::mk_simple_visitor(@{
        visit_stmt: fn@(s: @ast::stmt) {
            alt s.node {
              ast::stmt_semi(@{id: _,
                               node: ast::expr_path(@path),
                               span: _}, _) {
                cx.span_lint(
                    level, s.span,
                    "path statement with no effect");
              }
              _ {}
            }
        }
        with *visit::default_simple_visitor()
    });
    visit::visit_item(it, (), visit);
}

fn check_item_old_vecs(cx: ctxt, level: level, it: @ast::item) {
    let uses_vstore = int_hash();

    let visit = visit::mk_simple_visitor(@{

        visit_expr:fn@(e: @ast::expr) {
            alt e.node {
              ast::expr_vec(_, _) |
              ast::expr_lit(@{node: ast::lit_str(_), span:_})
              if ! uses_vstore.contains_key(e.id) {
                cx.span_lint(level, e.span, "deprecated vec/str expr");
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
                cx.span_lint(level, t.span, "deprecated vec type");
              }

              ast::ty_path(@{span: _, global: _, idents: ids,
                             rp: none, types: _}, _)
              if ids == ["str"] && (! uses_vstore.contains_key(t.id)) {
                cx.span_lint(level, t.span, "deprecated str type");
              }

              ast::ty_vstore(inner, _) {
                uses_vstore.insert(inner.id, true);
              }
              _ { }
            }
        }

        with *visit::default_simple_visitor()
    });
    visit::visit_item(it, (), visit);
}


fn check_crate(tcx: ty::ctxt, crate: @ast::crate,
               lint_opts: [(lint, level)], time_pass: bool) {

    fn hash_lint(&&lint: lint) -> uint { lint as uint }
    fn eq_lint(&&a: lint, &&b: lint) -> bool { a == b }

    let cx = @{dict: get_lint_dict(),
               curr: std::smallintmap::mk(),
               tcx: tcx};

    // Install defaults.
    for cx.dict.each {|_k, spec| cx.set_level(spec.lint, spec.default); }

    // Install command-line options, overriding defaults.
    for lint_opts.each {|pair|
        let (lint,level) = pair;
        cx.set_level(lint, level);
    }

    time(time_pass, "lint checking") {||
        cx.with_warn_attrs(crate.node.attrs) {|cx|
            let visit = visit::mk_simple_visitor(@{
                visit_item: fn@(i: @ast::item) { check_item(cx, i); }
                with *visit::default_simple_visitor()
            });
            visit::visit_crate(*crate, (), visit);
        }
    }

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
