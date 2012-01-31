import driver::session::session;
import middle::ty::ctxt;
import syntax::{ast, visit};
import front::attr;
import std::io;
import io::writer_util;

enum option {
    ctypes,
}

impl opt_ for option {
    fn desc() -> str {
        "lint: " + alt self {
          ctypes { "ctypes usage checking" }
        }
    }
    fn run(tcx: ty::ctxt, crate: @ast::crate, time_pass: bool) {
        let checker = alt self {
          ctypes {
            bind check_ctypes(tcx, crate)
          }
        };
        time(time_pass, self.desc(), checker);
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

// Merge lint options specified by crate attributes and rustc command
// line. Precedence: cmdline > attribute > default
fn merge_opts(attrs: [ast::attribute], cmd_opts: [(option, bool)]) ->
    [(option, bool)] {
    fn str_to_option(name: str) -> (option, bool) {
        ret alt name {
          "ctypes" { (ctypes, true) }
          "no_ctypes" { (ctypes, false) }
        }
    }

    fn meta_to_option(meta: @ast::meta_item) -> (option, bool) {
        ret alt meta.node {
          ast::meta_word(name) {
            str_to_option(name)
          }
          _ { fail "meta_to_option: meta_list contains a non-meta-word"; }
        };
    }

    fn default() -> [(option, bool)] {
        [(ctypes, true)]
    }

    fn contains(xs: [(option, bool)], x: option) -> bool {
        for (o, _) in xs {
            if o == x { ret true; }
        }
        ret false;
    }

    let result = cmd_opts;

    let lint_metas =
        attr::attr_metas(attr::find_attrs_by_name(attrs, "lint"));

    vec::iter(lint_metas) {|mi|
        alt mi.node {
          ast::meta_list(_, list) {
            vec::iter(list) {|e|
                let (o, v) = meta_to_option(e);
                if !contains(cmd_opts, o) {
                    result += [(o, v)];
                }
            }
          }
          _ { }
        }
    };

    for (o, v) in default() {
        if !contains(result, o) {
            result += [(o, v)];
        }
    }

    ret result;
}

fn check_ctypes(tcx: ty::ctxt, crate: @ast::crate) {
    fn check_native_fn(tcx: ty::ctxt, decl: ast::fn_decl) {
        let tys = vec::map(decl.inputs) {|a| a.ty };
        for ty in (tys + [decl.output]) {
            alt ty.node {
              ast::ty_int(ast::ty_i) {
                tcx.sess.span_warn(
                    ty.span,
                    "found rust type `int` in native module, while \
                     ctypes::c_int or ctypes::long should be used");
              }
              ast::ty_uint(ast::ty_u) {
                tcx.sess.span_warn(
                    ty.span,
                    "found rust type `uint` in native module, while \
                     ctypes::c_uint or ctypes::ulong should be used");
              }
              _ { }
            }
        }
    }

    fn check_item(tcx: ty::ctxt, it: @ast::item) {
        alt it.node {
          ast::item_native_mod(nmod) {
            for ni in nmod.items {
                alt ni.node {
                  ast::native_item_fn(decl, tps) {
                    check_native_fn(tcx, decl);
                  }
                  _ { }
                }
            }
          }
          _ {/* nothing to do */ }
        }
    }

    let visit = visit::mk_simple_visitor(@{
        visit_item: bind check_item(tcx, _)
        with *visit::default_simple_visitor()
    });
    visit::visit_crate(*crate, (), visit);
}

fn check_crate(tcx: ty::ctxt, crate: @ast::crate,
               opts: [(option, bool)], time: bool) {
    let lint_opts = lint::merge_opts(crate.node.attrs, opts);
    for (lopt, switch) in lint_opts {
        if switch == true {
            lopt.run(tcx, crate, time);
        }
    }
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
