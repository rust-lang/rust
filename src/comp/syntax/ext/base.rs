import core::{vec, option};
import std::map::hashmap;
import driver::session::session;
import codemap::span;
import std::map::new_str_hash;
import codemap;

type syntax_expander =
    fn@(ext_ctxt, span, @ast::expr, option::t<str>) -> @ast::expr;
type macro_def = {ident: str, ext: syntax_extension};
type macro_definer =
    fn@(ext_ctxt, span, @ast::expr, option::t<str>) -> macro_def;

enum syntax_extension {
    normal(syntax_expander),
    macro_defining(macro_definer),
}

// A temporary hard-coded map of methods for expanding syntax extension
// AST nodes into full ASTs
fn syntax_expander_table() -> hashmap<str, syntax_extension> {
    let syntax_expanders = new_str_hash::<syntax_extension>();
    syntax_expanders.insert("fmt", normal(ext::fmt::expand_syntax_ext));
    syntax_expanders.insert("env", normal(ext::env::expand_syntax_ext));
    syntax_expanders.insert("macro",
                            macro_defining(ext::simplext::add_new_extension));
    syntax_expanders.insert("concat_idents",
                            normal(ext::concat_idents::expand_syntax_ext));
    syntax_expanders.insert("ident_to_str",
                            normal(ext::ident_to_str::expand_syntax_ext));
    syntax_expanders.insert("log_syntax",
                            normal(ext::log_syntax::expand_syntax_ext));
    ret syntax_expanders;
}

iface ext_ctxt {
    fn session() -> session;
    fn print_backtrace();
    fn backtrace() -> codemap::opt_span;
    fn bt_push(sp: span);
    fn bt_pop();
    fn span_fatal(sp: span, msg: str) -> !;
    fn span_err(sp: span, msg: str);
    fn span_unimpl(sp: span, msg: str) -> !;
    fn span_bug(sp: span, msg: str) -> !;
    fn bug(msg: str) -> !;
    fn next_id() -> ast::node_id;
}

fn mk_ctxt(sess: session) -> ext_ctxt {
    type ctxt_repr = {sess: session,
                      mutable backtrace: codemap::opt_span};
    impl of ext_ctxt for ctxt_repr {
        fn session() -> session { self.sess }
        fn print_backtrace() { }
        fn backtrace() -> codemap::opt_span { self.backtrace }
        fn bt_push(sp: span) {
            self.backtrace = codemap::os_some(
                @{lo: sp.lo, hi: sp.hi, expanded_from: self.backtrace});
        }
        fn bt_pop() {
            alt self.backtrace {
              codemap::os_some(@{expanded_from: pre, _}) {
                let tmp = pre;
                self.backtrace = tmp;
              }
              _ { self.bug("tried to pop without a push"); }
            }
        }
        fn span_fatal(sp: span, msg: str) -> ! {
            self.print_backtrace();
            self.sess.span_fatal(sp, msg);
        }
        fn span_err(sp: span, msg: str) {
            self.print_backtrace();
            self.sess.span_err(sp, msg);
        }
        fn span_unimpl(sp: span, msg: str) -> ! {
            self.print_backtrace();
            self.sess.span_unimpl(sp, msg);
        }
        fn span_bug(sp: span, msg: str) -> ! {
            self.print_backtrace();
            self.sess.span_bug(sp, msg);
        }
        fn bug(msg: str) -> ! { self.print_backtrace(); self.sess.bug(msg); }
        fn next_id() -> ast::node_id { ret self.sess.next_node_id(); }
    }
    {sess: sess, mutable backtrace: codemap::os_none} as ext_ctxt
}

fn expr_to_str(cx: ext_ctxt, expr: @ast::expr, error: str) -> str {
    alt expr.node {
      ast::expr_lit(l) {
        alt l.node {
          ast::lit_str(s) { ret s; }
          _ { cx.span_fatal(l.span, error); }
        }
      }
      _ { cx.span_fatal(expr.span, error); }
    }
}

fn expr_to_ident(cx: ext_ctxt, expr: @ast::expr, error: str) -> ast::ident {
    alt expr.node {
      ast::expr_path(p) {
        if vec::len(p.node.types) > 0u || vec::len(p.node.idents) != 1u {
            cx.span_fatal(expr.span, error);
        } else { ret p.node.idents[0]; }
      }
      _ { cx.span_fatal(expr.span, error); }
    }
}

fn make_new_lit(cx: ext_ctxt, sp: codemap::span, lit: ast::lit_) ->
   @ast::expr {
    let sp_lit = @{node: lit, span: sp};
    ret @{id: cx.next_id(), node: ast::expr_lit(sp_lit), span: sp};
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
