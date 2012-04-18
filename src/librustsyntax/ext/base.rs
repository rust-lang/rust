import std::map::hashmap;
import parse::parser;
import diagnostic::span_handler;
import codemap::{codemap, span, expn_info, expanded_from};
import std::map::str_hash;

type syntax_expander_ =
    fn@(ext_ctxt, span, ast::mac_arg, ast::mac_body) -> @ast::expr;
type syntax_expander = {
    expander: syntax_expander_,
    span: option<span>};
type macro_def = {ident: str, ext: syntax_extension};
type macro_definer =
    fn@(ext_ctxt, span, ast::mac_arg, ast::mac_body) -> macro_def;
type item_decorator =
    fn@(ext_ctxt, span, ast::meta_item, [@ast::item]) -> [@ast::item];

enum syntax_extension {
    normal(syntax_expander),
    macro_defining(macro_definer),
    item_decorator(item_decorator),
}

// A temporary hard-coded map of methods for expanding syntax extension
// AST nodes into full ASTs
fn syntax_expander_table() -> hashmap<str, syntax_extension> {
    fn builtin(f: syntax_expander_) -> syntax_extension
        {normal({expander: f, span: none})}
    let syntax_expanders = str_hash::<syntax_extension>();
    syntax_expanders.insert("fmt", builtin(ext::fmt::expand_syntax_ext));
    syntax_expanders.insert("auto_serialize",
                            item_decorator(ext::auto_serialize::expand));
    syntax_expanders.insert("env", builtin(ext::env::expand_syntax_ext));
    syntax_expanders.insert("macro",
                            macro_defining(ext::simplext::add_new_extension));
    syntax_expanders.insert("concat_idents",
                            builtin(ext::concat_idents::expand_syntax_ext));
    syntax_expanders.insert("ident_to_str",
                            builtin(ext::ident_to_str::expand_syntax_ext));
    syntax_expanders.insert("log_syntax",
                            builtin(ext::log_syntax::expand_syntax_ext));
    syntax_expanders.insert("ast",
                            builtin(ext::qquote::expand_ast));
    ret syntax_expanders;
}

iface ext_ctxt {
    fn codemap() -> codemap;
    fn parse_sess() -> parse::parse_sess;
    fn cfg() -> ast::crate_cfg;
    fn print_backtrace();
    fn backtrace() -> expn_info;
    fn bt_push(ei: codemap::expn_info_);
    fn bt_pop();
    fn span_fatal(sp: span, msg: str) -> !;
    fn span_err(sp: span, msg: str);
    fn span_unimpl(sp: span, msg: str) -> !;
    fn span_bug(sp: span, msg: str) -> !;
    fn bug(msg: str) -> !;
    fn next_id() -> ast::node_id;
}

fn mk_ctxt(parse_sess: parse::parse_sess,
           cfg: ast::crate_cfg) -> ext_ctxt {
    type ctxt_repr = {parse_sess: parse::parse_sess,
                      cfg: ast::crate_cfg,
                      mut backtrace: expn_info};
    impl of ext_ctxt for ctxt_repr {
        fn codemap() -> codemap { self.parse_sess.cm }
        fn parse_sess() -> parse::parse_sess { self.parse_sess }
        fn cfg() -> ast::crate_cfg { self.cfg }
        fn print_backtrace() { }
        fn backtrace() -> expn_info { self.backtrace }
        fn bt_push(ei: codemap::expn_info_) {
            alt ei {
              expanded_from({call_site: cs, callie: callie}) {
                self.backtrace =
                    some(@expanded_from({
                        call_site: {lo: cs.lo, hi: cs.hi,
                                    expn_info: self.backtrace},
                        callie: callie}));
              }
            }
        }
        fn bt_pop() {
            alt self.backtrace {
              some(@expanded_from({call_site: {expn_info: prev, _}, _})) {
                self.backtrace = prev
              }
              _ { self.bug("tried to pop without a push"); }
            }
        }
        fn span_fatal(sp: span, msg: str) -> ! {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_fatal(sp, msg);
        }
        fn span_err(sp: span, msg: str) {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_err(sp, msg);
        }
        fn span_unimpl(sp: span, msg: str) -> ! {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_unimpl(sp, msg);
        }
        fn span_bug(sp: span, msg: str) -> ! {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_bug(sp, msg);
        }
        fn bug(msg: str) -> ! {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.handler().bug(msg);
        }
        fn next_id() -> ast::node_id {
            ret parse::next_node_id(self.parse_sess);
        }
    }
    let imp : ctxt_repr = {
        parse_sess: parse_sess,
        cfg: cfg,
        mut backtrace: none
    };
    ret imp as ext_ctxt
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

fn get_mac_arg(cx: ext_ctxt, sp: span, arg: ast::mac_arg) -> @ast::expr {
    alt (arg) {
      some(expr) {expr}
      none {cx.span_fatal(sp, "missing macro args")}
    }
}

fn get_mac_body(cx: ext_ctxt, sp: span, args: ast::mac_body)
    -> ast::mac_body_
{
    alt (args) {
      some(body) {body}
      none {cx.span_fatal(sp, "missing macro body")}
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
