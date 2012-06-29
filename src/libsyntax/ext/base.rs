import std::map::hashmap;
import parse::parser;
import diagnostic::span_handler;
import codemap::{codemap, span, expn_info, expanded_from};
import std::map::str_hash;

type syntax_expander_ =
    fn@(ext_ctxt, span, ast::mac_arg, ast::mac_body) -> @ast::expr;
type syntax_expander = {expander: syntax_expander_, span: option<span>};

type macro_def = {ident: ast::ident, ext: syntax_extension};
type macro_definer =
    fn@(ext_ctxt, span, ast::mac_arg, ast::mac_body) -> macro_def;
type item_decorator =
    fn@(ext_ctxt, span, ast::meta_item, ~[@ast::item]) -> ~[@ast::item];

type syntax_expander_tt = {expander: syntax_expander_tt_, span: option<span>};
type syntax_expander_tt_ = fn@(ext_ctxt, span, ast::token_tree) -> @ast::expr;

enum syntax_extension {
    normal(syntax_expander),
    macro_defining(macro_definer),
    item_decorator(item_decorator),

    normal_tt(syntax_expander_tt)
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
    syntax_expanders.insert("line",
                            builtin(ext::source_util::expand_line));
    syntax_expanders.insert("col",
                            builtin(ext::source_util::expand_col));
    syntax_expanders.insert("file",
                            builtin(ext::source_util::expand_file));
    syntax_expanders.insert("stringify",
                            builtin(ext::source_util::expand_stringify));
    syntax_expanders.insert("include",
                            builtin(ext::source_util::expand_include));
    syntax_expanders.insert("include_str",
                            builtin(ext::source_util::expand_include_str));
    syntax_expanders.insert("include_bin",
                            builtin(ext::source_util::expand_include_bin));
    syntax_expanders.insert("mod",
                            builtin(ext::source_util::expand_mod));
    ret syntax_expanders;
}

iface ext_ctxt {
    fn codemap() -> codemap;
    fn parse_sess() -> parse::parse_sess;
    fn cfg() -> ast::crate_cfg;
    fn print_backtrace();
    fn backtrace() -> expn_info;
    fn mod_push(mod_name: ast::ident);
    fn mod_pop();
    fn mod_path() -> ~[ast::ident];
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
                      mut backtrace: expn_info,
                      mut mod_path: ~[ast::ident]};
    impl of ext_ctxt for ctxt_repr {
        fn codemap() -> codemap { self.parse_sess.cm }
        fn parse_sess() -> parse::parse_sess { self.parse_sess }
        fn cfg() -> ast::crate_cfg { self.cfg }
        fn print_backtrace() { }
        fn backtrace() -> expn_info { self.backtrace }
        fn mod_push(i: ast::ident) { vec::push(self.mod_path, i); }
        fn mod_pop() { vec::pop(self.mod_path); }
        fn mod_path() -> ~[ast::ident] { ret self.mod_path; }
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
        mut backtrace: none,
        mut mod_path: ~[]
    };
    ret imp as ext_ctxt
}

fn expr_to_str(cx: ext_ctxt, expr: @ast::expr, error: str) -> str {
    alt expr.node {
      ast::expr_lit(l) {
        alt l.node {
          ast::lit_str(s) { ret *s; }
          _ { cx.span_fatal(l.span, error); }
        }
      }
      _ { cx.span_fatal(expr.span, error); }
    }
}

fn expr_to_ident(cx: ext_ctxt, expr: @ast::expr, error: str) -> ast::ident {
    alt expr.node {
      ast::expr_path(p) {
        if vec::len(p.types) > 0u || vec::len(p.idents) != 1u {
            cx.span_fatal(expr.span, error);
        } else { ret p.idents[0]; }
      }
      _ { cx.span_fatal(expr.span, error); }
    }
}

fn get_mac_args_no_max(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
                       min: uint, name: str) -> ~[@ast::expr] {
    ret get_mac_args(cx, sp, arg, min, none, name);
}

fn get_mac_args(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
                min: uint, max: option<uint>, name: str) -> ~[@ast::expr] {
    alt arg {
      some(expr) {
        alt expr.node {
          ast::expr_vec(elts, _) {
            let elts_len = vec::len(elts);
            alt max {
              some(max) if ! (min <= elts_len && elts_len <= max) {
                cx.span_fatal(sp,
                              #fmt["#%s takes between %u and %u arguments.",
                                   name, min, max]);
              }
              none if ! (min <= elts_len) {
                cx.span_fatal(sp, #fmt["#%s needs at least %u arguments.",
                                       name, min]);
              }
              _ { ret elts; /* we're good */}
            }
          }
          _ {
            cx.span_fatal(sp, #fmt["#%s: malformed invocation", name])
          }
        }
      }
      none {cx.span_fatal(sp, #fmt["#%s: missing arguments", name])}
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
