import driver::session;

import option::{none, some};

import syntax::ast::{crate, expr_, mac_invoc,
                     mac_qq, mac_aq, mac_var};
import syntax::fold::*;
import syntax::visit::*;
import syntax::ext::base::*;
import syntax::ext::build::*;
import syntax::parse::parser;
import syntax::parse::parser::{parser, parse_from_source_str};

import syntax::print::*;
import std::io::*;

import codemap::span;

type aq_ctxt = @{lo: uint,
                 mutable gather: [{lo: uint, hi: uint,
                                   e: @ast::expr, constr: str}]};
enum fragment {
    from_expr(@ast::expr),
    from_ty(@ast::ty)
}

iface qq_helper {
    fn span() -> span;
    fn visit(aq_ctxt, vt<aq_ctxt>);
    fn extract_mac() -> option<ast::mac_>;
    fn mk_parse_fn(ext_ctxt,span) -> @ast::expr;
}
impl of qq_helper for @ast::expr {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_expr(self, cx, v);}
    fn extract_mac() -> option<ast::mac_> {
        alt (self.node) {
          ast::expr_mac({node: mac, _}) {some(mac)}
          _ {none}
        }
    }
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp, ["syntax", "parse", "parser", "parse_expr"])
    }
}
impl of qq_helper for @ast::ty {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_ty(self, cx, v);}
    fn extract_mac() -> option<ast::mac_> {
        alt (self.node) {
          ast::ty_mac({node: mac, _}) {some(mac)}
          _ {none}
        }
    }
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp, ["syntax", "ext", "qquote", "parse_ty"])
    }
}
impl of qq_helper for @ast::item {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_item(self, cx, v);}
    fn extract_mac() -> option<ast::mac_> {fail}
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp, ["syntax", "ext", "qquote", "parse_item"])
    }
}
impl of qq_helper for @ast::stmt {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_stmt(self, cx, v);}
    fn extract_mac() -> option<ast::mac_> {fail}
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp, ["syntax", "ext", "qquote", "parse_stmt"])
    }
}
impl of qq_helper for @ast::pat {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_pat(self, cx, v);}
    fn extract_mac() -> option<ast::mac_> {fail}
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp, ["syntax", "parse", "parser", "parse_pat"])
    }
}

fn gather_anti_quotes<N: qq_helper>(lo: uint, node: N) -> aq_ctxt
{
    let v = @{visit_expr: visit_aq_expr,
              visit_ty: visit_aq_ty
              with *default_visitor()};
    let cx = @{lo:lo, mutable gather: []};
    node.visit(cx, mk_vt(v));
    ret cx;
}

fn visit_aq<T:qq_helper>(node: T, constr: str, &&cx: aq_ctxt, v: vt<aq_ctxt>)
{
    alt (node.extract_mac()) {
      some(mac_aq(sp, e)) {
        cx.gather += [{lo: sp.lo - cx.lo, hi: sp.hi - cx.lo,
                       e: e, constr: constr}];
      }
      _ {node.visit(cx, v);}
    }
}
// FIXME: these are only here because I (kevina) couldn't figure out how to
// get bind to work in gather_anti_quotes
fn visit_aq_expr(node: @ast::expr, &&cx: aq_ctxt, v: vt<aq_ctxt>) {
    visit_aq(node,"from_expr",cx,v);
}
fn visit_aq_ty(node: @ast::ty, &&cx: aq_ctxt, v: vt<aq_ctxt>) {
    visit_aq(node,"from_ty",cx,v);
}

fn is_space(c: char) -> bool {
    syntax::parse::lexer::is_whitespace(c)
}

fn expand_ast(ecx: ext_ctxt, _sp: span,
              arg: ast::mac_arg, body: ast::mac_body)
    -> @ast::expr
{
    let what = "expr";
    option::may(arg) {|arg|
        let args: [@ast::expr] =
            alt arg.node {
              ast::expr_vec(elts, _) { elts }
              _ {
                ecx.span_fatal
                    (_sp, "#ast requires arguments of the form `[...]`.")
              }
            };
        if vec::len::<@ast::expr>(args) != 1u {
            ecx.span_fatal(_sp, "#ast requires exactly one arg");
        }
        alt (args[0].node) {
          ast::expr_path(@{node: {idents: id, _},_}) if vec::len(id) == 1u
              {what = id[0]}
          _ {ecx.span_fatal(args[0].span, "expected an identifier");}
        }
    }
    let body = get_mac_body(ecx,_sp,body);
    fn finish<T: qq_helper>(ecx: ext_ctxt, body: ast::mac_body_,
                            f: fn (p: parser) -> T)
        -> @ast::expr
    {
        let cm = ecx.session().parse_sess.cm;
        let str = @codemap::span_to_snippet(body.span, cm);
        let (fname, ss) = codemap::get_substr_info
            (cm, body.span.lo, body.span.hi);
        let node = parse_from_source_str
            (f, fname, some(ss), str,
             ecx.session().opts.cfg, ecx.session().parse_sess);
        ret expand_qquote(ecx, node.span(), some(*str), node);
    }

    ret alt what {
      "expr" {finish(ecx, body, parser::parse_expr)}
      "ty" {finish(ecx, body, parse_ty)}
      "item" {finish(ecx, body, parse_item)}
      "stmt" {finish(ecx, body, parse_stmt)}
      "pat" {finish(ecx, body, parser::parse_pat)}
      _ {ecx.span_fatal(_sp, "unsupported ast type")}
    };
}

fn parse_ty(p: parser) -> @ast::ty {
    parser::parse_ty(p, false)
}

fn parse_stmt(p: parser) -> @ast::stmt {
    parser::parse_stmt(p, [])
}

fn parse_item(p: parser) -> @ast::item {
    alt (parser::parse_item(p, [])) {
      some(item) {item}
      none {fail; /* FIXME: Error message, somehow */}
    }
}

fn expand_qquote<N: qq_helper>
    (ecx: ext_ctxt, sp: span, maybe_str: option::t<str>, node: N)
    -> @ast::expr
{
    let str = alt(maybe_str) {
      some(s) {s}
      none {codemap::span_to_snippet(sp, ecx.session().parse_sess.cm)}
    };
    let qcx = gather_anti_quotes(sp.lo, node);
    let cx = qcx;
    let prev = 0u;
    for {lo: lo, _} in cx.gather {
        assert lo > prev;
        prev = lo;
    }
    let str2 = "";
    enum state {active, skip(uint), blank};
    let state = active;
    let i = 0u, j = 0u;
    let g_len = vec::len(cx.gather);
    str::chars_iter(str) {|ch|
        if (j < g_len && i == cx.gather[j].lo) {
            assert ch == '$';
            let repl = #fmt("$%u ", j);
            state = skip(str::char_len(repl));
            str2 += repl;
        }
        alt state {
          active {str::push_char(str2, ch);}
          skip(1u) {state = blank;}
          skip(sk) {state = skip (sk-1u);}
          blank if is_space(ch) {str::push_char(str2, ch);}
          blank {str::push_char(str2, ' ');}
        }
        i += 1u;
        if (j < g_len && i == cx.gather[j].hi) {
            assert ch == ')';
            state = active;
            j += 1u;
        }
    }

    let cx = ecx;
    let session_call = bind mk_call_(cx,sp,
                                     mk_access(cx,sp,["ext_cx"], "session"),
                                     []);
    let pcall = mk_call(cx,sp,
                       ["syntax", "parse", "parser",
                        "parse_from_source_str"],
                       [node.mk_parse_fn(cx,sp),
                        mk_str(cx,sp, "<anon>"),
                        mk_path(cx,sp, ["option","none"]),
                        mk_unary(cx,sp, ast::box(ast::imm),
                                 mk_str(cx,sp, str2)),
                        mk_access_(cx,sp,
                                   mk_access_(cx,sp, session_call(), "opts"),
                                   "cfg"),
                        mk_access_(cx,sp, session_call(), "parse_sess")]
                      );
    let rcall = pcall;
    if (g_len > 0u) {
        rcall = mk_call(cx,sp,
                        ["syntax", "ext", "qquote", "replace"],
                        [pcall,
                         mk_vec_e(cx,sp, vec::map(qcx.gather) {|g|
                             mk_call(cx,sp,
                                     ["syntax", "ext", "qquote", g.constr],
                                     [g.e])
                         })]);
    }

    ret rcall;
}

fn replace(e: @ast::expr, repls: [fragment]) -> @ast::expr {
    let aft = default_ast_fold();
    let f_pre = {fold_expr: bind replace_expr(repls, _, _, _,
                                              aft.fold_expr),
                 fold_ty: bind replace_ty(repls, _, _, _,
                                          aft.fold_ty)
                 with *aft};
    let f = make_fold(f_pre);
    ret f.fold_expr(e);
}

fn replace_expr(repls: [fragment],
                e: ast::expr_, s: span, fld: ast_fold,
                orig: fn@(ast::expr_, span, ast_fold)->(ast::expr_, span))
    -> (ast::expr_, span)
{
    alt e {
      ast::expr_mac({node: mac_var(i), _}) {
        alt (repls[i]) {
          from_expr(r) {(r.node, r.span)}
          _ {fail /* fixme error message */}}}
      _ {orig(e,s,fld)}
    }
}

fn replace_ty(repls: [fragment],
                e: ast::ty_, s: span, fld: ast_fold,
                orig: fn@(ast::ty_, span, ast_fold)->(ast::ty_, span))
    -> (ast::ty_, span)
{
    alt e {
      ast::ty_mac({node: mac_var(i), _}) {
        alt (repls[i]) {
          from_ty(r) {(r.node, r.span)}
          _ {fail /* fixme error message */}}}
      _ {orig(e,s,fld)}
    }
}

fn print_expr(expr: @ast::expr) {
    let stdout = std::io::stdout();
    let pp = pprust::rust_printer(stdout);
    pprust::print_expr(pp, expr);
    pp::eof(pp.s);
    stdout.write_str("\n");
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
