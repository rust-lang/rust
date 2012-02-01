import driver::session;

import option::{none, some};

import syntax::ast::{crate, expr_, expr_mac, mac_invoc,
                     mac_qq, mac_aq, mac_var};
import syntax::fold::*;
import syntax::visit::*;
import syntax::ext::base::*;
import syntax::ext::build::*;
import syntax::parse::parser;
import syntax::parse::parser::{parse_from_source_str};

import syntax::print::*;
import std::io::*;

import codemap::span;

type aq_ctxt = @{lo: uint,
                 mutable gather: [{lo: uint, hi: uint, e: @ast::expr}]};

fn gather_anti_quotes(lo: uint, e: @ast::expr) -> aq_ctxt
{
    let v = @{visit_expr: visit_expr_aq
              with *default_visitor()};
    let cx = @{lo:lo, mutable gather: []};
    visit_expr_aq(e, cx, mk_vt(v));
    ret cx;
}

fn visit_expr_aq(expr: @ast::expr, &&cx: aq_ctxt, v: vt<aq_ctxt>)
{
    alt (expr.node) {
      expr_mac({node: mac_aq(sp, e), _}) {
        cx.gather += [{lo: sp.lo - cx.lo, hi: sp.hi - cx.lo,
                       e: e}];
      }
      _ {visit_expr(expr, cx, v);}
    }
}

fn is_space(c: char) -> bool {
    syntax::parse::lexer::is_whitespace(c)
}

fn expand_ast(ecx: ext_ctxt, _sp: span, _arg:
              ast::mac_arg, body: ast::mac_body)
    -> @ast::expr
{
    let body = get_mac_body(ecx,_sp,body);
    let cm = ecx.session().parse_sess.cm;
    let str = @codemap::span_to_snippet(body.span, cm);
    let (fname, ss) = codemap::get_substr_info(cm,
                                               body.span.lo, body.span.hi);
    let {node: e, _} = parse_from_source_str(parser::parse_expr,
                                             fname, some(ss), str,
                                             ecx.session().opts.cfg,
                                             ecx.session().parse_sess);
    ret expand_qquote(ecx, e.span, some(*str), e);
}

fn expand_qquote(ecx: ext_ctxt, sp: span, maybe_str: option::t<str>,
                 e: @ast::expr)
    -> @ast::expr
{
    let str = alt(maybe_str) {
      some(s) {s}
      none {codemap::span_to_snippet(sp, ecx.session().parse_sess.cm)}
    };
    let qcx = gather_anti_quotes(sp.lo, e);
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
                        "parse_expr_from_source_str"],
                       [mk_str(cx,sp, "<anon>"),
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
                         mk_vec_e(cx,sp, vec::map(qcx.gather, {|g| g.e}))]);
    }

    ret rcall;
}

fn replace(e: @ast::expr, repls: [@ast::expr]) -> @ast::expr {
    let aft = default_ast_fold();
    let f_pre = {fold_expr: bind replace_expr(repls, _, _, _,
                                              aft.fold_expr)
                 with *aft};
    let f = make_fold(f_pre);
    ret f.fold_expr(e);
}

fn replace_expr(repls: [@ast::expr],
                e: ast::expr_, s: span, fld: ast_fold,
                orig: fn@(ast::expr_, span, ast_fold)->(ast::expr_, span))
    -> (ast::expr_, span)
{
    alt e {
      expr_mac({node: mac_var(i), _}) {let r = repls[i]; (r.node, r.span)}
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
