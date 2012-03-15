import driver::session;

import syntax::ast::{crate, expr_, mac_invoc,
                     mac_aq, mac_var};
import syntax::fold::*;
import syntax::visit::*;
import syntax::ext::base::*;
import syntax::ext::build::*;
import syntax::parse::parser;
import syntax::parse::parser::{parser, parse_from_source_str};

import syntax::print::*;
import io::*;

import codemap::span;

type aq_ctxt = @{lo: uint,
                 mutable gather: [{lo: uint, hi: uint,
                                   e: @ast::expr,
                                   constr: str}]};
enum fragment {
    from_expr(@ast::expr),
    from_ty(@ast::ty)
}

iface qq_helper {
    fn span() -> span;
    fn visit(aq_ctxt, vt<aq_ctxt>);
    fn extract_mac() -> option<ast::mac_>;
    fn mk_parse_fn(ext_ctxt,span) -> @ast::expr;
    fn get_fold_fn() -> str;
}

impl of qq_helper for @ast::crate {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_crate(*self, cx, v);}
    fn extract_mac() -> option<ast::mac_> {fail}
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp, ["syntax", "ext", "qquote", "parse_crate"])
    }
    fn get_fold_fn() -> str {"fold_crate"}
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
    fn get_fold_fn() -> str {"fold_expr"}
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
    fn get_fold_fn() -> str {"fold_ty"}
}
impl of qq_helper for @ast::item {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_item(self, cx, v);}
    fn extract_mac() -> option<ast::mac_> {fail}
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp, ["syntax", "ext", "qquote", "parse_item"])
    }
    fn get_fold_fn() -> str {"fold_item"}
}
impl of qq_helper for @ast::stmt {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_stmt(self, cx, v);}
    fn extract_mac() -> option<ast::mac_> {fail}
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp, ["syntax", "ext", "qquote", "parse_stmt"])
    }
    fn get_fold_fn() -> str {"fold_stmt"}
}
impl of qq_helper for @ast::pat {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_pat(self, cx, v);}
    fn extract_mac() -> option<ast::mac_> {fail}
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp, ["syntax", "parse", "parser", "parse_pat"])
    }
    fn get_fold_fn() -> str {"fold_pat"}
}

fn gather_anti_quotes<N: qq_helper>(lo: uint, node: N) -> aq_ctxt
{
    let v = @{visit_expr: visit_aq_expr,
              visit_ty: visit_aq_ty
              with *default_visitor()};
    let cx = @{lo:lo, mutable gather: []};
    node.visit(cx, mk_vt(v));
    // FIXME: Maybe this is an overkill (merge_sort), it might be better
    //   to just keep the gather array in sorted order ...
    cx.gather = std::sort::merge_sort({|a,b| a.lo < b.lo}, copy cx.gather);
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
    let mut what = "expr";
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

    ret alt what {
      "crate" {finish(ecx, body, parse_crate)}
      "expr" {finish(ecx, body, parser::parse_expr)}
      "ty" {finish(ecx, body, parse_ty)}
      "item" {finish(ecx, body, parse_item)}
      "stmt" {finish(ecx, body, parse_stmt)}
      "pat" {finish(ecx, body, parser::parse_pat)}
      _ {ecx.span_fatal(_sp, "unsupported ast type")}
    };
}

fn parse_crate(p: parser) -> @ast::crate {
    parser::parse_crate_mod(p, [])
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

fn finish<T: qq_helper>
    (ecx: ext_ctxt, body: ast::mac_body_, f: fn (p: parser) -> T)
    -> @ast::expr
{
    let cm = ecx.session().parse_sess.cm;
    let str = @codemap::span_to_snippet(body.span, cm);
    #debug["qquote--str==%?", str];
    let fname = codemap::mk_substr_filename(cm, body.span);
    let node = parse_from_source_str
        (f, fname, codemap::fss_internal(body.span), str,
         ecx.session().opts.cfg, ecx.session().parse_sess);
    let loc = codemap::lookup_char_pos(cm, body.span.lo);

    let sp = node.span();
    let qcx = gather_anti_quotes(sp.lo, node);
    let cx = qcx;

    uint::range(1u, vec::len(cx.gather)) {|i|
        assert cx.gather[i-1u].lo < cx.gather[i].lo;
        // ^^ check that the vector is sorted
        assert cx.gather[i-1u].hi <= cx.gather[i].lo;
        // ^^ check that the spans are non-overlapping
    }

    let mut str2 = "";
    enum state {active, skip(uint), blank};
    let mut state = active;
    let mut i = 0u, j = 0u;
    let g_len = vec::len(cx.gather);
    str::chars_iter(*str) {|ch|
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
    let session_call = {||
        mk_call_(cx, sp, mk_access(cx, sp, ["ext_cx"], "session"), [])
    };

    let pcall = mk_call(cx,sp,
                       ["syntax", "parse", "parser",
                        "parse_from_source_str"],
                       [node.mk_parse_fn(cx,sp),
                        mk_str(cx,sp, fname),
                        mk_call(cx,sp,
                                ["syntax","ext","qquote", "mk_file_substr"],
                                [mk_str(cx,sp, loc.file.name),
                                 mk_uint(cx,sp, loc.line),
                                 mk_uint(cx,sp, loc.col)]),
                        mk_unary(cx,sp, ast::box(ast::m_imm),
                                 mk_str(cx,sp, str2)),
                        mk_access_(cx,sp,
                                   mk_access_(cx,sp, session_call(), "opts"),
                                   "cfg"),
                        mk_access_(cx,sp, session_call(), "parse_sess")]
                      );
    let mut rcall = pcall;
    if (g_len > 0u) {
        rcall = mk_call(cx,sp,
                        ["syntax", "ext", "qquote", "replace"],
                        [pcall,
                         mk_vec_e(cx,sp, vec::map(copy qcx.gather) {|g|
                             mk_call(cx,sp,
                                     ["syntax", "ext", "qquote", g.constr],
                                     [g.e])}),
                         mk_path(cx,sp,
                                 ["syntax", "ext", "qquote",
                                  node.get_fold_fn()])]);
    }
    ret rcall;
}

fn replace<T>(node: T, repls: [fragment], ff: fn (ast_fold, T) -> T)
    -> T
{
    let aft = default_ast_fold();
    let f_pre = {fold_expr: bind replace_expr(repls, _, _, _,
                                              aft.fold_expr),
                 fold_ty: bind replace_ty(repls, _, _, _,
                                          aft.fold_ty)
                 with *aft};
    ret ff(make_fold(f_pre), node);
}
fn fold_crate(f: ast_fold, &&n: @ast::crate) -> @ast::crate {
    @f.fold_crate(*n)
}
fn fold_expr(f: ast_fold, &&n: @ast::expr) -> @ast::expr {f.fold_expr(n)}
fn fold_ty(f: ast_fold, &&n: @ast::ty) -> @ast::ty {f.fold_ty(n)}
fn fold_item(f: ast_fold, &&n: @ast::item) -> @ast::item {f.fold_item(n)}
fn fold_stmt(f: ast_fold, &&n: @ast::stmt) -> @ast::stmt {f.fold_stmt(n)}
fn fold_pat(f: ast_fold, &&n: @ast::pat) -> @ast::pat {f.fold_pat(n)}

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
    let stdout = io::stdout();
    let pp = pprust::rust_printer(stdout);
    pprust::print_expr(pp, expr);
    pp::eof(pp.s);
    stdout.write_str("\n");
}

fn mk_file_substr(fname: str, line: uint, col: uint) -> codemap::file_substr {
    codemap::fss_external({filename: fname, line: line, col: col})
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
