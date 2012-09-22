use ast::{crate, expr_, mac_invoc,
                     mac_aq, mac_var};
use parse::parser;
use parse::parser::parse_from_source_str;
use dvec::DVec;
use parse::token::ident_interner;

use fold::*;
use visit::*;
use ext::base::*;
use ext::build::*;
use print::*;
use io::*;

use codemap::span;

struct gather_item {
    lo: uint,
    hi: uint,
    e: @ast::expr,
    constr: ~str
}

type aq_ctxt = @{lo: uint, gather: DVec<gather_item>};
enum fragment {
    from_expr(@ast::expr),
    from_ty(@ast::ty)
}

fn ids_ext(cx: ext_ctxt, strs: ~[~str]) -> ~[ast::ident] {
    strs.map(|str| cx.parse_sess().interner.intern(@*str))
}
fn id_ext(cx: ext_ctxt, str: ~str) -> ast::ident {
    cx.parse_sess().interner.intern(@str)
}


trait qq_helper {
    fn span() -> span;
    fn visit(aq_ctxt, vt<aq_ctxt>);
    fn extract_mac() -> Option<ast::mac_>;
    fn mk_parse_fn(ext_ctxt,span) -> @ast::expr;
    fn get_fold_fn() -> ~str;
}

impl @ast::crate: qq_helper {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_crate(*self, cx, v);}
    fn extract_mac() -> Option<ast::mac_> {fail}
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp,
                ids_ext(cx, ~[~"syntax", ~"ext", ~"qquote", ~"parse_crate"]))
    }
    fn get_fold_fn() -> ~str {~"fold_crate"}
}
impl @ast::expr: qq_helper {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_expr(self, cx, v);}
    fn extract_mac() -> Option<ast::mac_> {
        match (self.node) {
          ast::expr_mac({node: mac, _}) => Some(mac),
          _ => None
        }
    }
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp,
                ids_ext(cx, ~[~"syntax", ~"ext", ~"qquote", ~"parse_expr"]))
    }
    fn get_fold_fn() -> ~str {~"fold_expr"}
}
impl @ast::ty: qq_helper {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_ty(self, cx, v);}
    fn extract_mac() -> Option<ast::mac_> {
        match (self.node) {
          ast::ty_mac({node: mac, _}) => Some(mac),
          _ => None
        }
    }
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp,
                ids_ext(cx, ~[~"syntax", ~"ext", ~"qquote", ~"parse_ty"]))
    }
    fn get_fold_fn() -> ~str {~"fold_ty"}
}
impl @ast::item: qq_helper {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_item(self, cx, v);}
    fn extract_mac() -> Option<ast::mac_> {fail}
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp,
                ids_ext(cx, ~[~"syntax", ~"ext", ~"qquote", ~"parse_item"]))
    }
    fn get_fold_fn() -> ~str {~"fold_item"}
}
impl @ast::stmt: qq_helper {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_stmt(self, cx, v);}
    fn extract_mac() -> Option<ast::mac_> {fail}
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp,
                ids_ext(cx, ~[~"syntax", ~"ext", ~"qquote", ~"parse_stmt"]))
    }
    fn get_fold_fn() -> ~str {~"fold_stmt"}
}
impl @ast::pat: qq_helper {
    fn span() -> span {self.span}
    fn visit(cx: aq_ctxt, v: vt<aq_ctxt>) {visit_pat(self, cx, v);}
    fn extract_mac() -> Option<ast::mac_> {fail}
    fn mk_parse_fn(cx: ext_ctxt, sp: span) -> @ast::expr {
        mk_path(cx, sp, ids_ext(cx, ~[~"syntax", ~"ext", ~"qquote",
                                      ~"parse_pat"]))
    }
    fn get_fold_fn() -> ~str {~"fold_pat"}
}

fn gather_anti_quotes<N: qq_helper>(lo: uint, node: N) -> aq_ctxt
{
    let v = @{visit_expr: |node, &&cx, v| visit_aq(node, ~"from_expr", cx, v),
              visit_ty: |node, &&cx, v| visit_aq(node, ~"from_ty", cx, v),
              .. *default_visitor()};
    let cx = @{lo:lo, gather: DVec()};
    node.visit(cx, mk_vt(v));
    // FIXME (#2250): Maybe this is an overkill (merge_sort), it might
    // be better to just keep the gather array in sorted order.
    do cx.gather.swap |v| {
        pure fn by_lo(a: &gather_item, b: &gather_item) -> bool {
            a.lo < b.lo
        }
        std::sort::merge_sort(by_lo, v)
    };
    return cx;
}

fn visit_aq<T:qq_helper>(node: T, constr: ~str, &&cx: aq_ctxt, v: vt<aq_ctxt>)
{
    match (node.extract_mac()) {
      Some(mac_aq(sp, e)) => {
        cx.gather.push(gather_item {
            lo: sp.lo - cx.lo,
            hi: sp.hi - cx.lo,
            e: e,
            constr: constr});
      }
      _ => node.visit(cx, v)
    }
}

fn is_space(c: char) -> bool {
    parse::lexer::is_whitespace(c)
}

fn expand_ast(ecx: ext_ctxt, _sp: span,
              arg: ast::mac_arg, body: ast::mac_body)
    -> @ast::expr
{
    let mut what = ~"expr";
    do option::iter(arg) |arg| {
        let args: ~[@ast::expr] =
            match arg.node {
              ast::expr_vec(elts, _) => elts,
              _ => {
                ecx.span_fatal
                    (_sp, ~"#ast requires arguments of the form `~[...]`.")
              }
            };
        if vec::len::<@ast::expr>(args) != 1u {
            ecx.span_fatal(_sp, ~"#ast requires exactly one arg");
        }
        match (args[0].node) {
          ast::expr_path(@{idents: id, _}) if vec::len(id) == 1u
            => what = *ecx.parse_sess().interner.get(id[0]),
          _ => ecx.span_fatal(args[0].span, ~"expected an identifier")
        }
    }
    let body = get_mac_body(ecx,_sp,body);

    return match what {
      ~"crate" => finish(ecx, body, parse_crate),
      ~"expr" => finish(ecx, body, parse_expr),
      ~"ty" => finish(ecx, body, parse_ty),
      ~"item" => finish(ecx, body, parse_item),
      ~"stmt" => finish(ecx, body, parse_stmt),
      ~"pat" => finish(ecx, body, parse_pat),
      _ => ecx.span_fatal(_sp, ~"unsupported ast type")
    };
}

fn parse_crate(p: parser) -> @ast::crate { p.parse_crate_mod(~[]) }
fn parse_ty(p: parser) -> @ast::ty { p.parse_ty(false) }
fn parse_stmt(p: parser) -> @ast::stmt { p.parse_stmt(~[]) }
fn parse_expr(p: parser) -> @ast::expr { p.parse_expr() }
fn parse_pat(p: parser) -> @ast::pat { p.parse_pat(true) }

fn parse_item(p: parser) -> @ast::item {
    match p.parse_item(~[]) {
      Some(item) => item,
      None       => fail ~"parse_item: parsing an item failed"
    }
}

fn finish<T: qq_helper>
    (ecx: ext_ctxt, body: ast::mac_body_, f: fn (p: parser) -> T)
    -> @ast::expr
{
    let cm = ecx.codemap();
    let str = @codemap::span_to_snippet(body.span, cm);
    debug!("qquote--str==%?", str);
    let fname = codemap::mk_substr_filename(cm, body.span);
    let node = parse_from_source_str
        (f, fname, codemap::fss_internal(body.span), str,
         ecx.cfg(), ecx.parse_sess());
    let loc = codemap::lookup_char_pos(cm, body.span.lo);

    let sp = node.span();
    let qcx = gather_anti_quotes(sp.lo, node);
    let cx = qcx;

    for uint::range(1u, cx.gather.len()) |i| {
        assert cx.gather[i-1u].lo < cx.gather[i].lo;
        // ^^ check that the vector is sorted
        assert cx.gather[i-1u].hi <= cx.gather[i].lo;
        // ^^ check that the spans are non-overlapping
    }

    let mut str2 = ~"";
    enum state {active, skip(uint), blank};
    let mut state = active;
    let mut i = 0u, j = 0u;
    let g_len = cx.gather.len();
    for str::chars_each(*str) |ch| {
        if (j < g_len && i == cx.gather[j].lo) {
            assert ch == '$';
            let repl = fmt!("$%u ", j);
            state = skip(str::char_len(repl));
            str2 += repl;
        }
        match copy state {
          active => str::push_char(&mut str2, ch),
          skip(1u) => state = blank,
          skip(sk) => state = skip (sk-1u),
          blank if is_space(ch) => str::push_char(&mut str2, ch),
          blank => str::push_char(&mut str2, ' ')
        }
        i += 1u;
        if (j < g_len && i == cx.gather[j].hi) {
            assert ch == ')';
            state = active;
            j += 1u;
        }
    }

    let cx = ecx;

    let cfg_call = || mk_call_(
        cx, sp, mk_access(cx, sp, ids_ext(cx, ~[~"ext_cx"]),
                          id_ext(cx, ~"cfg")), ~[]);

    let parse_sess_call = || mk_call_(
        cx, sp, mk_access(cx, sp, ids_ext(cx, ~[~"ext_cx"]),
                          id_ext(cx, ~"parse_sess")), ~[]);

    let pcall = mk_call(cx,sp,
                       ids_ext(cx, ~[~"syntax", ~"parse", ~"parser",
                        ~"parse_from_source_str"]),
                       ~[node.mk_parse_fn(cx,sp),
                        mk_uniq_str(cx,sp, fname),
                        mk_call(cx,sp,
                                ids_ext(cx, ~[~"syntax",~"ext",
                                 ~"qquote", ~"mk_file_substr"]),
                                ~[mk_uniq_str(cx,sp, loc.file.name),
                                 mk_uint(cx,sp, loc.line),
                                 mk_uint(cx,sp, loc.col)]),
                        mk_unary(cx,sp, ast::box(ast::m_imm),
                                 mk_uniq_str(cx,sp, str2)),
                        cfg_call(),
                        parse_sess_call()]
                      );
    let mut rcall = pcall;
    if (g_len > 0u) {
        rcall = mk_call(cx,sp,
                        ids_ext(cx, ~[~"syntax", ~"ext", ~"qquote",
                                      ~"replace"]),
                        ~[pcall,
                          mk_uniq_vec_e(cx,sp, qcx.gather.map_to_vec(|g| {
                             mk_call(cx,sp,
                                     ids_ext(cx, ~[~"syntax", ~"ext",
                                                   ~"qquote", g.constr]),
                                     ~[g.e])})),
                         mk_path(cx,sp,
                                 ids_ext(cx, ~[~"syntax", ~"ext", ~"qquote",
                                               node.get_fold_fn()]))]);
    }
    return rcall;
}

fn replace<T>(node: T, repls: ~[fragment], ff: fn (ast_fold, T) -> T)
    -> T
{
    let aft = default_ast_fold();
    let f_pre = @{fold_expr: |a,b,c|replace_expr(repls, a, b, c,
                                                  aft.fold_expr),
                  fold_ty: |a,b,c|replace_ty(repls, a, b, c,
                                              aft.fold_ty),
                  .. *aft};
    return ff(make_fold(f_pre), node);
}
fn fold_crate(f: ast_fold, &&n: @ast::crate) -> @ast::crate {
    @f.fold_crate(*n)
}
fn fold_expr(f: ast_fold, &&n: @ast::expr) -> @ast::expr {f.fold_expr(n)}
fn fold_ty(f: ast_fold, &&n: @ast::ty) -> @ast::ty {f.fold_ty(n)}
fn fold_item(f: ast_fold, &&n: @ast::item) -> @ast::item {
    option::get(f.fold_item(n)) //HACK: we know we don't drop items
}
fn fold_stmt(f: ast_fold, &&n: @ast::stmt) -> @ast::stmt {f.fold_stmt(n)}
fn fold_pat(f: ast_fold, &&n: @ast::pat) -> @ast::pat {f.fold_pat(n)}

fn replace_expr(repls: ~[fragment],
                e: ast::expr_, s: span, fld: ast_fold,
                orig: fn@(ast::expr_, span, ast_fold)->(ast::expr_, span))
    -> (ast::expr_, span)
{
    match e {
      ast::expr_mac({node: mac_var(i), _}) => match (repls[i]) {
        from_expr(r) => (r.node, r.span),
        _ => fail /* fixme error message */
      },
      _ => orig(e,s,fld)
    }
}

fn replace_ty(repls: ~[fragment],
                e: ast::ty_, s: span, fld: ast_fold,
                orig: fn@(ast::ty_, span, ast_fold)->(ast::ty_, span))
    -> (ast::ty_, span)
{
    match e {
      ast::ty_mac({node: mac_var(i), _}) => match (repls[i]) {
        from_ty(r) => (r.node, r.span),
        _ => fail /* fixme error message */
      },
      _ => orig(e,s,fld)
    }
}

fn mk_file_substr(fname: ~str, line: uint, col: uint) ->
    codemap::file_substr {
    codemap::fss_external({filename: fname, line: line, col: col})
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
