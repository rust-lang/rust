use std::map::HashMap;
use syntax::ast;
use ast::{ty, pat};
use syntax::codemap::{span};
use syntax::visit;
use syntax::print;

fn indent<R>(op: fn() -> R) -> R {
    // Use in conjunction with the log post-processor like `src/etc/indenter`
    // to make debug output more readable.
    debug!(">>");
    let r <- op();
    debug!("<< (Result = %?)", r);
    move r
}

struct _indenter {
    _i: (),
    drop { debug!("<<"); }
}

fn _indenter(_i: ()) -> _indenter {
    _indenter {
        _i: ()
    }
}

fn indenter() -> _indenter {
    debug!(">>");
    _indenter(())
}

type flag = HashMap<~str, ()>;

fn field_expr(f: ast::field) -> @ast::expr { return f.node.expr; }

fn field_exprs(fields: ~[ast::field]) -> ~[@ast::expr] {
    let mut es = ~[];
    for fields.each |f| { es.push(f.node.expr); }
    return es;
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
fn loop_query(b: ast::blk, p: fn@(ast::expr_) -> bool) -> bool {
    let rs = @mut false;
    let visit_expr =
        |e: @ast::expr, &&flag: @mut bool, v: visit::vt<@mut bool>| {
        *flag |= p(e.node);
        match e.node {
          // Skip inner loops, since a break in the inner loop isn't a
          // break inside the outer loop
          ast::expr_loop(*) | ast::expr_while(*)
          | ast::expr_loop_body(*) => {}
          _ => visit::visit_expr(e, flag, v)
        }
    };
    let v = visit::mk_vt(@{visit_expr: visit_expr
                           ,.. *visit::default_visitor()});
    visit::visit_block(b, rs, v);
    return *rs;
}

fn has_nonlocal_exits(b: ast::blk) -> bool {
    do loop_query(b) |e| {
        match e {
          ast::expr_break(_) | ast::expr_again(_) => true,
          _ => false
        }
    }
}

fn may_break(b: ast::blk) -> bool {
    do loop_query(b) |e| {
        match e {
          ast::expr_break(_) => true,
          _ => false
        }
    }
}

fn local_rhs_span(l: @ast::local, def: span) -> span {
    match l.node.init {
      Some(i) => return i.expr.span,
      _ => return def
    }
}

fn is_main_name(path: syntax::ast_map::path) -> bool {
    // FIXME (#34): path should be a constrained type, so we know
    // the call to last doesn't fail.
    vec::last(path) == syntax::ast_map::path_name(
        syntax::parse::token::special_idents::main
    )
}

fn pluralize(n: uint, s: ~str) -> ~str {
    if n == 1 { s }
    else { str::concat([s, ~"s"]) }
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
