import std::option;
import std::map::hashmap;

import driver::session::session;
import util::common::span;
import util::common::new_str_hash;

type syntax_expander = fn(&ext_ctxt, &parser::parser, span,
                          &vec[@ast::expr],
                          option::t[str]) -> @ast::expr;

// Temporary: to introduce a tag in order to make a recursive type work
tag syntax_extension {
    x(syntax_expander);
}

// A temporary hard-coded map of methods for expanding syntax extension
// AST nodes into full ASTs
fn syntax_expander_table() -> hashmap[str, syntax_extension] {
    auto syntax_expanders = new_str_hash[syntax_extension]();
    syntax_expanders.insert("fmt", x(extfmt::expand_syntax_ext));
    syntax_expanders.insert("env", x(extenv::expand_syntax_ext));
    ret syntax_expanders;
}

type span_err_fn = fn (span sp, str msg) -> !;

// Provides a limited set of services necessary for syntax extensions
// to do their thing
type ext_ctxt = rec(span_err_fn span_err);

fn mk_ctxt(session sess) -> ext_ctxt {
    fn ext_span_err_(session sess, span sp, str err) -> ! {
        sess.span_err(sp, err);
    }
    auto ext_span_err = bind ext_span_err_(sess, _, _);
    ret rec(span_err = ext_span_err);
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
