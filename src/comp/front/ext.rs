
import std::option;
import std::map::hashmap;
import driver::session::session;
import front::parser::parser;
import util::common::span;
import util::common::new_str_hash;

type syntax_expander =
    fn(&ext_ctxt, span, &vec[@ast::expr], option::t[str]) -> @ast::expr ;


// Temporary: to introduce a tag in order to make a recursive type work
tag syntax_extension { x(syntax_expander); }


// A temporary hard-coded map of methods for expanding syntax extension
// AST nodes into full ASTs
fn syntax_expander_table() -> hashmap[str, syntax_extension] {
    auto syntax_expanders = new_str_hash[syntax_extension]();
    syntax_expanders.insert("fmt", x(extfmt::expand_syntax_ext));
    syntax_expanders.insert("env", x(extenv::expand_syntax_ext));
    ret syntax_expanders;
}

type span_msg_fn = fn(span, str) -> !  ;

type next_id_fn = fn() -> ast::node_id ;


// Provides a limited set of services necessary for syntax extensions
// to do their thing
type ext_ctxt =
    rec(span_msg_fn span_fatal,
        span_msg_fn span_unimpl,
        next_id_fn next_id);

fn mk_ctxt(parser parser) -> ext_ctxt {
    auto sess = parser.get_session();
    fn ext_span_fatal_(session sess, span sp, str msg) -> ! {
        sess.span_fatal(sp, msg);
    }
    auto ext_span_fatal = bind ext_span_fatal_(sess, _, _);
    fn ext_span_unimpl_(session sess, span sp, str msg) -> ! {
        sess.span_unimpl(sp, msg);
    }
    auto ext_span_unimpl = bind ext_span_unimpl_(sess, _, _);
    fn ext_next_id_(parser parser) -> ast::node_id { parser.get_id() }
    auto ext_next_id = bind ext_next_id_(parser);
    ret rec(span_fatal=ext_span_fatal,
            span_unimpl=ext_span_unimpl,
            next_id=ext_next_id);
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
