import std::ivec;
import std::option;
import std::map::hashmap;
import driver::session::session;
import codemap::span;
import std::map::new_str_hash;
import codemap;

type syntax_expander =
    fn(&ext_ctxt, span, @ast::expr, option::t[str]) -> @ast::expr ;
type macro_def = {ident: str, ext: syntax_extension};
type macro_definer =
    fn(&ext_ctxt, span, @ast::expr, option::t[str]) -> macro_def ;

tag syntax_extension {
    normal(syntax_expander);
    macro_defining(macro_definer);
}

// A temporary hard-coded map of methods for expanding syntax extension
// AST nodes into full ASTs
fn syntax_expander_table() -> hashmap[str, syntax_extension] {
    let syntax_expanders = new_str_hash[syntax_extension]();
    syntax_expanders.insert("fmt", normal(ext::fmt::expand_syntax_ext));
    syntax_expanders.insert("env", normal(ext::env::expand_syntax_ext));
    syntax_expanders.insert("macro",
                            macro_defining(ext::simplext::add_new_extension));
    syntax_expanders.insert("concat_idents",
                            normal(ext::concat_idents::expand_syntax_ext));
    syntax_expanders.insert("ident_to_str",
                            normal(ext::ident_to_str::expand_syntax_ext));
    ret syntax_expanders;
}

obj ext_ctxt(sess: @session, crate_file_name_hack: str, 
             mutable backtrace: span[]) {
    fn crate_file_name() -> str { ret crate_file_name_hack; }

    fn print_backtrace() {
        for sp: span in backtrace {
            sess.span_note(sp, "(while expanding this)")
        }
    }
    
    fn bt_push(sp: span) { backtrace += ~[sp]; }
    fn bt_pop() { ivec::pop(backtrace); }

    fn span_fatal(sp: span, msg: str) -> ! {
        self.print_backtrace();
        sess.span_fatal(sp, msg);
    }
    fn span_err(sp: span, msg: str) {
        self.print_backtrace();
        sess.span_err(sp, msg);
    }
    fn span_unimpl(sp:span, msg: str) -> ! {
        self.print_backtrace();
        sess.span_unimpl(sp, msg);
    }
    fn span_bug(sp:span, msg: str) -> ! {
        self.print_backtrace();
        sess.span_bug(sp, msg); 
    }
    fn bug(msg: str) -> ! {
        self.print_backtrace();
        sess.bug(msg);
    }
    fn next_id() -> ast::node_id {
        ret sess.next_node_id();
    }

}

fn mk_ctxt(sess: &session) -> ext_ctxt {
    // FIXME: Some extensions work by building ASTs with paths to functions
    // they need to call at runtime. As those functions live in the std crate,
    // the paths are prefixed with "std::". Unfortunately, these paths can't
    // work for code called from inside the stdard library, so here we pass
    // the extensions the file name of the crate being compiled so they can
    // use it to guess whether paths should be prepended with "std::". This is
    // super-ugly and needs a better solution.
    let crate_file_name_hack = sess.get_codemap().files.(0).name;

    ret ext_ctxt(@sess, crate_file_name_hack, ~[]);
}

fn expr_to_str(cx: &ext_ctxt, expr: @ast::expr, error: str) -> str {
    alt expr.node {
      ast::expr_lit(l) {
        alt l.node {
          ast::lit_str(s, _) { ret s; }
          _ { cx.span_fatal(l.span, error); }
        }
      }
      _ { cx.span_fatal(expr.span, error); }
    }
}

fn expr_to_ident(cx: &ext_ctxt, expr: @ast::expr, error: str) -> ast::ident {
    alt expr.node {
      ast::expr_path(p) {
        if ivec::len(p.node.types) > 0u || ivec::len(p.node.idents) != 1u {
            cx.span_fatal(expr.span, error);
        } else { ret p.node.idents.(0); }
      }
      _ { cx.span_fatal(expr.span, error); }
    }
}

fn make_new_lit(cx: &ext_ctxt, sp: codemap::span, lit: ast::lit_) ->
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
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
