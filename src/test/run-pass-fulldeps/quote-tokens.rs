#[allow(non_implicitly_copyable_typarams)];

extern mod syntax;

use syntax::ext::base::ext_ctxt;

fn syntax_extension(ext_cx: @ext_ctxt) {
    let e_toks : ~[syntax::ast::token_tree] = quote_tokens!(1 + 2);
    let p_toks : ~[syntax::ast::token_tree] = quote_tokens!((x, 1 .. 4, *));

    let a: @syntax::ast::expr = quote_expr!(1 + 2);
    let _b: Option<@syntax::ast::item> = quote_item!( const foo : int = $e_toks; );
    let _c: @syntax::ast::pat = quote_pat!( (x, 1 .. 4, *) );
    let _d: @syntax::ast::stmt = quote_stmt!( let x = $a; );
    let _e: @syntax::ast::expr = quote_expr!( match foo { $p_toks => 10 } );
}

fn main() {
}

