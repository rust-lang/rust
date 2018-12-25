// ignore-cross-compile

#![feature(quote, rustc_private)]

extern crate syntax;
extern crate syntax_pos;

use syntax::ast;
use syntax::source_map::FilePathMapping;
use syntax::print::pprust;
use syntax::symbol::Symbol;
use syntax_pos::DUMMY_SP;

fn main() {
    let ps = syntax::parse::ParseSess::new(FilePathMapping::empty());
    let mut resolver = syntax::ext::base::DummyResolver;
    let mut cx = syntax::ext::base::ExtCtxt::new(
        &ps,
        syntax::ext::expand::ExpansionConfig::default("qquote".to_string()),
        &mut resolver);
    let cx = &mut cx;

    assert_eq!(pprust::expr_to_string(&*quote_expr!(&cx, 23)), "23");

    let expr = quote_expr!(&cx, 2 - $abcd + 7); //~ ERROR cannot find value `abcd` in this scope
    assert_eq!(pprust::expr_to_string(&*expr), "2 - $abcd + 7");
}
