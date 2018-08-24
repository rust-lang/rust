// ignore-cross-compile

// error-pattern:expected expression, found statement (`let`)

#![feature(quote, rustc_private)]

extern crate syntax;
extern crate syntax_pos;

use syntax::ast;
use syntax::source_map;
use syntax::print::pprust;
use syntax::symbol::Symbol;
use syntax_pos::DUMMY_SP;

fn main() {
    syntax::with_globals(|| run());
}

fn run() {
    let ps = syntax::parse::ParseSess::new(source_map::FilePathMapping::empty());
    let mut resolver = syntax::ext::base::DummyResolver;
    let mut cx = syntax::ext::base::ExtCtxt::new(
        &ps,
        syntax::ext::expand::ExpansionConfig::default("qquote".to_string()),
        &mut resolver);
    let cx = &mut cx;

    println!("{}", pprust::expr_to_string(&*quote_expr!(&cx, 23)));
    assert_eq!(pprust::expr_to_string(&*quote_expr!(&cx, 23)), "23");

    let expr = quote_expr!(&cx, let x isize = 20;);
    println!("{}", pprust::expr_to_string(&*expr));
    assert_eq!(pprust::expr_to_string(&*expr), "let x isize = 20;");
}
