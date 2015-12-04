// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-cross-compile

#![feature(rustc_private)]

extern crate syntax;

use syntax::ast::*;
use syntax::attr::*;
use syntax::ast;
use syntax::parse;
use syntax::parse::{ParseSess,filemap_to_tts, PResult};
use syntax::parse::new_parser_from_source_str;
use syntax::parse::parser::Parser;
use syntax::parse::token;
use syntax::ptr::P;
use syntax::str::char_at;
use syntax::parse::attr::*;
use syntax::print::pprust;
use std::fmt;

// Copied out of syntax::util::parser_testing

pub fn string_to_parser<'a>(ps: &'a ParseSess, source_str: String) -> Parser<'a> {
    new_parser_from_source_str(ps,
                               Vec::new(),
                               "bogofile".to_string(),
                               source_str)
}

fn with_error_checking_parse<T, F>(s: String, f: F) -> PResult<T> where
    F: FnOnce(&mut Parser) -> PResult<T>,
{
    let ps = ParseSess::new();
    let mut p = string_to_parser(&ps, s);
    let x = f(&mut p);

    if ps.span_diagnostic.handler().has_errors() || p.token != token::Eof {
        return Err(p.fatal("parse error"));
    }

    x
}

fn expr(s: &str) -> PResult<P<ast::Expr>> {
    with_error_checking_parse(s.to_string(), |p| {
        p.parse_expr()
    })
}

fn stmt(s: &str) -> PResult<P<ast::Stmt>> {
    with_error_checking_parse(s.to_string(), |p| {
        p.parse_stmt().map(|s| s.unwrap())
    })
}

fn attr(s: &str) -> PResult<ast::Attribute> {
    with_error_checking_parse(s.to_string(), |p| {
        p.parse_attribute(true)
    })
}

fn str_compare<T, F: Fn(&T) -> String>(e: &str, expected: &[T], actual: &[T], f: F) {
    let expected: Vec<_> = expected.iter().map(|e| f(e)).collect();
    let actual: Vec<_> = actual.iter().map(|e| f(e)).collect();

    if expected != actual {
        panic!("parsed `{}` as {:?}, expected {:?}", e, actual, expected);
    }
}

fn check_expr_attrs(es: &str, expected: &[&str]) {
    let e = expr(es).expect("parse error");
    let actual = &e.attrs;
    str_compare(es,
                &expected.iter().map(|r| attr(r).unwrap()).collect::<Vec<_>>(),
                actual.as_attr_slice(),
                pprust::attribute_to_string);
}

fn check_stmt_attrs(es: &str, expected: &[&str]) {
    let e = stmt(es).expect("parse error");
    let actual = e.node.attrs();
    str_compare(es,
                &expected.iter().map(|r| attr(r).unwrap()).collect::<Vec<_>>(),
                actual,
                pprust::attribute_to_string);
}

fn reject_expr_parse(es: &str) {
    assert!(expr(es).is_err(), "parser did not reject `{}`", es);
}

fn reject_stmt_parse(es: &str) {
    assert!(stmt(es).is_err(), "parser did not reject `{}`", es);
}

fn main() {
    let both = &["#[attr]", "#![attr]"];
    let outer = &["#[attr]"];
    let none = &[];

    check_expr_attrs("#[attr] box 0", outer);
    reject_expr_parse("box #![attr] 0");

    check_expr_attrs("#[attr] 0 <- #[attr] 0", none);
    check_expr_attrs("#[attr] (0 <- 0)", outer);
    reject_expr_parse("0 #[attr] <- 0");
    reject_expr_parse("0 <- #![attr] 0");

    check_expr_attrs("in #[attr] 0 {#[attr] 0}", none);
    check_expr_attrs("#[attr] (in 0 {0})", outer);
    reject_expr_parse("in 0 #[attr] {0}");
    reject_expr_parse("in 0 {#![attr] 0}");

    check_expr_attrs("#[attr] [#![attr]]", both);
    check_expr_attrs("#[attr] [#![attr] 0]", both);
    check_expr_attrs("#[attr] [#![attr] 0; 0]", both);
    check_expr_attrs("#[attr] [#![attr] 0, 0, 0]", both);
    reject_expr_parse("[#[attr]]");

    check_expr_attrs("#[attr] foo()", outer);
    check_expr_attrs("#[attr] x.foo()", outer);
    reject_expr_parse("foo#[attr]()");
    reject_expr_parse("foo(#![attr])");
    reject_expr_parse("x.foo(#![attr])");
    reject_expr_parse("x.#[attr]foo()");
    reject_expr_parse("x.#![attr]foo()");

    check_expr_attrs("#[attr] (#![attr])", both);
    check_expr_attrs("#[attr] (#![attr] #[attr] 0,)", both);
    check_expr_attrs("#[attr] (#![attr] #[attr] 0, 0)", both);

    check_expr_attrs("#[attr] 0 + #[attr] 0", none);
    check_expr_attrs("#[attr] 0 / #[attr] 0", none);
    check_expr_attrs("#[attr] 0 & #[attr] 0", none);
    check_expr_attrs("#[attr] 0 % #[attr] 0", none);
    check_expr_attrs("#[attr] (0 + 0)", outer);
    reject_expr_parse("0 + #![attr] 0");

    check_expr_attrs("#[attr] !0", outer);
    check_expr_attrs("#[attr] -0", outer);
    reject_expr_parse("!#![attr] 0");
    reject_expr_parse("-#![attr] 0");

    check_expr_attrs("#[attr] false", outer);
    check_expr_attrs("#[attr] 0", outer);
    check_expr_attrs("#[attr] 'c'", outer);

    check_expr_attrs("#[attr] x as Y", none);
    check_expr_attrs("#[attr] (x as Y)", outer);
    reject_expr_parse("x #![attr] as Y");

    reject_expr_parse("#[attr] if false {}");
    reject_expr_parse("if false #[attr] {}");
    reject_expr_parse("if false {#![attr]}");
    reject_expr_parse("if false {} #[attr] else {}");
    reject_expr_parse("if false {} else #[attr] {}");
    reject_expr_parse("if false {} else {#![attr]}");
    reject_expr_parse("if false {} else #[attr] if true {}");
    reject_expr_parse("if false {} else if true #[attr] {}");
    reject_expr_parse("if false {} else if true {#![attr]}");

    reject_expr_parse("#[attr] if let Some(false) = false {}");
    reject_expr_parse("if let Some(false) = false #[attr] {}");
    reject_expr_parse("if let Some(false) = false {#![attr]}");
    reject_expr_parse("if let Some(false) = false {} #[attr] else {}");
    reject_expr_parse("if let Some(false) = false {} else #[attr] {}");
    reject_expr_parse("if let Some(false) = false {} else {#![attr]}");
    reject_expr_parse("if let Some(false) = false {} else #[attr] if let Some(false) = true {}");
    reject_expr_parse("if let Some(false) = false {} else if let Some(false) = true #[attr] {}");
    reject_expr_parse("if let Some(false) = false {} else if let Some(false) = true {#![attr]}");

    check_expr_attrs("#[attr] while true {#![attr]}", both);

    check_expr_attrs("#[attr] while let Some(false) = true {#![attr]}", both);

    check_expr_attrs("#[attr] for x in y {#![attr]}", both);

    check_expr_attrs("#[attr] loop {#![attr]}", both);

    check_expr_attrs("#[attr] match true {#![attr] #[attr] _ => false}", both);

    check_expr_attrs("#[attr]      || #[attr] foo", outer);
    check_expr_attrs("#[attr] move || #[attr] foo", outer);
    check_expr_attrs("#[attr]      || #[attr] { #![attr] foo }", outer);
    check_expr_attrs("#[attr] move || #[attr] { #![attr] foo }", outer);
    check_expr_attrs("#[attr]      || { #![attr] foo }", outer);
    check_expr_attrs("#[attr] move || { #![attr] foo }", outer);
    reject_expr_parse("|| #![attr] foo");
    reject_expr_parse("move || #![attr] foo");
    reject_expr_parse("|| #![attr] {foo}");
    reject_expr_parse("move || #![attr] {foo}");

    check_expr_attrs("#[attr] { #![attr] }", both);
    check_expr_attrs("#[attr] { #![attr] let _ = (); }", both);
    check_expr_attrs("#[attr] { #![attr] let _ = (); foo }", both);

    check_expr_attrs("#[attr] x = y", none);
    check_expr_attrs("#[attr] (x = y)", outer);

    check_expr_attrs("#[attr] x += y", none);
    check_expr_attrs("#[attr] (x += y)", outer);

    check_expr_attrs("#[attr] foo.bar", outer);
    check_expr_attrs("(#[attr] foo).bar", none);

    check_expr_attrs("#[attr] foo.0", outer);
    check_expr_attrs("(#[attr] foo).0", none);

    check_expr_attrs("#[attr] foo[bar]", outer);
    check_expr_attrs("(#[attr] foo)[bar]", none);

    check_expr_attrs("#[attr] 0..#[attr] 0", none);
    check_expr_attrs("#[attr] 0..", none);
    reject_expr_parse("#[attr] ..#[attr] 0");
    reject_expr_parse("#[attr] ..");

    check_expr_attrs("#[attr] (0..0)", outer);
    check_expr_attrs("#[attr] (0..)", outer);
    check_expr_attrs("#[attr] (..0)", outer);
    check_expr_attrs("#[attr] (..)", outer);

    check_expr_attrs("#[attr] foo::bar::baz", outer);

    check_expr_attrs("#[attr] &0", outer);
    check_expr_attrs("#[attr] &mut 0", outer);
    check_expr_attrs("#[attr] & #[attr] 0", outer);
    check_expr_attrs("#[attr] &mut #[attr] 0", outer);
    reject_expr_parse("#[attr] &#![attr] 0");
    reject_expr_parse("#[attr] &mut #![attr] 0");

    check_expr_attrs("#[attr] break", outer);
    check_expr_attrs("#[attr] continue", outer);
    check_expr_attrs("#[attr] return", outer);

    check_expr_attrs("#[attr] foo!()", outer);
    check_expr_attrs("#[attr] foo!(#![attr])", outer);
    check_expr_attrs("#[attr] foo![]", outer);
    check_expr_attrs("#[attr] foo![#![attr]]", outer);
    check_expr_attrs("#[attr] foo!{}", outer);
    check_expr_attrs("#[attr] foo!{#![attr]}", outer);

    check_expr_attrs("#[attr] Foo { #![attr] bar: baz }", both);
    check_expr_attrs("#[attr] Foo { #![attr] ..foo }", both);
    check_expr_attrs("#[attr] Foo { #![attr] bar: baz, ..foo }", both);

    check_expr_attrs("#[attr] (#![attr] 0)", both);

    // Look at statements in their natural habitat...
    check_expr_attrs("{
        #[attr] let _ = 0;
        #[attr] 0;
        #[attr] foo!();
        #[attr] foo!{}
        #[attr] foo![];
    }", none);

    check_stmt_attrs("#[attr] let _ = 0", outer);
    check_stmt_attrs("#[attr] 0",         outer);
    check_stmt_attrs("#[attr] {#![attr]}", both);
    check_stmt_attrs("#[attr] foo!()",    outer);
    check_stmt_attrs("#[attr] foo![]",    outer);
    check_stmt_attrs("#[attr] foo!{}",    outer);

    reject_stmt_parse("#[attr] #![attr] let _ = 0");
    reject_stmt_parse("#[attr] #![attr] 0");
    reject_stmt_parse("#[attr] #![attr] foo!()");
    reject_stmt_parse("#[attr] #![attr] foo![]");
    reject_stmt_parse("#[attr] #![attr] foo!{}");

    // FIXME: Allow attributes in pattern constexprs?
    // would require parens in patterns to allow disambiguation...

    reject_expr_parse("match 0 {
        0...#[attr] 10 => ()
    }");
    reject_expr_parse("match 0 {
        0...#[attr] -10 => ()
    }");
    reject_expr_parse("match 0 {
        0...-#[attr] 10 => ()
    }");
    reject_expr_parse("match 0 {
        0...#[attr] FOO => ()
    }");

    // make sure we don't catch this bug again...
    reject_expr_parse("{
        fn foo() {
            #[attr];
        }
    }");
    reject_expr_parse("{
        fn foo() {
            #[attr]
        }
    }");
}
