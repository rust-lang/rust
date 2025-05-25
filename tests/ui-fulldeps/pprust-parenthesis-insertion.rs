//@ run-pass
//@ ignore-cross-compile
//@ aux-crate: parser=parser.rs
//@ edition: 2021

// This test covers the AST pretty-printer's automatic insertion of parentheses
// into unparenthesized syntax trees according to precedence and various grammar
// restrictions and edge cases.
//
// For example if the following syntax tree represents the expression a*(b+c),
// in which the parenthesis is necessary for precedence:
//
//     Binary('*', Path("a"), Paren(Binary('+', Path("b"), Path("c"))))
//
// then the pretty-printer needs to be able to print the following
// unparenthesized syntax tree with an automatically inserted parenthesization.
//
//     Binary('*', Path("a"), Binary('+', Path("b"), Path("c")))
//
// Handling this correctly is relevant in real-world code when pretty-printing
// macro-generated syntax trees, in which expressions can get interpolated into
// one another without any parenthesization being visible in the syntax tree.
//
//     macro_rules! repro {
//         ($rhs:expr) => {
//             a * $rhs
//         };
//     }
//
//     let _ = repro!(b + c);

#![feature(rustc_private)]

extern crate rustc_ast;
extern crate rustc_ast_pretty;
extern crate rustc_parse;
extern crate rustc_session;
extern crate rustc_span;

use std::mem;
use std::process::ExitCode;

use parser::parse_expr;
use rustc_ast::ast::{Expr, ExprKind};
use rustc_ast::mut_visit::{self, MutVisitor};
use rustc_ast::ptr::P;
use rustc_ast_pretty::pprust;
use rustc_session::parse::ParseSess;

// Every parenthesis in the following expressions is re-inserted by the
// pretty-printer.
//
// FIXME: Some of them shouldn't be.
static EXPRS: &[&str] = &[
    // Straightforward binary operator precedence.
    "2 * 2 + 2",
    "2 + 2 * 2",
    "(2 + 2) * 2",
    "2 * (2 + 2)",
    "2 + 2 + 2",
    // Right-associative operator.
    "2 += 2 += 2",
    "(2 += 2) += 2",
    // Return has lower precedence than a binary operator.
    "(return 2) + 2",
    "2 + (return 2)", // FIXME: no parenthesis needed.
    "(return) + 2",   // FIXME: no parenthesis needed.
    // These mean different things.
    "return - 2",
    "(return) - 2",
    // Closures and jumps have equal precedence.
    "|| return break 2",
    "return break || 2",
    // Closures with a return type have especially high precedence.
    "|| -> T { x } + 1",
    "(|| { x }) + 1",
    // These mean different things.
    "if let _ = true && false {}",
    "if let _ = (true && false) {}",
    // Parentheses to call a named field, but not an unnamed field.
    "(self.fun)()",
    "self.0()",
    // Conditions end at the first curly brace, so struct expressions need to be
    // parenthesized. Except in a match guard, where conditions end at arrow.
    "if let _ = (Struct {}) {}",
    "match 2 { _ if let _ = Struct {} => {} }",
    // Match arms terminate eagerly, so parenthesization is needed around some
    // expressions.
    "match 2 { _ => 1 - 1 }",
    "match 2 { _ => ({ 1 }) - 1 }",
    // Grammar restriction: break value starting with a labeled loop is not
    // allowed, except if the break is also labeled.
    "break 'outer 'inner: loop {} + 2",
    "break ('inner: loop {} + 2)",
    // Grammar restriction: ranges cannot be the endpoint of another range.
    "(2..2)..2",
    "2..(2..2)",
    "(2..2)..",
    "..(2..2)",
    // Grammar restriction: comparison operators cannot be chained (1 < 2 == false).
    "((1 < 2) == false) as usize",
    // Grammar restriction: the value in let-else is not allowed to end in a
    // curly brace.
    "{ let _ = 1 + 1 else {}; }",
    "{ let _ = (loop {}) else {}; }",
    "{ let _ = mac!() else {}; }",
    "{ let _ = (mac! {}) else {}; }",
    // Parentheses are necessary to prevent an eager statement boundary.
    "{ 2 - 1 }",
    "{ (match 2 {}) - 1 }",
    "{ (match 2 {})() - 1 }",
    "{ (match 2 {})[0] - 1 }",
    "{ (loop {}) - 1 }",
    "match 2 { _ => (loop {}) - 1 }",
    // No eager statement boundary if followed by `.` or `?`.
    "{ loop {}.to_string() - 1 }",
    "match 2 { _ => loop {}.to_string() - 1 }",
    // Angle bracket is eagerly parsed as a path's generic argument list.
    "(2 as T) < U",
    "(2 as T<U>) < V", // FIXME: no parentheses needed.
    /*
    // FIXME: pretty-printer produces invalid syntax. `2 + 2 as T < U`
    "(2 + 2 as T) < U",
    */
    /*
    // FIXME: pretty-printer produces invalid syntax. `if (let _ = () && Struct {}.x) {}`
    "if let _ = () && (Struct {}).x {}",
    */
    /*
    // FIXME: pretty-printer produces invalid syntax. `for _ in 1..{ 2 } {}`
    "for _ in (1..{ 2 }) {}",
    */
    /*
    // FIXME: pretty-printer loses the attribute. `{ let Struct { field } = s; }`
    "{ let Struct { #[attr] field } = s; }",
    */
    /*
    // FIXME: pretty-printer turns this into a range. `0..to_string()`
    "(0.).to_string()",
    "0. .. 1.",
    */
];

// Flatten the content of parenthesis nodes into their parent node. For example
// this syntax tree representing the expression a*(b+c):
//
//     Binary('*', Path("a"), Paren(Binary('+', Path("b"), Path("c"))))
//
// would unparenthesize to:
//
//     Binary('*', Path("a"), Binary('+', Path("b"), Path("c")))
struct Unparenthesize;

impl MutVisitor for Unparenthesize {
    fn visit_expr(&mut self, e: &mut P<Expr>) {
        while let ExprKind::Paren(paren) = &mut e.kind {
            *e = mem::replace(paren, Expr::dummy());
        }
        mut_visit::walk_expr(self, e);
    }
}

fn main() -> ExitCode {
    let mut status = ExitCode::SUCCESS;
    let mut fail = |description: &str, before: &str, after: &str| {
        status = ExitCode::FAILURE;
        eprint!(
            "{description}\n  BEFORE: {before}\n   AFTER: {after}\n\n",
            before = before.replace('\n', "\n          "),
            after = after.replace('\n', "\n          "),
        );
    };

    rustc_span::create_default_session_globals_then(|| {
        let psess = &ParseSess::new(vec![rustc_parse::DEFAULT_LOCALE_RESOURCE]);

        for &source_code in EXPRS {
            let Some(expr) = parse_expr(psess, source_code) else {
                panic!("Failed to parse original test case: {source_code}");
            };

            // Check for FALSE POSITIVE: pretty-printer inserting parentheses where not needed.
            // Pseudocode:
            //   assert(expr == parse(print(expr)))
            let printed = &pprust::expr_to_string(&expr);
            let Some(expr2) = parse_expr(psess, printed) else {
                fail("Pretty-printer produced invalid syntax", source_code, printed);
                continue;
            };
            if format!("{expr:#?}") != format!("{expr2:#?}") {
                fail("Pretty-printer inserted unnecessary parenthesis", source_code, printed);
                continue;
            }

            // Check for FALSE NEGATIVE: pretty-printer failing to place necessary parentheses.
            // Pseudocode:
            //   assert(unparenthesize(expr) == unparenthesize(parse(print(unparenthesize(expr)))))
            let mut expr = expr;
            Unparenthesize.visit_expr(&mut expr);
            let printed = &pprust::expr_to_string(&expr);
            let Some(mut expr2) = parse_expr(psess, printed) else {
                fail("Pretty-printer with no parens produced invalid syntax", source_code, printed);
                continue;
            };
            Unparenthesize.visit_expr(&mut expr2);
            if format!("{expr:#?}") != format!("{expr2:#?}") {
                fail("Pretty-printer lost necessary parentheses", source_code, printed);
                continue;
            }
        }
    });

    status
}
