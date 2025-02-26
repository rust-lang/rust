#![feature(rustc_private)]

extern crate rustc_ast;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_parse;
extern crate rustc_session;
extern crate rustc_span;

use rustc_ast::ast::{DUMMY_NODE_ID, Expr};
use rustc_ast::mut_visit::MutVisitor;
use rustc_ast::node_id::NodeId;
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_errors::Diag;
use rustc_parse::parser::Recovery;
use rustc_session::parse::ParseSess;
use rustc_span::{DUMMY_SP, FileName, Span};

pub fn parse_expr(psess: &ParseSess, source_code: &str) -> Option<P<Expr>> {
    let parser = rustc_parse::unwrap_or_emit_fatal(rustc_parse::new_parser_from_source_str(
        psess,
        FileName::anon_source_code(source_code),
        source_code.to_owned(),
    ));

    let mut parser = parser.recovery(Recovery::Forbidden);
    let mut expr = parser.parse_expr().map_err(Diag::cancel).ok()?;
    if parser.token != token::Eof {
        return None;
    }

    Normalize.visit_expr(&mut expr);
    Some(expr)
}

// Erase Span information that could distinguish between identical expressions
// parsed from different source strings.
struct Normalize;

impl MutVisitor for Normalize {
    const VISIT_TOKENS: bool = true;

    fn visit_id(&mut self, id: &mut NodeId) {
        *id = DUMMY_NODE_ID;
    }

    fn visit_span(&mut self, span: &mut Span) {
        *span = DUMMY_SP;
    }
}
