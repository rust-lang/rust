use rustc_ast::ast;
use rustc_parse::parser::asm::{AsmArg, parse_asm_args};

use crate::rewrite::RewriteContext;

#[allow(dead_code)]
pub(crate) fn parse_asm(context: &RewriteContext<'_>, mac: &ast::MacCall) -> Option<Vec<AsmArg>> {
    let ts = mac.args.tokens.clone();
    let mut parser = super::build_parser(context, ts);
    parse_asm_args(&mut parser, mac.span(), ast::AsmMacro::Asm).ok()
}
