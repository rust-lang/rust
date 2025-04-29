use rustc_ast::ast;
use rustc_builtin_macros::asm::{RawAsmArg, parse_raw_asm_args};

use crate::rewrite::RewriteContext;

#[allow(dead_code)]
pub(crate) fn parse_asm(
    context: &RewriteContext<'_>,
    mac: &ast::MacCall,
) -> Option<Vec<RawAsmArg>> {
    let ts = mac.args.tokens.clone();
    let mut parser = super::build_parser(context, ts);
    parse_raw_asm_args(&mut parser, mac.span(), ast::AsmMacro::Asm).ok()
}
