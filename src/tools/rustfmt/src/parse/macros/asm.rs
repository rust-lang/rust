use rustc_ast::ast;
use rustc_builtin_macros::asm::{parse_asm_args, AsmArgs};

use crate::rewrite::RewriteContext;

#[allow(dead_code)]
pub(crate) fn parse_asm(context: &RewriteContext<'_>, mac: &ast::MacCall) -> Option<AsmArgs> {
    let ts = mac.args.inner_tokens();
    let mut parser = super::build_parser(context, ts);
    parse_asm_args(&mut parser, context.parse_sess.inner(), mac.span(), false).ok()
}
