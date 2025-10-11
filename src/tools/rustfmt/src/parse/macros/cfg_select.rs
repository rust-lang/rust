use rustc_ast::tokenstream::TokenStream;
use rustc_parse::parser::{self, cfg_select::CfgSelectBranches};

use crate::rewrite::RewriteContext;

pub(crate) fn parse_cfg_select(
    context: &RewriteContext<'_>,
    ts: TokenStream,
) -> Option<CfgSelectBranches> {
    let mut parser = super::build_parser(context, ts);
    parser::cfg_select::parse_cfg_select(&mut parser).ok()
}
