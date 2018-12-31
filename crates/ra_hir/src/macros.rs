use std::sync::Arc;

use ra_db::SyntaxDatabase;
use ra_syntax::{TextRange, TextUnit, SourceFileNode, AstNode, ast};

// Hard-coded defs for now :-(
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MacroDef {
    CTry,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroInput {
    // Should be token trees
    text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroExpansion {
    text: String,
    ranges_map: Vec<(TextRange, TextRange)>,
}

salsa::query_group! {

pub trait MacrosDatabase: SyntaxDatabase {
    fn expand_macro(def: MacroDef, input: MacroInput) -> Option<Arc<MacroExpansion>> {
        type ExpandMacroQuery;
    }
}

}

fn expand_macro(
    _db: &impl MacrosDatabase,
    def: MacroDef,
    input: MacroInput,
) -> Option<Arc<MacroExpansion>> {
    let MacroDef::CTry = def;
    let text = format!(
        r"
        fn dummy() {{
            match {} {{
                None => return Ok(None),
                Some(it) => it,
            }}
        }}",
        input.text
    );
    let file = SourceFileNode::parse(&text);
    let match_expr = file.syntax().descendants().find_map(ast::MatchExpr::cast)?;
    let match_arg = match_expr.expr()?;
    let src_range = TextRange::offset_len(0.into(), TextUnit::of_str(&input.text));
    let ranges_map = vec![(src_range, match_arg.syntax().range())];
    let res = MacroExpansion { text, ranges_map };
    Some(Arc::new(res))
}
