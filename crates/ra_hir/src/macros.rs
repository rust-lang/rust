use std::sync::Arc;

use ra_db::{SyntaxDatabase, LocalSyntaxPtr};
use ra_syntax::{
    TextRange, TextUnit, SourceFileNode, AstNode, SyntaxNode,
    ast,
};

// Hard-coded defs for now :-(
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MacroDef {
    CTry,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroInput {
    // Should be token trees
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroExpansion {
    text: String,
    ranges_map: Vec<(TextRange, TextRange)>,
    ptr: LocalSyntaxPtr,
}

salsa::query_group! {

pub trait MacroDatabase: SyntaxDatabase {
    fn expand_macro(def: MacroDef, input: MacroInput) -> Option<Arc<MacroExpansion>> {
        type ExpandMacroQuery;
    }
}

}

fn expand_macro(
    _db: &impl MacroDatabase,
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
    let ptr = LocalSyntaxPtr::new(match_arg.syntax());
    let src_range = TextRange::offset_len(0.into(), TextUnit::of_str(&input.text));
    let ranges_map = vec![(src_range, match_arg.syntax().range())];
    let res = MacroExpansion {
        text,
        ranges_map,
        ptr,
    };
    Some(Arc::new(res))
}

impl MacroExpansion {
    pub fn file(&self) -> SourceFileNode {
        SourceFileNode::parse(&self.text)
    }

    pub fn syntax(&self) -> SyntaxNode {
        self.ptr.resolve(&self.file())
    }
    pub fn map_range_back(&self, tgt_range: TextRange) -> Option<TextRange> {
        for (s_range, t_range) in self.ranges_map.iter() {
            if tgt_range.is_subrange(&t_range) {
                let tgt_at_zero_range = tgt_range - tgt_range.start();
                let tgt_range_offset = tgt_range.start() - t_range.start();
                let src_range = tgt_at_zero_range + tgt_range_offset + s_range.start();
                return Some(src_range);
            }
        }
        None
    }
    pub fn map_range_forward(&self, src_range: TextRange) -> Option<TextRange> {
        for (s_range, t_range) in self.ranges_map.iter() {
            if src_range.is_subrange(&s_range) {
                let src_at_zero_range = src_range - src_range.start();
                let src_range_offset = src_range.start() - s_range.start();
                let src_range = src_at_zero_range + src_range_offset + t_range.start();
                return Some(src_range);
            }
        }
        None
    }
}
