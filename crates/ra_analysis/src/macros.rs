/// Begining of macro expansion.
///
/// This code should be moved out of ra_analysis into hir (?) ideally.
use ra_syntax::{ast, AstNode, SourceFileNode, TextRange};

use crate::{db::RootDatabase, FileId};

pub(crate) fn expand(
    _db: &RootDatabase,
    _file_id: FileId,
    macro_call: ast::MacroCall,
) -> Option<MacroExpansion> {
    let path = macro_call.path()?;
    if path.qualifier().is_some() {
        return None;
    }
    let name_ref = path.segment()?.name_ref()?;
    if name_ref.text() != "ctry" {
        return None;
    }

    let arg = macro_call.token_tree()?;
    let text = format!(
        r"
        fn dummy() {{
            match {} {{
                None => return Ok(None),
                Some(it) => it,
            }}
        }}",
        arg.syntax().text()
    );
    let file = SourceFileNode::parse(&text);
    let match_expr = file.syntax().descendants().find_map(ast::MatchExpr::cast)?;
    let match_arg = match_expr.expr()?;
    let ranges_map = vec![(arg.syntax().range(), match_arg.syntax().range())];
    let res = MacroExpansion {
        source_file: file,
        ranges_map,
    };
    Some(res)
}

pub(crate) struct MacroExpansion {
    pub(crate) source_file: SourceFileNode,
    pub(crate) ranges_map: Vec<(TextRange, TextRange)>,
}

impl MacroExpansion {
    pub(crate) fn source_file(&self) -> &SourceFileNode {
        &self.source_file
    }
    pub(crate) fn map_range_back(&self, tgt_range: TextRange) -> Option<TextRange> {
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
    pub(crate) fn map_range_forward(&self, src_range: TextRange) -> Option<TextRange> {
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
