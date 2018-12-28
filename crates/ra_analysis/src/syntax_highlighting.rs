use ra_syntax::{ast, AstNode, SourceFileNode, TextRange};
use ra_editor::HighlightedRange;
use ra_db::SyntaxDatabase;

use crate::{
    db::RootDatabase,
    FileId, Cancelable,
};

pub(crate) fn highlight(db: &RootDatabase, file_id: FileId) -> Cancelable<Vec<HighlightedRange>> {
    let source_file = db.source_file(file_id);
    let mut res = ra_editor::highlight(&source_file);
    for macro_call in source_file
        .syntax()
        .descendants()
        .filter_map(ast::MacroCall::cast)
    {
        if let Some(exp) = expand(db, file_id, macro_call) {
            let mapped_ranges = ra_editor::highlight(exp.source_file())
                .into_iter()
                .filter_map(|r| {
                    let mapped_range = exp.map_range_back(r.range)?;
                    let res = HighlightedRange {
                        range: mapped_range,
                        tag: r.tag,
                    };
                    Some(res)
                });
            res.extend(mapped_ranges);
        }
    }
    Ok(res)
}

fn expand(
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

struct MacroExpansion {
    source_file: SourceFileNode,
    ranges_map: Vec<(TextRange, TextRange)>,
}

impl MacroExpansion {
    fn source_file(&self) -> &SourceFileNode {
        &self.source_file
    }
    fn map_range_back(&self, tgt_range: TextRange) -> Option<TextRange> {
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
}

#[cfg(test)]
mod tests {
    use crate::mock_analysis::single_file;
    use test_utils::assert_eq_dbg;

    #[test]
    fn highlights_code_inside_macros() {
        let (analysis, file_id) = single_file(
            "
            fn main() {
                ctry!({ let x = 92; x});
            }
        ",
        );
        let highlights = analysis.highlight(file_id).unwrap();
        assert_eq_dbg(
            r#"[HighlightedRange { range: [13; 15), tag: "keyword" },
                HighlightedRange { range: [16; 20), tag: "function" },
                HighlightedRange { range: [41; 45), tag: "text" },
                HighlightedRange { range: [49; 52), tag: "keyword" },
                HighlightedRange { range: [57; 59), tag: "literal" },
                HighlightedRange { range: [49; 52), tag: "keyword" },
                HighlightedRange { range: [53; 54), tag: "function" },
                HighlightedRange { range: [57; 59), tag: "literal" },
                HighlightedRange { range: [61; 62), tag: "text" }]"#,
            &highlights,
        )
    }
}
