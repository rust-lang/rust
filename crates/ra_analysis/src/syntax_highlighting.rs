use ra_syntax::{ast, AstNode,};
use ra_editor::HighlightedRange;
use ra_db::SyntaxDatabase;

use crate::{
    db::RootDatabase,
    FileId, Cancelable,
};

pub(crate) fn highlight(db: &RootDatabase, file_id: FileId) -> Cancelable<Vec<HighlightedRange>> {
    let source_file = db.source_file(file_id);
    let mut res = ra_editor::highlight(source_file.syntax());
    for macro_call in source_file
        .syntax()
        .descendants()
        .filter_map(ast::MacroCall::cast)
    {
        if let Some((off, exp)) = crate::macros::expand(db, file_id, macro_call) {
            let mapped_ranges = ra_editor::highlight(exp.syntax().borrowed())
                .into_iter()
                .filter_map(|r| {
                    let mapped_range = exp.map_range_back(r.range)?;
                    let res = HighlightedRange {
                        range: mapped_range + off,
                        tag: r.tag,
                    };
                    Some(res)
                });
            res.extend(mapped_ranges);
        }
    }
    Ok(res)
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
                HighlightedRange { range: [41; 46), tag: "macro" },
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
