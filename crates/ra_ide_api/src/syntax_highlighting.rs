use ra_syntax::{ast, AstNode,};
use ra_db::FilesDatabase;

use crate::{
    FileId, HighlightedRange,
    db::RootDatabase,
};

pub(crate) fn highlight(db: &RootDatabase, file_id: FileId) -> Vec<HighlightedRange> {
    let source_file = db.source_file(file_id);
    let mut res = ra_ide_api_light::highlight(source_file.syntax());
    for macro_call in source_file
        .syntax()
        .descendants()
        .filter_map(ast::MacroCall::cast)
    {
        if let Some((off, exp)) = hir::MacroDef::ast_expand(macro_call) {
            let mapped_ranges = ra_ide_api_light::highlight(&exp.syntax())
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
    res
}

#[cfg(test)]
mod tests {
    use crate::mock_analysis::single_file;

    use insta::assert_debug_snapshot_matches;

    #[test]
    fn highlights_code_inside_macros() {
        let (analysis, file_id) = single_file(
            "
            fn main() {
                ctry!({ let x = 92; x});
                vec![{ let x = 92; x}];
            }
            ",
        );
        let highlights = analysis.highlight(file_id).unwrap();
        assert_debug_snapshot_matches!("highlights_code_inside_macros", &highlights);
    }

    // FIXME: this test is not really necessary: artifact of the inital hacky
    // macros implementation.
    #[test]
    fn highlight_query_group_macro() {
        let (analysis, file_id) = single_file(
            "
            salsa::query_group! {
                pub trait HirDatabase: SyntaxDatabase {}
            }
            ",
        );
        let highlights = analysis.highlight(file_id).unwrap();
        assert_debug_snapshot_matches!("highlight_query_group_macro", &highlights);
    }
}
