use ra_db::FilesDatabase;
use ra_syntax::{
    SyntaxNode, AstNode, SourceFile,
    ast, algo::find_covering_node,
};

use crate::{
    TextRange, FileRange,
    db::RootDatabase,
};

pub(crate) fn extend_selection(db: &RootDatabase, frange: FileRange) -> TextRange {
    let source_file = db.source_file(frange.file_id);
    if let Some(range) = extend_selection_in_macro(db, &source_file, frange) {
        return range;
    }
    ra_ide_api_light::extend_selection(source_file.syntax(), frange.range).unwrap_or(frange.range)
}

fn extend_selection_in_macro(
    _db: &RootDatabase,
    source_file: &SourceFile,
    frange: FileRange,
) -> Option<TextRange> {
    let macro_call = find_macro_call(source_file.syntax(), frange.range)?;
    let (off, exp) = hir::MacroDef::ast_expand(macro_call)?;
    let dst_range = exp.map_range_forward(frange.range - off)?;
    let dst_range = ra_ide_api_light::extend_selection(&exp.syntax(), dst_range)?;
    let src_range = exp.map_range_back(dst_range)? + off;
    Some(src_range)
}

fn find_macro_call(node: &SyntaxNode, range: TextRange) -> Option<&ast::MacroCall> {
    find_covering_node(node, range)
        .ancestors()
        .find_map(ast::MacroCall::cast)
}

#[cfg(test)]
mod tests {
    use ra_syntax::TextRange;

    use crate::mock_analysis::single_file_with_range;

    #[test]
    fn extend_selection_inside_macros() {
        let (analysis, frange) = single_file_with_range(
            "
            fn main() {
                ctry!(foo(|x| <|>x<|>));
            }
        ",
        );
        let r = analysis.extend_selection(frange).unwrap();
        assert_eq!(r, TextRange::from_to(51.into(), 56.into()));
    }
}
