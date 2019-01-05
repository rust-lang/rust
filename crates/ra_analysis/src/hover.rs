use ra_db::{Cancelable, SyntaxDatabase};
use ra_syntax::{ast, AstNode};

use crate::{db::RootDatabase, RangeInfo, FilePosition, FileRange};

pub(crate) fn hover(
    db: &RootDatabase,
    position: FilePosition,
) -> Cancelable<Option<RangeInfo<String>>> {
    let mut res = Vec::new();
    let range = if let Some(rr) = db.approximately_resolve_symbol(position)? {
        for nav in rr.resolves_to {
            res.extend(db.doc_text_for(nav)?)
        }
        rr.reference_range
    } else {
        let file = db.source_file(position.file_id);
        let expr: ast::Expr = ctry!(ra_editor::find_node_at_offset(
            file.syntax(),
            position.offset
        ));
        let frange = FileRange {
            file_id: position.file_id,
            range: expr.syntax().range(),
        };
        res.extend(db.type_of(frange)?);
        expr.syntax().range()
    };
    if res.is_empty() {
        return Ok(None);
    }
    let res = RangeInfo::new(range, res.join("\n\n---\n"));
    Ok(Some(res))
}

#[cfg(test)]
mod tests {
    use ra_syntax::TextRange;

    use crate::mock_analysis::single_file_with_position;

    #[test]
    fn hover_shows_type_of_an_expression() {
        let (analysis, position) = single_file_with_position(
            "
            pub fn foo() -> u32 { 1 }

            fn main() {
                let foo_test = foo()<|>;
            }
        ",
        );
        let hover = analysis.hover(position).unwrap().unwrap();
        assert_eq!(hover.range, TextRange::from_to(95.into(), 100.into()));
        assert_eq!(hover.info, "u32");
    }
}
