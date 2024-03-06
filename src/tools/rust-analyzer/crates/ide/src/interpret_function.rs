use hir::Semantics;
use ide_db::{
    base_db::{FilePosition, SourceDatabaseExt},
    LineIndexDatabase, RootDatabase,
};
use std::{fmt::Write, time::Instant};
use syntax::{algo::ancestors_at_offset, ast, AstNode, TextRange};

// Feature: Interpret Function
//
// |===
// | Editor  | Action Name
//
// | VS Code | **rust-analyzer: Interpret Function**
// |===
pub(crate) fn interpret_function(db: &RootDatabase, position: FilePosition) -> String {
    let start_time = Instant::now();
    let mut result =
        find_and_interpret(db, position).unwrap_or_else(|| "Not inside a function body".to_owned());
    let duration = Instant::now() - start_time;
    writeln!(result).unwrap();
    writeln!(result, "----------------------").unwrap();
    writeln!(result, "  Finished in {}s", duration.as_secs_f32()).unwrap();
    result
}

fn find_and_interpret(db: &RootDatabase, position: FilePosition) -> Option<String> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(position.file_id);

    let item = ancestors_at_offset(source_file.syntax(), position.offset)
        .filter(|it| !ast::MacroCall::can_cast(it.kind()))
        .find_map(ast::Item::cast)?;
    let def = match item {
        ast::Item::Fn(it) => sema.to_def(&it)?,
        _ => return None,
    };
    let span_formatter = |file_id, text_range: TextRange| {
        let path = &db
            .source_root(db.file_source_root(file_id))
            .path_for_file(&file_id)
            .map(|x| x.to_string());
        let path = path.as_deref().unwrap_or("<unknown file>");
        match db.line_index(file_id).try_line_col(text_range.start()) {
            Some(line_col) => format!("file://{path}#{}:{}", line_col.line + 1, line_col.col),
            None => format!("file://{path} range {:?}", text_range),
        }
    };
    Some(def.eval(db, span_formatter))
}
