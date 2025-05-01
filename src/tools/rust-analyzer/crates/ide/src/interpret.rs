use hir::{ConstEvalError, DefWithBody, DisplayTarget, Semantics};
use ide_db::{FilePosition, LineIndexDatabase, RootDatabase, base_db::SourceDatabase};
use std::time::{Duration, Instant};
use stdx::format_to;
use syntax::{AstNode, TextRange, algo::ancestors_at_offset, ast};

// Feature: Interpret A Function, Static Or Const.
//
// | Editor  | Action Name |
// |---------|-------------|
// | VS Code | **rust-analyzer: Interpret** |
pub(crate) fn interpret(db: &RootDatabase, position: FilePosition) -> String {
    match find_and_interpret(db, position) {
        Some((duration, mut result)) => {
            result.push('\n');
            format_to!(result, "----------------------\n");
            format_to!(result, "  Finished in {}s\n", duration.as_secs_f32());
            result
        }
        _ => "Not inside a function, const or static".to_owned(),
    }
}

fn find_and_interpret(db: &RootDatabase, position: FilePosition) -> Option<(Duration, String)> {
    let sema = Semantics::new(db);
    let source_file = sema.parse_guess_edition(position.file_id);

    let item = ancestors_at_offset(source_file.syntax(), position.offset)
        .filter(|it| !ast::MacroCall::can_cast(it.kind()))
        .find_map(ast::Item::cast)?;
    let def: DefWithBody = match item {
        ast::Item::Fn(it) => sema.to_def(&it)?.into(),
        ast::Item::Const(it) => sema.to_def(&it)?.into(),
        ast::Item::Static(it) => sema.to_def(&it)?.into(),
        _ => return None,
    };
    let span_formatter = |file_id, text_range: TextRange| {
        let source_root = db.file_source_root(file_id).source_root_id(db);
        let source_root = db.source_root(source_root).source_root(db);

        let path = source_root.path_for_file(&file_id).map(|x| x.to_string());
        let path = path.as_deref().unwrap_or("<unknown file>");
        match db.line_index(file_id).try_line_col(text_range.start()) {
            Some(line_col) => format!("file://{path}:{}:{}", line_col.line + 1, line_col.col),
            None => format!("file://{path} range {text_range:?}"),
        }
    };
    let display_target = def.module(db).krate().to_display_target(db);
    let start_time = Instant::now();
    let res = match def {
        DefWithBody::Function(it) => it.eval(db, span_formatter),
        DefWithBody::Static(it) => it.eval(db).map(|it| it.render(db, display_target)),
        DefWithBody::Const(it) => it.eval(db).map(|it| it.render(db, display_target)),
        _ => unreachable!(),
    };
    let res = res.unwrap_or_else(|e| render_const_eval_error(db, e, display_target));
    let duration = Instant::now() - start_time;
    Some((duration, res))
}

pub(crate) fn render_const_eval_error(
    db: &RootDatabase,
    e: ConstEvalError,
    display_target: DisplayTarget,
) -> String {
    let span_formatter = |file_id, text_range: TextRange| {
        let source_root = db.file_source_root(file_id).source_root_id(db);
        let source_root = db.source_root(source_root).source_root(db);
        let path = source_root.path_for_file(&file_id).map(|x| x.to_string());
        let path = path.as_deref().unwrap_or("<unknown file>");
        match db.line_index(file_id).try_line_col(text_range.start()) {
            Some(line_col) => format!("file://{path}:{}:{}", line_col.line + 1, line_col.col),
            None => format!("file://{path} range {text_range:?}"),
        }
    };
    let mut r = String::new();
    _ = e.pretty_print(&mut r, db, span_formatter, display_target);
    r
}
