use hir::{DefWithBody, Semantics};
use ide_db::{FilePosition, RootDatabase};
use syntax::{AstNode, algo::ancestors_at_offset, ast};

// Feature: View Mir
//
// | Editor  | Action Name |
// |---------|-------------|
// | VS Code | **rust-analyzer: View Mir**
pub(crate) fn view_mir(db: &RootDatabase, position: FilePosition) -> String {
    body_mir(db, position).unwrap_or_else(|| "Not inside a function body".to_owned())
}

fn body_mir(db: &RootDatabase, position: FilePosition) -> Option<String> {
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
    Some(def.debug_mir(db))
}

#[cfg(test)]
mod tests {
    use crate::fixture;

    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        let (analysis, position) = fixture::position(ra_fixture);
        let mir = analysis.view_mir(position);
        assert!(mir.is_ok());
    }

    #[test]
    fn tuple_field_projection_uses_base_type() {
        check(
            r#"
//- minicore: sized
struct Inner { value: u32 }
fn test$0(value: (u32, Inner)) -> Inner { value.1 }
"#,
        );
    }

    #[test]
    fn nested_struct_field_projection_uses_base_type() {
        check(
            r#"
//- minicore: sized
struct Inner { value: u32 }
struct Outer { first: u32, second: Inner }
fn test$0(value: Outer) -> u32 { value.second.value }
"#,
        );
    }

    #[test]
    fn enum_field_projection_uses_base_variant() {
        check(
            r#"
//- minicore: sized
struct Inner { value: u32 }
enum Outer { Variant { first: u32, second: Inner } }
fn test$0(value: Outer) -> Inner {
    match value { Outer::Variant { second, .. } => second }
}
"#,
        );
    }
}
