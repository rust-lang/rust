use hir::{db::AstDatabase, diagnostics::IncorrectCase, InFile, Semantics};
use ide_assists::{Assist, AssistResolveStrategy};
use ide_db::{base_db::FilePosition, RootDatabase};
use syntax::AstNode;

use crate::{
    diagnostics::{unresolved_fix, DiagnosticWithFixes},
    references::rename::rename_with_semantics,
};

impl DiagnosticWithFixes for IncorrectCase {
    fn fixes(
        &self,
        sema: &Semantics<RootDatabase>,
        resolve: &AssistResolveStrategy,
    ) -> Option<Vec<Assist>> {
        let root = sema.db.parse_or_expand(self.file)?;
        let name_node = self.ident.to_node(&root);

        let name_node = InFile::new(self.file, name_node.syntax());
        let frange = name_node.original_file_range(sema.db);
        let file_position = FilePosition { file_id: frange.file_id, offset: frange.range.start() };

        let label = format!("Rename to {}", self.suggested_text);
        let mut res = unresolved_fix("change_case", &label, frange.range);
        if resolve.should_resolve(&res.id) {
            let source_change = rename_with_semantics(sema, file_position, &self.suggested_text);
            res.source_change = Some(source_change.ok().unwrap_or_default());
        }

        Some(vec![res])
    }
}

#[cfg(test)]
mod change_case {
    use crate::{
        diagnostics::tests::{check_fix, check_no_diagnostics},
        fixture, AssistResolveStrategy, DiagnosticsConfig,
    };

    #[test]
    fn test_rename_incorrect_case() {
        check_fix(
            r#"
pub struct test_struct$0 { one: i32 }

pub fn some_fn(val: test_struct) -> test_struct {
    test_struct { one: val.one + 1 }
}
"#,
            r#"
pub struct TestStruct { one: i32 }

pub fn some_fn(val: TestStruct) -> TestStruct {
    TestStruct { one: val.one + 1 }
}
"#,
        );

        check_fix(
            r#"
pub fn some_fn(NonSnakeCase$0: u8) -> u8 {
    NonSnakeCase
}
"#,
            r#"
pub fn some_fn(non_snake_case: u8) -> u8 {
    non_snake_case
}
"#,
        );

        check_fix(
            r#"
pub fn SomeFn$0(val: u8) -> u8 {
    if val != 0 { SomeFn(val - 1) } else { val }
}
"#,
            r#"
pub fn some_fn(val: u8) -> u8 {
    if val != 0 { some_fn(val - 1) } else { val }
}
"#,
        );

        check_fix(
            r#"
fn some_fn() {
    let whatAWeird_Formatting$0 = 10;
    another_func(whatAWeird_Formatting);
}
"#,
            r#"
fn some_fn() {
    let what_a_weird_formatting = 10;
    another_func(what_a_weird_formatting);
}
"#,
        );
    }

    #[test]
    fn test_uppercase_const_no_diagnostics() {
        check_no_diagnostics(
            r#"
fn foo() {
    const ANOTHER_ITEM$0: &str = "some_item";
}
"#,
        );
    }

    #[test]
    fn test_rename_incorrect_case_struct_method() {
        check_fix(
            r#"
pub struct TestStruct;

impl TestStruct {
    pub fn SomeFn$0() -> TestStruct {
        TestStruct
    }
}
"#,
            r#"
pub struct TestStruct;

impl TestStruct {
    pub fn some_fn() -> TestStruct {
        TestStruct
    }
}
"#,
        );
    }

    #[test]
    fn test_single_incorrect_case_diagnostic_in_function_name_issue_6970() {
        let input = r#"fn FOO$0() {}"#;
        let expected = r#"fn foo() {}"#;

        let (analysis, file_position) = fixture::position(input);
        let diagnostics = analysis
            .diagnostics(
                &DiagnosticsConfig::default(),
                AssistResolveStrategy::All,
                file_position.file_id,
            )
            .unwrap();
        assert_eq!(diagnostics.len(), 1);

        check_fix(input, expected);
    }
}
