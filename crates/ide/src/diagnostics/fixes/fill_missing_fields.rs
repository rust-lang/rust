use hir::{db::AstDatabase, diagnostics::MissingFields, Semantics};
use ide_assists::AssistResolveStrategy;
use ide_db::{source_change::SourceChange, RootDatabase};
use syntax::{algo, ast::make, AstNode};
use text_edit::TextEdit;

use crate::{
    diagnostics::{fix, fixes::DiagnosticWithFixes},
    Assist,
};

impl DiagnosticWithFixes for MissingFields {
    fn fixes(
        &self,
        sema: &Semantics<RootDatabase>,
        _resolve: &AssistResolveStrategy,
    ) -> Option<Vec<Assist>> {
        // Note that although we could add a diagnostics to
        // fill the missing tuple field, e.g :
        // `struct A(usize);`
        // `let a = A { 0: () }`
        // but it is uncommon usage and it should not be encouraged.
        if self.missed_fields.iter().any(|it| it.as_tuple_index().is_some()) {
            return None;
        }

        let root = sema.db.parse_or_expand(self.file)?;
        let field_list_parent = self.field_list_parent.to_node(&root);
        let old_field_list = field_list_parent.record_expr_field_list()?;
        let new_field_list = old_field_list.clone_for_update();
        for f in self.missed_fields.iter() {
            let field =
                make::record_expr_field(make::name_ref(&f.to_string()), Some(make::expr_unit()))
                    .clone_for_update();
            new_field_list.add_field(field);
        }

        let edit = {
            let mut builder = TextEdit::builder();
            algo::diff(&old_field_list.syntax(), &new_field_list.syntax())
                .into_text_edit(&mut builder);
            builder.finish()
        };
        Some(vec![fix(
            "fill_missing_fields",
            "Fill struct fields",
            SourceChange::from_text_edit(self.file.original_file(sema.db), edit),
            sema.original_range(&field_list_parent.syntax()).range,
        )])
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::{check_fix, check_no_diagnostics};

    #[test]
    fn test_fill_struct_fields_empty() {
        check_fix(
            r#"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let s = TestStruct {$0};
}
"#,
            r#"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let s = TestStruct { one: (), two: () };
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_self() {
        check_fix(
            r#"
struct TestStruct { one: i32 }

impl TestStruct {
    fn test_fn() { let s = Self {$0}; }
}
"#,
            r#"
struct TestStruct { one: i32 }

impl TestStruct {
    fn test_fn() { let s = Self { one: () }; }
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_enum() {
        check_fix(
            r#"
enum Expr {
    Bin { lhs: Box<Expr>, rhs: Box<Expr> }
}

impl Expr {
    fn new_bin(lhs: Box<Expr>, rhs: Box<Expr>) -> Expr {
        Expr::Bin {$0 }
    }
}
"#,
            r#"
enum Expr {
    Bin { lhs: Box<Expr>, rhs: Box<Expr> }
}

impl Expr {
    fn new_bin(lhs: Box<Expr>, rhs: Box<Expr>) -> Expr {
        Expr::Bin { lhs: (), rhs: () }
    }
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_partial() {
        check_fix(
            r#"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let s = TestStruct{ two: 2$0 };
}
"#,
            r"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let s = TestStruct{ two: 2, one: () };
}
",
        );
    }

    #[test]
    fn test_fill_struct_fields_raw_ident() {
        check_fix(
            r#"
struct TestStruct { r#type: u8 }

fn test_fn() {
    TestStruct { $0 };
}
"#,
            r"
struct TestStruct { r#type: u8 }

fn test_fn() {
    TestStruct { r#type: ()  };
}
",
        );
    }

    #[test]
    fn test_fill_struct_fields_no_diagnostic() {
        check_no_diagnostics(
            r#"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let one = 1;
    let s = TestStruct{ one, two: 2 };
}
        "#,
        );
    }

    #[test]
    fn test_fill_struct_fields_no_diagnostic_on_spread() {
        check_no_diagnostics(
            r#"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let one = 1;
    let s = TestStruct{ ..a };
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_blank_line() {
        check_fix(
            r#"
struct S { a: (), b: () }

fn f() {
    S {
        $0
    };
}
"#,
            r#"
struct S { a: (), b: () }

fn f() {
    S {
        a: (),
        b: (),
    };
}
"#,
        );
    }
}
