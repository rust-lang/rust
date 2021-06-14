use either::Either;
use hir::{db::AstDatabase, InFile};
use ide_db::{assists::Assist, source_change::SourceChange};
use stdx::format_to;
use syntax::{algo, ast::make, AstNode, SyntaxNodePtr};
use text_edit::TextEdit;

use crate::{fix, Diagnostic, DiagnosticsContext};

// Diagnostic: missing-fields
//
// This diagnostic is triggered if record lacks some fields that exist in the corresponding structure.
//
// Example:
//
// ```rust
// struct A { a: u8, b: u8 }
//
// let a = A { a: 10 };
// ```
pub(crate) fn missing_fields(ctx: &DiagnosticsContext<'_>, d: &hir::MissingFields) -> Diagnostic {
    let mut message = String::from("missing structure fields:\n");
    for field in &d.missed_fields {
        format_to!(message, "- {}\n", field);
    }

    let ptr = InFile::new(
        d.file,
        d.field_list_parent_path
            .clone()
            .map(SyntaxNodePtr::from)
            .unwrap_or_else(|| d.field_list_parent.clone().either(|it| it.into(), |it| it.into())),
    );

    Diagnostic::new("missing-fields", message, ctx.sema.diagnostics_display_range(ptr).range)
        .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::MissingFields) -> Option<Vec<Assist>> {
    // Note that although we could add a diagnostics to
    // fill the missing tuple field, e.g :
    // `struct A(usize);`
    // `let a = A { 0: () }`
    // but it is uncommon usage and it should not be encouraged.
    if d.missed_fields.iter().any(|it| it.as_tuple_index().is_some()) {
        return None;
    }

    let root = ctx.sema.db.parse_or_expand(d.file)?;
    let field_list_parent = match &d.field_list_parent {
        Either::Left(record_expr) => record_expr.to_node(&root),
        // FIXE: patterns should be fixable as well.
        Either::Right(_) => return None,
    };
    let old_field_list = field_list_parent.record_expr_field_list()?;
    let new_field_list = old_field_list.clone_for_update();
    for f in d.missed_fields.iter() {
        let field =
            make::record_expr_field(make::name_ref(&f.to_string()), Some(make::expr_unit()))
                .clone_for_update();
        new_field_list.add_field(field);
    }

    let edit = {
        let mut builder = TextEdit::builder();
        algo::diff(old_field_list.syntax(), new_field_list.syntax()).into_text_edit(&mut builder);
        builder.finish()
    };
    Some(vec![fix(
        "fill_missing_fields",
        "Fill struct fields",
        SourceChange::from_text_edit(d.file.original_file(ctx.sema.db), edit),
        ctx.sema.original_range(field_list_parent.syntax()).range,
    )])
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn missing_record_pat_field_diagnostic() {
        check_diagnostics(
            r#"
struct S { foo: i32, bar: () }
fn baz(s: S) {
    let S { foo: _ } = s;
      //^ error: missing structure fields:
      //| - bar
}
"#,
        );
    }

    #[test]
    fn missing_record_pat_field_no_diagnostic_if_not_exhaustive() {
        check_diagnostics(
            r"
struct S { foo: i32, bar: () }
fn baz(s: S) -> i32 {
    match s {
        S { foo, .. } => foo,
    }
}
",
        )
    }

    #[test]
    fn missing_record_pat_field_box() {
        check_diagnostics(
            r"
struct S { s: Box<u32> }
fn x(a: S) {
    let S { box s } = a;
}
",
        )
    }

    #[test]
    fn missing_record_pat_field_ref() {
        check_diagnostics(
            r"
struct S { s: u32 }
fn x(a: S) {
    let S { ref s } = a;
}
",
        )
    }

    #[test]
    fn range_mapping_out_of_macros() {
        // FIXME: this is very wrong, but somewhat tricky to fix.
        check_fix(
            r#"
fn some() {}
fn items() {}
fn here() {}

macro_rules! id { ($($tt:tt)*) => { $($tt)*}; }

fn main() {
    let _x = id![Foo { a: $042 }];
}

pub struct Foo { pub a: i32, pub b: i32 }
"#,
            r#"
fn some(, b: () ) {}
fn items() {}
fn here() {}

macro_rules! id { ($($tt:tt)*) => { $($tt)*}; }

fn main() {
    let _x = id![Foo { a: 42 }];
}

pub struct Foo { pub a: i32, pub b: i32 }
"#,
        );
    }

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
        check_diagnostics(
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
        check_diagnostics(
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

    #[test]
    fn import_extern_crate_clash_with_inner_item() {
        // This is more of a resolver test, but doesn't really work with the hir_def testsuite.

        check_diagnostics(
            r#"
//- /lib.rs crate:lib deps:jwt
mod permissions;

use permissions::jwt;

fn f() {
    fn inner() {}
    jwt::Claims {}; // should resolve to the local one with 0 fields, and not get a diagnostic
}

//- /permissions.rs
pub mod jwt  {
    pub struct Claims {}
}

//- /jwt/lib.rs crate:jwt
pub struct Claims {
    field: u8,
}
        "#,
        );
    }
}
