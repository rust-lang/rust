use hir::{db::AstDatabase, diagnostics::MissingOkOrSomeInTailExpr, Semantics};
use ide_assists::{Assist, AssistResolveStrategy};
use ide_db::{source_change::SourceChange, RootDatabase};
use syntax::AstNode;
use text_edit::TextEdit;

use crate::diagnostics::{fix, DiagnosticWithFixes};

impl DiagnosticWithFixes for MissingOkOrSomeInTailExpr {
    fn fixes(
        &self,
        sema: &Semantics<RootDatabase>,
        _resolve: &AssistResolveStrategy,
    ) -> Option<Vec<Assist>> {
        let root = sema.db.parse_or_expand(self.file)?;
        let tail_expr = self.expr.to_node(&root);
        let tail_expr_range = tail_expr.syntax().text_range();
        let replacement = format!("{}({})", self.required, tail_expr.syntax());
        let edit = TextEdit::replace(tail_expr_range, replacement);
        let source_change = SourceChange::from_text_edit(self.file.original_file(sema.db), edit);
        let name = if self.required == "Ok" { "Wrap with Ok" } else { "Wrap with Some" };
        Some(vec![fix("wrap_tail_expr", name, source_change, tail_expr_range)])
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::{check_fix, check_no_diagnostics};

    #[test]
    fn test_wrap_return_type_option() {
        check_fix(
            r#"
//- /main.rs crate:main deps:core
use core::option::Option::{self, Some, None};

fn div(x: i32, y: i32) -> Option<i32> {
    if y == 0 {
        return None;
    }
    x / y$0
}
//- /core/lib.rs crate:core
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
pub mod option {
    pub enum Option<T> { Some(T), None }
}
"#,
            r#"
use core::option::Option::{self, Some, None};

fn div(x: i32, y: i32) -> Option<i32> {
    if y == 0 {
        return None;
    }
    Some(x / y)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type() {
        check_fix(
            r#"
//- /main.rs crate:main deps:core
use core::result::Result::{self, Ok, Err};

fn div(x: i32, y: i32) -> Result<i32, ()> {
    if y == 0 {
        return Err(());
    }
    x / y$0
}
//- /core/lib.rs crate:core
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
pub mod option {
    pub enum Option<T> { Some(T), None }
}
"#,
            r#"
use core::result::Result::{self, Ok, Err};

fn div(x: i32, y: i32) -> Result<i32, ()> {
    if y == 0 {
        return Err(());
    }
    Ok(x / y)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_handles_generic_functions() {
        check_fix(
            r#"
//- /main.rs crate:main deps:core
use core::result::Result::{self, Ok, Err};

fn div<T>(x: T) -> Result<T, i32> {
    if x == 0 {
        return Err(7);
    }
    $0x
}
//- /core/lib.rs crate:core
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
pub mod option {
    pub enum Option<T> { Some(T), None }
}
"#,
            r#"
use core::result::Result::{self, Ok, Err};

fn div<T>(x: T) -> Result<T, i32> {
    if x == 0 {
        return Err(7);
    }
    Ok(x)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_handles_type_aliases() {
        check_fix(
            r#"
//- /main.rs crate:main deps:core
use core::result::Result::{self, Ok, Err};

type MyResult<T> = Result<T, ()>;

fn div(x: i32, y: i32) -> MyResult<i32> {
    if y == 0 {
        return Err(());
    }
    x $0/ y
}
//- /core/lib.rs crate:core
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
pub mod option {
    pub enum Option<T> { Some(T), None }
}
"#,
            r#"
use core::result::Result::{self, Ok, Err};

type MyResult<T> = Result<T, ()>;

fn div(x: i32, y: i32) -> MyResult<i32> {
    if y == 0 {
        return Err(());
    }
    Ok(x / y)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_not_applicable_when_expr_type_does_not_match_ok_type() {
        check_no_diagnostics(
            r#"
//- /main.rs crate:main deps:core
use core::result::Result::{self, Ok, Err};

fn foo() -> Result<(), i32> { 0 }

//- /core/lib.rs crate:core
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
pub mod option {
    pub enum Option<T> { Some(T), None }
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_not_applicable_when_return_type_is_not_result_or_option() {
        check_no_diagnostics(
            r#"
//- /main.rs crate:main deps:core
use core::result::Result::{self, Ok, Err};

enum SomeOtherEnum { Ok(i32), Err(String) }

fn foo() -> SomeOtherEnum { 0 }

//- /core/lib.rs crate:core
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
pub mod option {
    pub enum Option<T> { Some(T), None }
}
"#,
        );
    }
}
