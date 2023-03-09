use hir::db::AstDatabase;
use ide_db::{assists::Assist, source_change::SourceChange};
use syntax::ast::{ExprStmt, LetStmt};
use syntax::AstNode;
use syntax::{ast, SyntaxNode};
use text_edit::TextEdit;

use crate::{fix, Diagnostic, DiagnosticsContext};

// Diagnostic: missing-unsafe
//
// This diagnostic is triggered if an operation marked as `unsafe` is used outside of an `unsafe` function or block.
pub(crate) fn missing_unsafe(ctx: &DiagnosticsContext<'_>, d: &hir::MissingUnsafe) -> Diagnostic {
    Diagnostic::new(
        "missing-unsafe",
        "this operation is unsafe and requires an unsafe function or block",
        ctx.sema.diagnostics_display_range(d.expr.clone().map(|it| it.into())).range,
    )
    .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::MissingUnsafe) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.expr.file_id)?;
    let expr = d.expr.value.to_node(&root);

    let node_to_add_unsafe_block = pick_best_node_to_add_unsafe_block(ctx, &expr);

    let replacement = format!("unsafe {{ {} }}", node_to_add_unsafe_block.text());
    let edit = TextEdit::replace(node_to_add_unsafe_block.text_range(), replacement);
    let source_change =
        SourceChange::from_text_edit(d.expr.file_id.original_file(ctx.sema.db), edit);
    Some(vec![fix("add_unsafe", "Add unsafe block", source_change, expr.syntax().text_range())])
}

// Find the let statement or expression statement closest to the `expr` in the
// ancestor chain.
//
// Why don't we just add an unsafe block around the `expr`?
//
// Consider this example:
// ```
// STATIC_MUT += 1;
// ```
// We can't add an unsafe block to the left-hand side of an assignment.
// ```
// unsafe { STATIC_MUT } += 1;
// ```
//
// Or this example:
// ```
// let z = STATIC_MUT.a;
// ```
// We can't add an unsafe block like this:
// ```
// let z = unsafe { STATIC_MUT } .a;
// ```
fn pick_best_node_to_add_unsafe_block(
    ctx: &DiagnosticsContext<'_>,
    expr: &ast::Expr,
) -> SyntaxNode {
    let Some(let_or_expr_stmt) = ctx.sema.ancestors_with_macros(expr.syntax().clone()).find(|node| {
        LetStmt::can_cast(node.kind()) || ExprStmt::can_cast(node.kind())
    }) else {
        // Is this reachable?
        return expr.syntax().clone();
    };
    let_or_expr_stmt
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn missing_unsafe_diagnostic_with_raw_ptr() {
        check_diagnostics(
            r#"
fn main() {
    let x = &5 as *const usize;
    unsafe { let y = *x; }
    let z = *x;
}         //^^💡 error: this operation is unsafe and requires an unsafe function or block
"#,
        )
    }

    #[test]
    fn missing_unsafe_diagnostic_with_unsafe_call() {
        check_diagnostics(
            r#"
struct HasUnsafe;

impl HasUnsafe {
    unsafe fn unsafe_fn(&self) {
        let x = &5 as *const usize;
        let y = *x;
    }
}

unsafe fn unsafe_fn() {
    let x = &5 as *const usize;
    let y = *x;
}

fn main() {
    unsafe_fn();
  //^^^^^^^^^^^💡 error: this operation is unsafe and requires an unsafe function or block
    HasUnsafe.unsafe_fn();
  //^^^^^^^^^^^^^^^^^^^^^💡 error: this operation is unsafe and requires an unsafe function or block
    unsafe {
        unsafe_fn();
        HasUnsafe.unsafe_fn();
    }
}
"#,
        );
    }

    #[test]
    fn missing_unsafe_diagnostic_with_static_mut() {
        check_diagnostics(
            r#"
struct Ty {
    a: u8,
}

static mut STATIC_MUT: Ty = Ty { a: 0 };

fn main() {
    let x = STATIC_MUT.a;
          //^^^^^^^^^^💡 error: this operation is unsafe and requires an unsafe function or block
    unsafe {
        let x = STATIC_MUT.a;
    }
}
"#,
        );
    }

    #[test]
    fn no_missing_unsafe_diagnostic_with_safe_intrinsic() {
        check_diagnostics(
            r#"
extern "rust-intrinsic" {
    #[rustc_safe_intrinsic]
    pub fn bitreverse(x: u32) -> u32; // Safe intrinsic
    pub fn floorf32(x: f32) -> f32; // Unsafe intrinsic
}

fn main() {
    let _ = bitreverse(12);
    let _ = floorf32(12.0);
          //^^^^^^^^^^^^^^💡 error: this operation is unsafe and requires an unsafe function or block
}
"#,
        );
    }

    #[test]
    fn add_unsafe_block_when_dereferencing_a_raw_pointer() {
        check_fix(
            r#"
fn main() {
    let x = &5 as *const usize;
    let z = *x$0;
}
"#,
            r#"
fn main() {
    let x = &5 as *const usize;
    unsafe { let z = *x; }
}
"#,
        );
    }

    #[test]
    fn add_unsafe_block_when_calling_unsafe_function() {
        check_fix(
            r#"
unsafe fn func() {
    let x = &5 as *const usize;
    let z = *x;
}
fn main() {
    func$0();
}
"#,
            r#"
unsafe fn func() {
    let x = &5 as *const usize;
    let z = *x;
}
fn main() {
    unsafe { func(); }
}
"#,
        )
    }

    #[test]
    fn add_unsafe_block_when_calling_unsafe_method() {
        check_fix(
            r#"
struct S(usize);
impl S {
    unsafe fn func(&self) {
        let x = &self.0 as *const usize;
        let z = *x;
    }
}
fn main() {
    let s = S(5);
    s.func$0();
}
"#,
            r#"
struct S(usize);
impl S {
    unsafe fn func(&self) {
        let x = &self.0 as *const usize;
        let z = *x;
    }
}
fn main() {
    let s = S(5);
    unsafe { s.func(); }
}
"#,
        )
    }

    #[test]
    fn add_unsafe_block_when_accessing_mutable_static() {
        check_fix(
            r#"
struct Ty {
    a: u8,
}

static mut STATIC_MUT: Ty = Ty { a: 0 };

fn main() {
    let x = STATIC_MUT$0.a;
}
"#,
            r#"
struct Ty {
    a: u8,
}

static mut STATIC_MUT: Ty = Ty { a: 0 };

fn main() {
    unsafe { let x = STATIC_MUT.a; }
}
"#,
        )
    }

    #[test]
    fn add_unsafe_block_when_calling_unsafe_intrinsic() {
        check_fix(
            r#"
extern "rust-intrinsic" {
    pub fn floorf32(x: f32) -> f32;
}

fn main() {
    let _ = floorf32$0(12.0);
}
"#,
            r#"
extern "rust-intrinsic" {
    pub fn floorf32(x: f32) -> f32;
}

fn main() {
    unsafe { let _ = floorf32(12.0); }
}
"#,
        )
    }
}
