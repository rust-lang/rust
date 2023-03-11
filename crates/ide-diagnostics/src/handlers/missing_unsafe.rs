use hir::db::AstDatabase;
use ide_db::{assists::Assist, source_change::SourceChange};
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

    let node_to_add_unsafe_block = pick_best_node_to_add_unsafe_block(&expr);

    let replacement = format!("unsafe {{ {} }}", node_to_add_unsafe_block.text());
    let edit = TextEdit::replace(node_to_add_unsafe_block.text_range(), replacement);
    let source_change =
        SourceChange::from_text_edit(d.expr.file_id.original_file(ctx.sema.db), edit);
    Some(vec![fix("add_unsafe", "Add unsafe block", source_change, expr.syntax().text_range())])
}

// Pick the first ancestor expression of the unsafe `expr` that is not a
// receiver of a method call, a field access, the left-hand side of an
// assignment, or a reference. As all of those cases would incur a forced move
// if wrapped which might not be wanted. That is:
// - `unsafe_expr.foo` -> `unsafe { unsafe_expr.foo }`
// - `unsafe_expr.foo.bar` -> `unsafe { unsafe_expr.foo.bar }`
// - `unsafe_expr.foo()` -> `unsafe { unsafe_expr.foo() }`
// - `unsafe_expr.foo.bar()` -> `unsafe { unsafe_expr.foo.bar() }`
// - `unsafe_expr += 1` -> `unsafe { unsafe_expr += 1 }`
// - `&unsafe_expr` -> `unsafe { &unsafe_expr }`
// - `&&unsafe_expr` -> `unsafe { &&unsafe_expr }`
fn pick_best_node_to_add_unsafe_block(unsafe_expr: &ast::Expr) -> SyntaxNode {
    // The `unsafe_expr` might be:
    // - `ast::CallExpr`: call an unsafe function
    // - `ast::MethodCallExpr`: call an unsafe method
    // - `ast::PrefixExpr`: dereference a raw pointer
    // - `ast::PathExpr`: access a static mut variable
    for node in unsafe_expr.syntax().ancestors() {
        let Some(parent) = node.parent() else {
            return node;
        };
        match parent.kind() {
            syntax::SyntaxKind::METHOD_CALL_EXPR => {
                // Check if the `node` is the receiver of the method call
                let method_call_expr = ast::MethodCallExpr::cast(parent.clone()).unwrap();
                if method_call_expr
                    .receiver()
                    .map(|receiver| {
                        receiver.syntax().text_range().contains_range(node.text_range())
                    })
                    .unwrap_or(false)
                {
                    // Actually, I think it's not necessary to check whether the
                    // text range of the `node` (which is the ancestor of the
                    // `unsafe_expr`) is contained in the text range of the
                    // receiver. The `node` could potentially be the receiver, the
                    // method name, or the argument list. Since the `node` is the
                    // ancestor of the unsafe_expr, it cannot be the method name.
                    // Additionally, if the `node` is the argument list, the loop
                    // would break at least when `parent` reaches the argument list.
                    //
                    // Dispite this, I still check the text range because I think it
                    // makes the code easier to understand.
                    continue;
                }
                return node;
            }
            syntax::SyntaxKind::FIELD_EXPR | syntax::SyntaxKind::REF_EXPR => continue,
            syntax::SyntaxKind::BIN_EXPR => {
                // Check if the `node` is the left-hand side of an assignment
                let is_left_hand_side_of_assignment = {
                    let bin_expr = ast::BinExpr::cast(parent.clone()).unwrap();
                    if let Some(ast::BinaryOp::Assignment { .. }) = bin_expr.op_kind() {
                        let is_left_hand_side = bin_expr
                            .lhs()
                            .map(|lhs| lhs.syntax().text_range().contains_range(node.text_range()))
                            .unwrap_or(false);
                        is_left_hand_side
                    } else {
                        false
                    }
                };
                if !is_left_hand_side_of_assignment {
                    return node;
                }
            }
            _ => {
                return node;
            }
        }
    }
    unsafe_expr.syntax().clone()
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
    let z = unsafe { *x };
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
    unsafe { func() };
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
    unsafe { s.func() };
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
    let x = unsafe { STATIC_MUT.a };
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
    let _ = unsafe { floorf32(12.0) };
}
"#,
        )
    }

    #[test]
    fn unsafe_expr_as_a_receiver_of_a_method_call() {
        check_fix(
            r#"
unsafe fn foo() -> String {
    "string".to_string()
}

fn main() {
    foo$0().len();
}
"#,
            r#"
unsafe fn foo() -> String {
    "string".to_string()
}

fn main() {
    unsafe { foo().len() };
}
"#,
        )
    }

    #[test]
    fn unsafe_expr_as_an_argument_of_a_method_call() {
        check_fix(
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let mut v = vec![];
    v.push(STATIC_MUT$0);
}
"#,
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let mut v = vec![];
    v.push(unsafe { STATIC_MUT });
}
"#,
        )
    }

    #[test]
    fn unsafe_expr_as_left_hand_side_of_assignment() {
        check_fix(
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    STATIC_MUT$0 = 1;
}
"#,
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    unsafe { STATIC_MUT = 1 };
}
"#,
        )
    }

    #[test]
    fn unsafe_expr_as_right_hand_side_of_assignment() {
        check_fix(
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let x;
    x = STATIC_MUT$0;
}
"#,
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let x;
    x = unsafe { STATIC_MUT };
}
"#,
        )
    }

    #[test]
    fn unsafe_expr_in_binary_plus() {
        check_fix(
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let x = STATIC_MUT$0 + 1;
}
"#,
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let x = unsafe { STATIC_MUT } + 1;
}
"#,
        )
    }

    #[test]
    fn ref_to_unsafe_expr() {
        check_fix(
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let x = &STATIC_MUT$0;
}
"#,
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let x = unsafe { &STATIC_MUT };
}
"#,
        )
    }

    #[test]
    fn ref_ref_to_unsafe_expr() {
        check_fix(
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let x = &&STATIC_MUT$0;
}
"#,
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let x = unsafe { &&STATIC_MUT };
}
"#,
        )
    }
}
