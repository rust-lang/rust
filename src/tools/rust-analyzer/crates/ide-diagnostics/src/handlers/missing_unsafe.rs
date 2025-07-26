use hir::db::ExpandDatabase;
use hir::{UnsafeLint, UnsafetyReason};
use ide_db::text_edit::TextEdit;
use ide_db::{assists::Assist, source_change::SourceChange};
use syntax::{AstNode, match_ast};
use syntax::{SyntaxNode, ast};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, fix};

// Diagnostic: missing-unsafe
//
// This diagnostic is triggered if an operation marked as `unsafe` is used outside of an `unsafe` function or block.
pub(crate) fn missing_unsafe(ctx: &DiagnosticsContext<'_>, d: &hir::MissingUnsafe) -> Diagnostic {
    let code = match d.lint {
        UnsafeLint::HardError => DiagnosticCode::RustcHardError("E0133"),
        UnsafeLint::UnsafeOpInUnsafeFn => DiagnosticCode::RustcLint("unsafe_op_in_unsafe_fn"),
        UnsafeLint::DeprecatedSafe2024 => DiagnosticCode::RustcLint("deprecated_safe_2024"),
    };
    let operation = display_unsafety_reason(d.reason);
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        code,
        format!("{operation} is unsafe and requires an unsafe function or block"),
        d.node.map(|it| it.into()),
    )
    .stable()
    .with_fixes(fixes(ctx, d))
}

fn display_unsafety_reason(reason: UnsafetyReason) -> &'static str {
    match reason {
        UnsafetyReason::UnionField => "access to union field",
        UnsafetyReason::UnsafeFnCall => "call to unsafe function",
        UnsafetyReason::InlineAsm => "use of inline assembly",
        UnsafetyReason::RawPtrDeref => "dereference of raw pointer",
        UnsafetyReason::MutableStatic => "use of mutable static",
        UnsafetyReason::ExternStatic => "use of extern static",
    }
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::MissingUnsafe) -> Option<Vec<Assist>> {
    // The fixit will not work correctly for macro expansions, so we don't offer it in that case.
    if d.node.file_id.is_macro() {
        return None;
    }

    let root = ctx.sema.db.parse_or_expand(d.node.file_id);
    let node = d.node.value.to_node(&root);
    let expr = node.syntax().ancestors().find_map(ast::Expr::cast)?;

    let node_to_add_unsafe_block = pick_best_node_to_add_unsafe_block(&expr)?;

    let replacement = format!("unsafe {{ {} }}", node_to_add_unsafe_block.text());
    let edit = TextEdit::replace(node_to_add_unsafe_block.text_range(), replacement);
    let source_change = SourceChange::from_text_edit(
        d.node.file_id.original_file(ctx.sema.db).file_id(ctx.sema.db),
        edit,
    );
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
fn pick_best_node_to_add_unsafe_block(unsafe_expr: &ast::Expr) -> Option<SyntaxNode> {
    // The `unsafe_expr` might be:
    // - `ast::CallExpr`: call an unsafe function
    // - `ast::MethodCallExpr`: call an unsafe method
    // - `ast::PrefixExpr`: dereference a raw pointer
    // - `ast::PathExpr`: access a static mut variable
    for (node, parent) in
        unsafe_expr.syntax().ancestors().zip(unsafe_expr.syntax().ancestors().skip(1))
    {
        match_ast! {
            match parent {
                // If the `parent` is a `MethodCallExpr`, that means the `node`
                // is the receiver of the method call, because only the receiver
                // can be a direct child of a method call. The method name
                // itself is not an expression but a `NameRef`, and an argument
                // is a direct child of an `ArgList`.
                ast::MethodCallExpr(_) => continue,
                ast::FieldExpr(_) => continue,
                ast::RefExpr(_) => continue,
                ast::BinExpr(it) => {
                    // Check if the `node` is the left-hand side of an
                    // assignment, if so, we don't want to wrap it in an unsafe
                    // block, e.g. `unsafe_expr += 1`
                    let is_left_hand_side_of_assignment = {
                        if let Some(ast::BinaryOp::Assignment { .. }) = it.op_kind() {
                            it.lhs().map(|lhs| lhs.syntax().text_range().contains_range(node.text_range())).unwrap_or(false)
                        } else {
                            false
                        }
                    };
                    if !is_left_hand_side_of_assignment {
                        return Some(node);
                    }
                },
                _ => { return Some(node); }

            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix, check_no_fix};

    #[test]
    fn missing_unsafe_diagnostic_with_raw_ptr() {
        check_diagnostics(
            r#"
//- minicore: sized
fn main() {
    let x = &5_usize as *const usize;
    unsafe { let _y = *x; }
    let _z = *x;
}          //^^💡 error: dereference of raw pointer is unsafe and requires an unsafe function or block
"#,
        )
    }

    #[test]
    fn missing_unsafe_diagnostic_with_unsafe_call() {
        check_diagnostics(
            r#"
//- minicore: sized
struct HasUnsafe;

impl HasUnsafe {
    unsafe fn unsafe_fn(&self) {
        let x = &5_usize as *const usize;
        let _y = unsafe {*x};
    }
}

unsafe fn unsafe_fn() {
    let x = &5_usize as *const usize;
    let _y = unsafe {*x};
}

fn main() {
    unsafe_fn();
  //^^^^^^^^^^^💡 error: call to unsafe function is unsafe and requires an unsafe function or block
    HasUnsafe.unsafe_fn();
  //^^^^^^^^^^^^^^^^^^^^^💡 error: call to unsafe function is unsafe and requires an unsafe function or block
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
//- minicore: copy

struct Ty {
    a: u8,
}

static mut STATIC_MUT: Ty = Ty { a: 0 };

fn main() {
    let _x = STATIC_MUT.a;
           //^^^^^^^^^^💡 error: use of mutable static is unsafe and requires an unsafe function or block
    unsafe {
        let _x = STATIC_MUT.a;
    }
}
"#,
        );
    }

    #[test]
    fn missing_unsafe_diagnostic_with_extern_static() {
        check_diagnostics(
            r#"
//- minicore: copy

extern "C" {
    static EXTERN: i32;
    static mut EXTERN_MUT: i32;
}

fn main() {
    let _x = EXTERN;
           //^^^^^^💡 error: use of extern static is unsafe and requires an unsafe function or block
    let _x = EXTERN_MUT;
           //^^^^^^^^^^💡 error: use of mutable static is unsafe and requires an unsafe function or block
    unsafe {
        let _x = EXTERN;
        let _x = EXTERN_MUT;
    }
}
"#,
        );
    }

    #[test]
    fn no_unsafe_diagnostic_with_addr_of_static() {
        check_diagnostics(
            r#"
//- minicore: copy, addr_of

use core::ptr::{addr_of, addr_of_mut};

extern "C" {
    static EXTERN: i32;
    static mut EXTERN_MUT: i32;
}
static mut STATIC_MUT: i32 = 0;

fn main() {
    let _x = addr_of!(EXTERN);
    let _x = addr_of!(EXTERN_MUT);
    let _x = addr_of!(STATIC_MUT);
    let _x = addr_of_mut!(EXTERN_MUT);
    let _x = addr_of_mut!(STATIC_MUT);
}
"#,
        );
    }

    #[test]
    fn no_missing_unsafe_diagnostic_with_safe_intrinsic() {
        check_diagnostics(
            r#"
#[rustc_intrinsic]
pub fn bitreverse(x: u32) -> u32; // Safe intrinsic
#[rustc_intrinsic]
pub unsafe fn floorf32(x: f32) -> f32; // Unsafe intrinsic

fn main() {
    let _ = bitreverse(12);
    let _ = floorf32(12.0);
          //^^^^^^^^^^^^^^💡 error: call to unsafe function is unsafe and requires an unsafe function or block
}
"#,
        );
    }

    #[test]
    fn no_missing_unsafe_diagnostic_with_legacy_safe_intrinsic() {
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
          //^^^^^^^^^^^^^^💡 error: call to unsafe function is unsafe and requires an unsafe function or block
}
"#,
        );
    }

    #[test]
    fn no_missing_unsafe_diagnostic_with_deprecated_safe_2024() {
        check_diagnostics(
            r#"
#[rustc_deprecated_safe_2024]
fn set_var() {}

fn main() {
    set_var();
}
"#,
        );
    }

    #[test]
    fn add_unsafe_block_when_dereferencing_a_raw_pointer() {
        check_fix(
            r#"
//- minicore: sized
fn main() {
    let x = &5_usize as *const usize;
    let _z = *x$0;
}
"#,
            r#"
fn main() {
    let x = &5_usize as *const usize;
    let _z = unsafe { *x };
}
"#,
        );
    }

    #[test]
    fn add_unsafe_block_when_calling_unsafe_function() {
        check_fix(
            r#"
//- minicore: sized
unsafe fn func() {
    let x = &5_usize as *const usize;
    let z = *x;
}
fn main() {
    func$0();
}
"#,
            r#"
unsafe fn func() {
    let x = &5_usize as *const usize;
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
//- minicore: sized
struct S(usize);
impl S {
    unsafe fn func(&self) {
        let x = &self.0 as *const usize;
        let _z = unsafe { *x };
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
        let _z = unsafe { *x };
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
//- minicore: copy
struct Ty {
    a: u8,
}

static mut STATIC_MUT: Ty = Ty { a: 0 };

fn main() {
    let _x = STATIC_MUT$0.a;
}
"#,
            r#"
struct Ty {
    a: u8,
}

static mut STATIC_MUT: Ty = Ty { a: 0 };

fn main() {
    let _x = unsafe { STATIC_MUT.a };
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
//- minicore: copy
static mut STATIC_MUT: u8 = 0;

fn main() {
    let _x;
    _x = STATIC_MUT$0;
}
"#,
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let _x;
    _x = unsafe { STATIC_MUT };
}
"#,
        )
    }

    #[test]
    fn unsafe_expr_in_binary_plus() {
        check_fix(
            r#"
//- minicore: copy
static mut STATIC_MUT: u8 = 0;

fn main() {
    let _x = STATIC_MUT$0 + 1;
}
"#,
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let _x = unsafe { STATIC_MUT } + 1;
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
    let _x = &STATIC_MUT$0;
}
"#,
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let _x = unsafe { &STATIC_MUT };
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
    let _x = &&STATIC_MUT$0;
}
"#,
            r#"
static mut STATIC_MUT: u8 = 0;

fn main() {
    let _x = unsafe { &&STATIC_MUT };
}
"#,
        )
    }

    #[test]
    fn unsafe_expr_in_macro_call() {
        check_no_fix(
            r#"
unsafe fn foo() -> u8 {
    0
}

fn main() {
    let x = format!("foo: {}", foo$0());
}
            "#,
        )
    }

    #[test]
    fn rustc_deprecated_safe_2024() {
        check_diagnostics(
            r#"
//- /ed2021.rs crate:ed2021 edition:2021
#[rustc_deprecated_safe_2024]
unsafe fn deprecated_safe() -> u8 {
    0
}

//- /ed2024.rs crate:ed2024 edition:2024
#[rustc_deprecated_safe_2024]
unsafe fn deprecated_safe() -> u8 {
    0
}

//- /dep1.rs crate:dep1 deps:ed2021,ed2024 edition:2021
fn main() {
    ed2021::deprecated_safe();
    ed2024::deprecated_safe();
}

//- /dep2.rs crate:dep2 deps:ed2021,ed2024 edition:2024
fn main() {
    ed2021::deprecated_safe();
 // ^^^^^^^^^^^^^^^^^^^^^^^^^💡 error: call to unsafe function is unsafe and requires an unsafe function or block
    ed2024::deprecated_safe();
 // ^^^^^^^^^^^^^^^^^^^^^^^^^💡 error: call to unsafe function is unsafe and requires an unsafe function or block
}

//- /dep3.rs crate:dep3 deps:ed2021,ed2024 edition:2021
#![warn(deprecated_safe)]

fn main() {
    ed2021::deprecated_safe();
 // ^^^^^^^^^^^^^^^^^^^^^^^^^💡 warn: call to unsafe function is unsafe and requires an unsafe function or block
    ed2024::deprecated_safe();
 // ^^^^^^^^^^^^^^^^^^^^^^^^^💡 warn: call to unsafe function is unsafe and requires an unsafe function or block
}
            "#,
        )
    }

    #[test]
    fn orphan_unsafe_format_args() {
        // Checks that we don't place orphan arguments for formatting under an unsafe block.
        check_diagnostics(
            r#"
//- minicore: fmt_before_1_89_0
fn foo() {
    let p = 0xDEADBEEF as *const i32;
    format_args!("", *p);
                  // ^^ error: dereference of raw pointer is unsafe and requires an unsafe function or block
}
        "#,
        );

        check_diagnostics(
            r#"
//- minicore: fmt
fn foo() {
    let p = 0xDEADBEEF as *const i32;
    format_args!("", *p);
                  // ^^ error: dereference of raw pointer is unsafe and requires an unsafe function or block
}
        "#,
        );
    }

    #[test]
    fn unsafe_op_in_unsafe_fn_allowed_by_default_in_edition_2021() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo edition:2021
unsafe fn foo(p: *mut i32) {
    *p = 123;
}
            "#,
        );
        check_diagnostics(
            r#"
//- /lib.rs crate:foo edition:2021
#![deny(warnings)]
unsafe fn foo(p: *mut i32) {
    *p = 123;
}
            "#,
        );
    }

    #[test]
    fn unsafe_op_in_unsafe_fn_warn_by_default_in_edition_2024() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo edition:2024
unsafe fn foo(p: *mut i32) {
    *p = 123;
  //^^💡 warn: dereference of raw pointer is unsafe and requires an unsafe function or block
}
            "#,
        );
        check_diagnostics(
            r#"
//- /lib.rs crate:foo edition:2024
#![deny(warnings)]
unsafe fn foo(p: *mut i32) {
    *p = 123;
  //^^💡 error: dereference of raw pointer is unsafe and requires an unsafe function or block
}
            "#,
        );
    }

    #[test]
    fn unsafe_op_in_unsafe_fn() {
        check_diagnostics(
            r#"
#![warn(unsafe_op_in_unsafe_fn)]
unsafe fn foo(p: *mut i32) {
    *p = 123;
  //^^💡 warn: dereference of raw pointer is unsafe and requires an unsafe function or block
}
            "#,
        )
    }

    #[test]
    fn no_unsafe_diagnostic_with_safe_kw() {
        check_diagnostics(
            r#"
unsafe extern {
    pub safe fn f();

    pub unsafe fn g();

    pub fn h();

    pub safe static S1: i32;

    pub unsafe static S2: i32;

    pub static S3: i32;
}

fn main() {
    f();
    g();
  //^^^💡 error: call to unsafe function is unsafe and requires an unsafe function or block
    h();
  //^^^💡 error: call to unsafe function is unsafe and requires an unsafe function or block

    let _ = S1;
    let _ = S2;
          //^^💡 error: use of extern static is unsafe and requires an unsafe function or block
    let _ = S3;
          //^^💡 error: use of extern static is unsafe and requires an unsafe function or block
}
"#,
        );
    }

    #[test]
    fn no_unsafe_diagnostic_when_destructuring_union_with_wildcard() {
        check_diagnostics(
            r#"
union Union { field: i32 }
fn foo(v: &Union) {
    let Union { field: _ } = v;
    let Union { field: _ | _ } = v;
    Union { field: _ } = *v;
}
"#,
        );
    }

    #[test]
    fn union_destructuring() {
        check_diagnostics(
            r#"
union Union { field: u8 }
fn foo(v @ Union { field: _field }: &Union) {
                       // ^^^^^^ error: access to union field is unsafe and requires an unsafe function or block
    let Union { mut field } = v;
             // ^^^^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    let Union { field: 0..=255 } = v;
                    // ^^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    let Union { field: 0
                    // ^💡 error: access to union field is unsafe and requires an unsafe function or block
        | 1..=255 } = v;
       // ^^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    Union { field } = *v;
         // ^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    match v {
        Union { field: _field } => {}
                    // ^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    }
    if let Union { field: _field } = v {}
                       // ^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    (|&Union { field }| { _ = field; })(v);
            // ^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
}
"#,
        );
    }

    #[test]
    fn union_field_access() {
        check_diagnostics(
            r#"
union Union { field: u8 }
fn foo(v: &Union) {
    v.field;
 // ^^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
}
"#,
        );
    }

    #[test]
    fn inline_asm() {
        check_diagnostics(
            r#"
//- minicore: asm
fn foo() {
    core::arch::asm!("");
                 // ^^^^ error: use of inline assembly is unsafe and requires an unsafe function or block
}
"#,
        );
    }

    #[test]
    fn unsafe_op_in_unsafe_fn_dismissed_in_signature() {
        check_diagnostics(
            r#"
#![warn(unsafe_op_in_unsafe_fn)]
union Union { field: u32 }
unsafe fn foo(Union { field: _field }: Union) {}
            "#,
        )
    }

    #[test]
    fn union_assignment_allowed() {
        check_diagnostics(
            r#"
union Union { field: u32 }
fn foo(mut v: Union) {
    v.field = 123;
    (v.field,) = (123,);
    *&mut v.field = 123;
       // ^^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
}
struct Struct { field: u32 }
union Union2 { field: Struct }
fn bar(mut v: Union2) {
    v.field.field = 123;
}

            "#,
        )
    }

    #[test]
    fn raw_ref_reborrow_is_safe() {
        check_diagnostics(
            r#"
fn main() {
    let ptr: *mut i32;
    let _addr = &raw const *ptr;

    let local = 1;
    let ptr = &local as *const i32;
    let _addr = &raw const *ptr;
}
"#,
        )
    }

    #[test]
    fn target_feature() {
        check_diagnostics(
            r#"
#[target_feature(enable = "avx")]
fn foo() {}

#[target_feature(enable = "avx2")]
fn bar() {
    foo();
}

fn baz() {
    foo();
 // ^^^^^ 💡 error: call to unsafe function is unsafe and requires an unsafe function or block
}
        "#,
        );
    }

    #[test]
    fn unsafe_fn_ptr_call() {
        check_diagnostics(
            r#"
fn f(it: unsafe fn()){
    it();
 // ^^^^ 💡 error: call to unsafe function is unsafe and requires an unsafe function or block
}
        "#,
        );
    }

    #[test]
    fn unsafe_call_in_const_expr() {
        check_diagnostics(
            r#"
unsafe fn f() {}
fn main() {
    const { f(); };
         // ^^^ 💡 error: call to unsafe function is unsafe and requires an unsafe function or block
}
        "#,
        );
    }

    #[test]
    fn asm_label() {
        check_diagnostics(
            r#"
//- minicore: asm
fn foo() {
    unsafe {
        core::arch::asm!(
            "jmp {}",
            label {
                let p = 0xDEADBEAF as *mut u8;
                *p = 3;
             // ^^ error: dereference of raw pointer is unsafe and requires an unsafe function or block
            },
        );
    }
}
            "#,
        );
    }

    #[test]
    fn regression_19823() {
        check_diagnostics(
            r#"
pub trait FooTrait {
    unsafe fn method1();
    unsafe fn method2();
}

unsafe fn some_unsafe_fn() {}

macro_rules! impl_foo {
    () => {
        unsafe fn method1() {
            some_unsafe_fn();
        }
        unsafe fn method2() {
            some_unsafe_fn();
        }
    };
}

pub struct S1;
#[allow(unsafe_op_in_unsafe_fn)]
impl FooTrait for S1 {
    unsafe fn method1() {
        some_unsafe_fn();
    }

    unsafe fn method2() {
        some_unsafe_fn();
    }
}

pub struct S2;
#[allow(unsafe_op_in_unsafe_fn)]
impl FooTrait for S2 {
    impl_foo!();
}
        "#,
        );
    }

    #[test]
    fn no_false_positive_on_format_args_since_1_89_0() {
        check_diagnostics(
            r#"
//- minicore: fmt
fn test() {
    let foo = 10;
    let bar = true;
    let _x = format_args!("{} {0} {} {last}", foo, bar, last = "!");
}
            "#,
        );
    }

    #[test]
    fn naked_asm_is_safe() {
        check_diagnostics(
            r#"
#[rustc_builtin_macro]
macro_rules! naked_asm { () => {} }

#[unsafe(naked)]
extern "C" fn naked() {
    naked_asm!("");
}
        "#,
        );
    }
}
