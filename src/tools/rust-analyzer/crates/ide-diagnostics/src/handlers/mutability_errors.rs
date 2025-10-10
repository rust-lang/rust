use hir::db::ExpandDatabase;
use ide_db::source_change::SourceChange;
use ide_db::text_edit::TextEdit;
use syntax::{AstNode, SyntaxKind, SyntaxNode, SyntaxNodePtr, SyntaxToken, T, ast};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, fix};

// Diagnostic: need-mut
//
// This diagnostic is triggered on mutating an immutable variable.
pub(crate) fn need_mut(ctx: &DiagnosticsContext<'_>, d: &hir::NeedMut) -> Option<Diagnostic> {
    let root = ctx.sema.db.parse_or_expand(d.span.file_id);
    let node = d.span.value.to_node(&root);
    let mut span = d.span;
    if let Some(parent) = node.parent()
        && ast::BinExpr::can_cast(parent.kind())
    {
        // In case of an assignment, the diagnostic is provided on the variable name.
        // We want to expand it to include the whole assignment, but only when this
        // is an ordinary assignment, not a destructuring assignment. So, the direct
        // parent is an assignment expression.
        span = d.span.with_value(SyntaxNodePtr::new(&parent));
    };

    let fixes = (|| {
        if d.local.is_ref(ctx.sema.db) {
            // There is no simple way to add `mut` to `ref x` and `ref mut x`
            return None;
        }
        let file_id = span.file_id.file_id()?;
        let mut edit_builder = TextEdit::builder();
        let use_range = span.value.text_range();
        for source in d.local.sources(ctx.sema.db) {
            let Some(ast) = source.name() else { continue };
            // FIXME: macros
            edit_builder.insert(ast.value.syntax().text_range().start(), "mut ".to_owned());
        }
        let edit = edit_builder.finish();
        Some(vec![fix(
            "add_mut",
            "Change it to be mutable",
            SourceChange::from_text_edit(file_id.file_id(ctx.sema.db), edit),
            use_range,
        )])
    })();

    Some(
        Diagnostic::new_with_syntax_node_ptr(
            ctx,
            // FIXME: `E0384` is not the only error that this diagnostic handles
            DiagnosticCode::RustcHardError("E0384"),
            format!(
                "cannot mutate immutable variable `{}`",
                d.local.name(ctx.sema.db).display(ctx.sema.db, ctx.edition)
            ),
            span,
        )
        .stable()
        .with_fixes(fixes),
    )
}

// Diagnostic: unused-mut
//
// This diagnostic is triggered when a mutable variable isn't actually mutated.
pub(crate) fn unused_mut(ctx: &DiagnosticsContext<'_>, d: &hir::UnusedMut) -> Option<Diagnostic> {
    let ast = d.local.primary_source(ctx.sema.db).syntax_ptr();
    let fixes = (|| {
        let file_id = ast.file_id.file_id()?;
        let mut edit_builder = TextEdit::builder();
        let use_range = ast.value.text_range();
        for source in d.local.sources(ctx.sema.db) {
            let ast = source.syntax();
            let Some(mut_token) = token(ast, T![mut]) else { continue };
            edit_builder.delete(mut_token.text_range());
            if let Some(token) = mut_token.next_token()
                && token.kind() == SyntaxKind::WHITESPACE
            {
                edit_builder.delete(token.text_range());
            }
        }
        let edit = edit_builder.finish();
        Some(vec![fix(
            "remove_mut",
            "Remove unnecessary `mut`",
            SourceChange::from_text_edit(file_id.file_id(ctx.sema.db), edit),
            use_range,
        )])
    })();
    let ast = d.local.primary_source(ctx.sema.db).syntax_ptr();
    Some(
        Diagnostic::new_with_syntax_node_ptr(
            ctx,
            DiagnosticCode::RustcLint("unused_mut"),
            "variable does not need to be mutable",
            ast,
        )
        // Not supporting `#[allow(unused_mut)]` in proc macros leads to false positive, hence not stable.
        .with_fixes(fixes),
    )
}

pub(super) fn token(parent: &SyntaxNode, kind: SyntaxKind) -> Option<SyntaxToken> {
    parent.children_with_tokens().filter_map(|it| it.into_token()).find(|it| it.kind() == kind)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_diagnostics_with_disabled, check_fix};

    #[test]
    fn unused_mut_simple() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut x = 2;
      //^^^^^ 💡 warn: variable does not need to be mutable
    f(x);
}
"#,
        );
    }

    #[test]
    fn no_false_positive_simple() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let x = 2;
    f(x);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut x = 2;
    x = 5;
    f(x);
}
"#,
        );
    }

    #[test]
    fn multiple_errors_for_single_variable() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let x = 2;
    x = 10;
  //^^^^^^ 💡 error: cannot mutate immutable variable `x`
    x = 5;
  //^^^^^ 💡 error: cannot mutate immutable variable `x`
    &mut x;
  //^^^^^^ 💡 error: cannot mutate immutable variable `x`
    f(x);
}
"#,
        );
    }

    #[test]
    fn unused_mut_fix() {
        check_fix(
            r#"
fn f(_: i32) {}
fn main() {
    let mu$0t x = 2;
    f(x);
}
"#,
            r#"
fn f(_: i32) {}
fn main() {
    let x = 2;
    f(x);
}
"#,
        );
        check_fix(
            r#"
fn f(_: i32) {}
fn main() {
    let ((mu$0t x, _) | (_, mut x)) = (2, 3);
    f(x);
}
"#,
            r#"
fn f(_: i32) {}
fn main() {
    let ((x, _) | (_, x)) = (2, 3);
    f(x);
}
"#,
        );
    }

    #[test]
    fn need_mut_fix() {
        check_fix(
            r#"
fn f(_: i32) {}
fn main() {
    let x = 2;
    x$0 = 5;
    f(x);
}
"#,
            r#"
fn f(_: i32) {}
fn main() {
    let mut x = 2;
    x = 5;
    f(x);
}
"#,
        );
        check_fix(
            r#"
fn f(_: i32) {}
fn main() {
    let ((x, _) | (_, x)) = (2, 3);
    x =$0 4;
    f(x);
}
"#,
            r#"
fn f(_: i32) {}
fn main() {
    let ((mut x, _) | (_, mut x)) = (2, 3);
    x = 4;
    f(x);
}
"#,
        );

        check_fix(
            r#"
struct Foo(i32);

impl Foo {
    fn foo(self) {
        self = Fo$0o(5);
    }
}
"#,
            r#"
struct Foo(i32);

impl Foo {
    fn foo(mut self) {
        self = Foo(5);
    }
}
"#,
        );
    }

    #[test]
    fn need_mut_fix_not_applicable_on_ref() {
        check_diagnostics(
            r#"
fn main() {
    let ref x = 2;
    x = &5;
  //^^^^^^ error: cannot mutate immutable variable `x`
}
"#,
        );
        check_diagnostics(
            r#"
fn main() {
    let ref mut x = 2;
    x = &mut 5;
  //^^^^^^^^^^ error: cannot mutate immutable variable `x`
}
"#,
        );
    }

    #[test]
    fn field_mutate() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut x = (2, 7);
      //^^^^^ 💡 warn: variable does not need to be mutable
    f(x.1);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut x = (2, 7);
    x.0 = 5;
    f(x.1);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let x = (2, 7);
    x.0 = 5;
  //^^^^^^^ 💡 error: cannot mutate immutable variable `x`
    f(x.1);
}
"#,
        );
    }

    #[test]
    fn mutable_reference() {
        check_diagnostics(
            r#"
fn main() {
    let mut x = &mut 2;
      //^^^^^ 💡 warn: variable does not need to be mutable
    *x = 5;
}
"#,
        );
        check_diagnostics(
            r#"
fn main() {
    let x = 2;
    &mut x;
  //^^^^^^ 💡 error: cannot mutate immutable variable `x`
}
"#,
        );
        check_diagnostics(
            r#"
fn main() {
    let x_own = 2;
    let ref mut x_ref = x_own;
      //^^^^^^^^^^^^^ 💡 error: cannot mutate immutable variable `x_own`
    _ = x_ref;
}
"#,
        );
        check_diagnostics(
            r#"
struct Foo;
impl Foo {
    fn method(&mut self, _x: i32) {}
}
fn main() {
    let x = Foo;
    x.method(2);
  //^ 💡 error: cannot mutate immutable variable `x`
}
"#,
        );
    }

    #[test]
    fn regression_14310() {
        check_diagnostics(
            r#"
            //- minicore: copy, builtin_impls
            fn clone(mut i: &!) -> ! {
                   //^^^^^ 💡 warn: variable does not need to be mutable
                *i
            }
        "#,
        );
    }

    #[test]
    fn match_closure_capture() {
        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    let mut v = &mut Some(2);
      //^^^^^ 💡 warn: variable does not need to be mutable
    let _ = || match v {
        Some(k) => {
            *k = 5;
        }
        None => {}
    };
    let v = &mut Some(2);
    let _ = || match v {
                   //^ 💡 error: cannot mutate immutable variable `v`
        ref mut k => {
            *k = &mut Some(5);
        }
    };
}
"#,
        );
    }

    #[test]
    fn match_bindings() {
        check_diagnostics(
            r#"
fn main() {
    match (2, 3) {
        (x, mut y) => {
          //^^^^^ 💡 warn: variable does not need to be mutable
            x = 7;
          //^^^^^ 💡 error: cannot mutate immutable variable `x`
            _ = y;
        }
    }
}
"#,
        );
    }

    #[test]
    fn mutation_in_dead_code() {
        // This one is interesting. Dead code is not represented at all in the MIR, so
        // there would be no mutability error for locals in dead code. Rustc tries to
        // not emit `unused_mut` in this case, but since it works without `mut`, and
        // special casing it is not trivial, we emit it.

        // Update: now MIR based `unused-variable` is taking over `unused-mut` for the same reason.
        check_diagnostics(
            r#"
fn main() {
    return;
    let mut x = 2;
      //^^^^^ 💡 warn: unused variable
    &mut x;
}
"#,
        );
        check_diagnostics(
            r#"
fn main() {
    loop {}
    let mut x = 2;
      //^^^^^ 💡 warn: unused variable
    &mut x;
}
"#,
        );
        check_diagnostics_with_disabled(
            r#"
enum X {}
fn g() -> X {
    loop {}
}
fn f() -> ! {
    loop {}
}
fn main(b: bool) {
    if b {
        f();
    } else {
        g();
    }
    let mut x = 2;
      //^^^^^ 💡 warn: unused variable
    &mut x;
}
"#,
            &["remove-unnecessary-else"],
        );
        check_diagnostics_with_disabled(
            r#"
fn main(b: bool) {
    if b {
        loop {}
    } else {
        return;
    }
    let mut x = 2;
      //^^^^^ 💡 warn: unused variable
    &mut x;
}
"#,
            &["remove-unnecessary-else"],
        );
    }

    #[test]
    fn initialization_is_not_mutation() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut x;
      //^^^^^ 💡 warn: variable does not need to be mutable
    x = 5;
    f(x);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main(b: bool) {
    let mut x;
      //^^^^^ 💡 warn: variable does not need to be mutable
    if b {
        x = 1;
    } else {
        x = 3;
    }
    f(x);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main(b: bool) {
    let x;
    if b {
        x = 1;
    }
    x = 3;
  //^^^^^ 💡 error: cannot mutate immutable variable `x`
    f(x);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let x;
    loop {
        x = 1;
      //^^^^^ 💡 error: cannot mutate immutable variable `x`
        f(x);
    }
}
"#,
        );
        check_diagnostics(
            r#"
fn check(_: i32) -> bool {
    false
}
fn main() {
    loop {
        let x = 1;
        if check(x) {
            break;
        }
        let y = (1, 2);
        if check(y.1) {
            return;
        }
        let z = (1, 2);
        match z {
            (k @ 5, ref mut t) if { continue; } => {
                  //^^^^^^^^^ 💡 error: cannot mutate immutable variable `z`
                *t = 5;
                _ = k;
            }
            _ => {
                let y = (1, 2);
                if check(y.1) {
                    return;
                }
            }
        }
    }
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    loop {
        let mut x = 1;
          //^^^^^ 💡 warn: variable does not need to be mutable
        f(x);
        if let mut y = 2 {
             //^^^^^ 💡 warn: variable does not need to be mutable
            f(y);
        }
        match 3 {
            mut z => f(z),
          //^^^^^ 💡 warn: variable does not need to be mutable
        }
    }
}
"#,
        );
    }

    #[test]
    fn initialization_is_not_mutation_in_loop() {
        check_diagnostics(
            r#"
fn main() {
    let a;
    loop {
        let c @ (
            mut b,
          //^^^^^ 💡 warn: variable does not need to be mutable
            mut d
          //^^^^^ 💡 warn: variable does not need to be mutable
        );
        a = 1;
      //^^^^^ 💡 error: cannot mutate immutable variable `a`
        b = 1;
        c = (2, 3);
        d = 3;
        _ = (c, b, d);
    }
}
"#,
        );
    }

    #[test]
    fn function_arguments_are_initialized() {
        check_diagnostics(
            r#"
fn f(mut x: i32) {
   //^^^^^ 💡 warn: variable does not need to be mutable
   f(x + 2);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(x: i32) {
   x = 5;
 //^^^^^ 💡 error: cannot mutate immutable variable `x`
}
"#,
        );
        check_diagnostics(
            r#"
fn f((x, y): (i32, i32)) {
    let t = [0; 2];
    x = 5;
  //^^^^^ 💡 error: cannot mutate immutable variable `x`
    _ = x;
    _ = y;
    _ = t;
}
"#,
        );
    }

    #[test]
    fn no_diagnostics_in_case_of_multiple_bounds() {
        check_diagnostics(
            r#"
fn f() {
    let (b, a, b) = (2, 3, 5);
    a = 8;
  //^^^^^ 💡 error: cannot mutate immutable variable `a`
}
"#,
        );
    }

    #[test]
    fn for_loop() {
        check_diagnostics(
            r#"
//- minicore: iterators, copy
fn f(x: [(i32, u8); 10]) {
    for (a, mut b) in x {
          //^^^^^ 💡 warn: variable does not need to be mutable
        a = 2;
      //^^^^^ 💡 error: cannot mutate immutable variable `a`
        _ = b;
    }
}
"#,
        );
    }

    #[test]
    fn while_let() {
        check_diagnostics(
            r#"
//- minicore: iterators, copy
fn f(x: [(i32, u8); 10]) {
    let mut it = x.into_iter();
    while let Some((a, mut b)) = it.next() {
                     //^^^^^ 💡 warn: variable does not need to be mutable
        while let Some((c, mut d)) = it.next() {
                         //^^^^^ 💡 warn: variable does not need to be mutable
            a = 2;
          //^^^^^ 💡 error: cannot mutate immutable variable `a`
            c = 2;
          //^^^^^ 💡 error: cannot mutate immutable variable `c`
            _ = (b, d);
        }
    }
}
"#,
        );
    }

    #[test]
    fn index() {
        check_diagnostics(
            r#"
//- minicore: coerce_unsized, index, slice
fn f() {
    let x = [1, 2, 3];
    x[2] = 5;
  //^^^^^^^^ 💡 error: cannot mutate immutable variable `x`
    let x = &mut x;
          //^^^^^^ 💡 error: cannot mutate immutable variable `x`
    let mut x = x;
      //^^^^^ 💡 warn: variable does not need to be mutable
    x[2] = 5;
}
"#,
        );
    }

    #[test]
    fn overloaded_index() {
        check_diagnostics(
            r#"
//- minicore: index, copy
use core::ops::{Index, IndexMut};

struct Foo;
impl Index<usize> for Foo {
    type Output = (i32, u8);
    fn index(&self, _index: usize) -> &(i32, u8) {
        &(5, 2)
    }
}
impl IndexMut<usize> for Foo {
    fn index_mut(&mut self, _index: usize) -> &mut (i32, u8) {
        &mut (5, 2)
    }
}
fn f() {
    let mut x = Foo;
      //^^^^^ 💡 warn: variable does not need to be mutable
    let y = &x[2];
    _ = (x, y);
    let x = Foo;
    let y = &mut x[2];
               //^💡 error: cannot mutate immutable variable `x`
    _ = (x, y);
    let mut x = &mut Foo;
      //^^^^^ 💡 warn: variable does not need to be mutable
    let y: &mut (i32, u8) = &mut x[2];
    _ = (x, y);
    let x = Foo;
    let ref mut y = x[7];
                  //^ 💡 error: cannot mutate immutable variable `x`
    _ = (x, y);
    let (ref mut y, _) = x[3];
                       //^ 💡 error: cannot mutate immutable variable `x`
    _ = y;
    match x[10] {
        //^ 💡 error: cannot mutate immutable variable `x`
        (ref y, 5) => _ = y,
        (_, ref mut y) => _ = y,
    }
    let mut x = Foo;
    let mut i = 5;
      //^^^^^ 💡 warn: variable does not need to be mutable
    let y = &mut x[i];
    _ = y;
}
"#,
        );
    }

    #[test]
    fn overloaded_deref() {
        check_diagnostics(
            r#"
//- minicore: deref_mut, copy
use core::ops::{Deref, DerefMut};

struct Foo;
impl Deref for Foo {
    type Target = (i32, u8);
    fn deref(&self) -> &(i32, u8) {
        &(5, 2)
    }
}
impl DerefMut for Foo {
    fn deref_mut(&mut self) -> &mut (i32, u8) {
        &mut (5, 2)
    }
}
fn f() {
    let mut x = Foo;
      //^^^^^ 💡 warn: variable does not need to be mutable
    let y = &*x;
    _ = (x, y);
    let x = Foo;
    let y = &mut *x;
               //^^ 💡 error: cannot mutate immutable variable `x`
    _ = (x, y);
    let x = Foo;
      //^ 💡 warn: unused variable
    let x = Foo;
    let y: &mut (i32, u8) = &mut x;
                          //^^^^^^ 💡 error: cannot mutate immutable variable `x`
    _ = (x, y);
    let ref mut y = *x;
                  //^^ 💡 error: cannot mutate immutable variable `x`
    _ = y;
    let (ref mut y, _) = *x;
                       //^^ 💡 error: cannot mutate immutable variable `x`
    _ = y;
    match *x {
        //^^ 💡 error: cannot mutate immutable variable `x`
        (ref y, 5) => _ = y,
        (_, ref mut y) => _ = y,
    }
}
"#,
        );
    }

    #[test]
    fn or_pattern() {
        check_diagnostics(
            r#"
//- minicore: option
fn f(_: i32) {}
fn main() {
    let ((Some(mut x), None) | (_, Some(mut x))) = (None, Some(7)) else { return };
             //^^^^^ 💡 warn: variable does not need to be mutable

    f(x);
}
"#,
        );
        check_diagnostics(
            r#"
struct Foo(i32);

const X: Foo = Foo(5);
const Y: Foo = Foo(12);

const fn f(mut a: Foo) -> bool {
         //^^^^^ 💡 warn: variable does not need to be mutable
    match a {
        X | Y => true,
        _ => false,
    }
}
"#,
        );
    }

    #[test]
    fn or_pattern_no_terminator() {
        check_diagnostics(
            r#"
enum Foo {
    A, B, C, D
}

use Foo::*;

fn f(inp: (Foo, Foo, Foo, Foo)) {
    let ((A, B, _, x) | (B, C | D, x, _)) = inp else {
        return;
    };
    x = B;
  //^^^^^ 💡 error: cannot mutate immutable variable `x`
}
"#,
        );
    }

    #[test]
    // FIXME: We should have tests for `is_ty_uninhabited_from`
    fn regression_14421() {
        check_diagnostics(
            r#"
pub enum Tree {
    Node(TreeNode),
    Leaf(TreeLeaf),
}

struct Box<T>(&T);

pub struct TreeNode {
    pub depth: usize,
    pub children: [Box<Tree>; 8]
}

pub struct TreeLeaf {
    pub depth: usize,
    pub data: u8
}

pub fn test() {
    let mut tree = Tree::Leaf(
      //^^^^^^^^ 💡 warn: variable does not need to be mutable
        TreeLeaf {
            depth: 0,
            data: 0
        }
    );
    _ = tree;
}
"#,
        );
    }

    #[test]
    fn fn_traits() {
        check_diagnostics(
            r#"
//- minicore: fn
fn fn_ref(mut x: impl Fn(u8) -> u8) -> u8 {
        //^^^^^ 💡 warn: variable does not need to be mutable
    x(2)
}
fn fn_mut(x: impl FnMut(u8) -> u8) -> u8 {
    x(2)
  //^ 💡 error: cannot mutate immutable variable `x`
}
fn fn_borrow_mut(mut x: &mut impl FnMut(u8) -> u8) -> u8 {
               //^^^^^ 💡 warn: variable does not need to be mutable
    x(2)
}
fn fn_once(mut x: impl FnOnce(u8) -> u8) -> u8 {
         //^^^^^ 💡 warn: variable does not need to be mutable
    x(2)
}
"#,
        );
    }

    #[test]
    fn closure() {
        check_diagnostics(
            r#"
        //- minicore: copy, fn
        struct X;

        impl X {
            fn mutate(&mut self) {}
        }

        fn f() {
            let x = 5;
            let closure1 = || { x = 2; };
                              //^^^^^ 💡 error: cannot mutate immutable variable `x`
            let _ = closure1();
                  //^^^^^^^^ 💡 error: cannot mutate immutable variable `closure1`
            let closure2 = || { x = x; };
                              //^^^^^ 💡 error: cannot mutate immutable variable `x`
            let closure3 = || {
                let x = 2;
                x = 5;
              //^^^^^ 💡 error: cannot mutate immutable variable `x`
                x
            };
            let x = X;
            let closure4 = || { x.mutate(); };
                              //^ 💡 error: cannot mutate immutable variable `x`
            _ = (closure2, closure3, closure4);
        }
                    "#,
        );
        check_diagnostics(
            r#"
        //- minicore: copy, fn
        fn f() {
            let mut x = 5;
              //^^^^^ 💡 warn: variable does not need to be mutable
            let mut y = 2;
            y = 7;
            let closure = || {
                let mut z = 8;
                z = 3;
                let mut k = z;
                  //^^^^^ 💡 warn: variable does not need to be mutable
                _ = k;
            };
            _ = (x, closure);
        }
                    "#,
        );
        check_diagnostics(
            r#"
//- minicore: copy, fn
fn f() {
    let closure = || {
        || {
            || {
                let x = 2;
                || { || { x = 5; } }
                        //^^^^^ 💡 error: cannot mutate immutable variable `x`
            }
        }
    };
    _ = closure;
}
            "#,
        );
        check_diagnostics(
            r#"
//- minicore: copy, fn
fn f() {
    struct X;
    let mut x = X;
      //^^^^^ 💡 warn: variable does not need to be mutable
    let c1 = || x;
    let mut x = X;
    let c2 = || { x = X; x };
    let mut x = X;
    let c3 = move || { x = X; };
    _ = (c1, c2, c3);
}
            "#,
        );
        check_diagnostics(
            r#"
        //- minicore: copy, fn, deref_mut
        struct X(i32, i64);

        fn f() {
            let mut x = &mut 5;
              //^^^^^ 💡 warn: variable does not need to be mutable
            let closure1 = || { *x = 2; };
            let _ = closure1();
                  //^^^^^^^^ 💡 error: cannot mutate immutable variable `closure1`
            let mut x = &mut 5;
              //^^^^^ 💡 warn: variable does not need to be mutable
            let closure1 = || { *x = 2; &x; };
            let _ = closure1();
                  //^^^^^^^^ 💡 error: cannot mutate immutable variable `closure1`
            let mut x = &mut 5;
            let closure1 = || { *x = 2; &x; x = &mut 3; };
            let _ = closure1();
                  //^^^^^^^^ 💡 error: cannot mutate immutable variable `closure1`
            let mut x = &mut 5;
              //^^^^^ 💡 warn: variable does not need to be mutable
            let closure1 = move || { *x = 2; };
            let _ = closure1();
                  //^^^^^^^^ 💡 error: cannot mutate immutable variable `closure1`
            let mut x = &mut X(1, 2);
              //^^^^^ 💡 warn: variable does not need to be mutable
            let closure1 = || { x.0 = 2; };
            let _ = closure1();
                  //^^^^^^^^ 💡 error: cannot mutate immutable variable `closure1`
        }
                    "#,
        );
    }

    #[test]
    fn slice_pattern() {
        check_diagnostics(
            r#"
//- minicore: coerce_unsized, deref_mut, slice, copy
fn x(t: &[u8]) {
    match t {
        &[a, mut b] | &[a, _, mut b] => {
           //^^^^^ 💡 warn: variable does not need to be mutable

            a = 2;
          //^^^^^ 💡 error: cannot mutate immutable variable `a`
            _ = b;
        }
        _ => {}
    }
}
            "#,
        );
    }

    #[test]
    fn boxes() {
        check_diagnostics(
            r#"
//- minicore: coerce_unsized, deref_mut, slice
use core::ops::{Deref, DerefMut};
use core::{marker::Unsize, ops::CoerceUnsized};

#[lang = "owned_box"]
pub struct Box<T: ?Sized> {
    inner: *mut T,
}
impl<T> Box<T> {
    fn new(t: T) -> Self {
        #[rustc_box]
        Box::new(t)
    }
}

impl<T: ?Sized> Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized> DerefMut for Box<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut **self
    }
}

fn f() {
    let x = Box::new(5);
    x = Box::new(7);
  //^^^^^^^^^^^^^^^ 💡 error: cannot mutate immutable variable `x`
    let x = Box::new(5);
    *x = 7;
  //^^^^^^ 💡 error: cannot mutate immutable variable `x`
    let mut y = Box::new(5);
      //^^^^^ 💡 warn: variable does not need to be mutable
    *x = *y;
  //^^^^^^^ 💡 error: cannot mutate immutable variable `x`
    let x = Box::new(5);
    let closure = || *x = 2;
                    //^ 💡 error: cannot mutate immutable variable `x`
    _ = closure;
}
"#,
        );
    }

    #[test]
    fn regression_15143() {
        check_diagnostics(
            r#"
        trait Tr {
            type Ty;
        }

        struct A;

        impl Tr for A {
            type Ty = (u32, i64);
        }

        struct B<T: Tr> {
            f: <T as Tr>::Ty,
        }

        fn main(b: B<A>) {
            let f = b.f.0;
            f = 5;
          //^^^^^ 💡 error: cannot mutate immutable variable `f`
        }
            "#,
        );
    }

    #[test]
    fn allow_unused_mut_for_identifiers_starting_with_underline() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut _x = 2;
    f(_x);
}
"#,
        );
    }

    #[test]
    fn respect_lint_attributes_for_unused_mut() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    #[allow(unused_mut)]
    let mut x = 2;
    f(x);
}

fn main2() {
    #[deny(unused_mut)]
    let mut x = 2;
      //^^^^^ 💡 error: variable does not need to be mutable
    f(x);
}
"#,
        );
        check_diagnostics(
            r#"
macro_rules! mac {
    ($($x:expr),*$(,)*) => ({
        #[allow(unused_mut)]
        let mut vec = 2;
        vec
    });
}

fn main2() {
    let mut x = mac![];
      //^^^^^ 💡 warn: variable does not need to be mutable
    _ = x;
}
        "#,
        );
    }

    #[test]
    fn regression_15099() {
        check_diagnostics(
            r#"
//- minicore: iterator, range
fn f() {
    loop {}
    for _ in 0..2 {}
}
"#,
        );
    }

    #[test]
    fn regression_15623() {
        check_diagnostics(
            r#"
//- minicore: fn

struct Foo;

impl Foo {
    fn needs_mut(&mut self) {}
}

fn foo(mut foo: Foo) {
    let mut call_me = || {
        let 0 = 1 else { return };
        foo.needs_mut();
    };
    call_me();
}
"#,
        );
    }

    #[test]
    fn regression_15670() {
        check_diagnostics(
            r#"
//- minicore: fn

pub struct A {}
pub unsafe fn foo(a: *mut A) {
    let mut b = || -> *mut A { unsafe { &mut *a } };
      //^^^^^ 💡 warn: variable does not need to be mutable
    let _ = b();
}
"#,
        );
    }

    #[test]
    fn regression_15799() {
        check_diagnostics(
            r#"
//- minicore: deref_mut
struct WrapPtr(*mut u32);

impl core::ops::Deref for WrapPtr {
    type Target = *mut u32;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn main() {
    let mut x = 0u32;
    let wrap = WrapPtr(&mut x);
    unsafe {
        **wrap = 6;
    }
}
"#,
        );
    }

    #[test]
    fn destructuring_assignment_needs_mut() {
        check_diagnostics(
            r#"
//- minicore: fn

fn main() {
	let mut var = 1;
	let mut func = || (var,) = (2,);
	func();
}
        "#,
        );
    }
}
