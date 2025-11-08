use hir::{Adjust, Mutability};
use ide_db::assists::AssistId;
use itertools::Itertools;
use syntax::{
    AstNode, T,
    ast::{self, syntax_factory::SyntaxFactory},
};

use crate::{AssistContext, Assists};

// Assist: add_explicit_method_call_deref
//
// Insert explicit method call reference and dereferences.
//
// ```
// struct Foo;
// impl Foo { fn foo(&self) {} }
// fn test() {
//     Foo$0.$0foo();
// }
// ```
// ->
// ```
// struct Foo;
// impl Foo { fn foo(&self) {} }
// fn test() {
//     (&Foo).foo();
// }
// ```
pub(crate) fn add_explicit_method_call_deref(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    if ctx.has_empty_selection() {
        return None;
    }
    let dot_token = ctx.find_token_syntax_at_offset(T![.])?;
    if ctx.selection_trimmed() != dot_token.text_range() {
        return None;
    }
    let method_call_expr = dot_token.parent().and_then(ast::MethodCallExpr::cast)?;
    let receiver = method_call_expr.receiver()?;

    let adjustments = ctx.sema.expr_adjustments(&receiver)?;
    let adjustments =
        adjustments.into_iter().filter_map(|adjust| simple_adjust_kind(adjust.kind)).collect_vec();
    if adjustments.is_empty() {
        return None;
    }

    acc.add(
        AssistId::refactor_rewrite("add_explicit_method_call_deref"),
        "Insert explicit method call derefs",
        dot_token.text_range(),
        |builder| {
            let mut edit = builder.make_editor(method_call_expr.syntax());
            let make = SyntaxFactory::without_mappings();
            let mut expr = receiver.clone();

            for adjust_kind in adjustments {
                expr = adjust_kind.wrap_expr(expr, &make);
            }

            expr = make.expr_paren(expr).into();
            edit.replace(receiver.syntax(), expr.syntax());

            builder.add_file_edits(ctx.vfs_file_id(), edit);
        },
    )
}

fn simple_adjust_kind(adjust: Adjust) -> Option<AdjustKind> {
    match adjust {
        Adjust::NeverToAny | Adjust::Pointer(_) => None,
        Adjust::Deref(_) => Some(AdjustKind::Deref),
        Adjust::Borrow(hir::AutoBorrow::Ref(mutability)) => Some(AdjustKind::Ref(mutability)),
        Adjust::Borrow(hir::AutoBorrow::RawPtr(mutability)) => Some(AdjustKind::RefRaw(mutability)),
    }
}

enum AdjustKind {
    Deref,
    Ref(Mutability),
    RefRaw(Mutability),
}

impl AdjustKind {
    fn wrap_expr(self, expr: ast::Expr, make: &SyntaxFactory) -> ast::Expr {
        match self {
            AdjustKind::Deref => make.expr_prefix(T![*], expr).into(),
            AdjustKind::Ref(mutability) => make.expr_ref(expr, mutability.is_mut()),
            AdjustKind::RefRaw(mutability) => make.expr_raw_ref(expr, mutability.is_mut()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn works_ref() {
        check_assist(
            add_explicit_method_call_deref,
            r#"
            struct Foo;
            impl Foo { fn foo(&self) {} }
            fn test() {
                Foo$0.$0foo();
            }"#,
            r#"
            struct Foo;
            impl Foo { fn foo(&self) {} }
            fn test() {
                (&Foo).foo();
            }"#,
        );
    }

    #[test]
    fn works_ref_mut() {
        check_assist(
            add_explicit_method_call_deref,
            r#"
            struct Foo;
            impl Foo { fn foo(&mut self) {} }
            fn test() {
                Foo$0.$0foo();
            }"#,
            r#"
            struct Foo;
            impl Foo { fn foo(&mut self) {} }
            fn test() {
                (&mut Foo).foo();
            }"#,
        );
    }

    #[test]
    fn works_deref() {
        check_assist(
            add_explicit_method_call_deref,
            r#"
            struct Foo;
            impl Foo { fn foo(self) {} }
            fn test() {
                let foo = &Foo;
                foo$0.$0foo();
            }"#,
            r#"
            struct Foo;
            impl Foo { fn foo(self) {} }
            fn test() {
                let foo = &Foo;
                (*foo).foo();
            }"#,
        );
    }

    #[test]
    fn works_reborrow() {
        check_assist(
            add_explicit_method_call_deref,
            r#"
            struct Foo;
            impl Foo { fn foo(&self) {} }
            fn test() {
                let foo = &mut Foo;
                foo$0.$0foo();
            }"#,
            r#"
            struct Foo;
            impl Foo { fn foo(&self) {} }
            fn test() {
                let foo = &mut Foo;
                (&*foo).foo();
            }"#,
        );
    }

    #[test]
    fn works_deref_reborrow() {
        check_assist(
            add_explicit_method_call_deref,
            r#"
            //- minicore: deref
            struct Foo;
            struct Bar;
            impl core::ops::Deref for Foo {
                type Target = Bar;
                fn deref(&self) -> &Self::Target {}
            }
            impl Bar { fn bar(&self) {} }
            fn test() {
                let foo = &mut Foo;
                foo$0.$0bar();
            }"#,
            r#"
            struct Foo;
            struct Bar;
            impl core::ops::Deref for Foo {
                type Target = Bar;
                fn deref(&self) -> &Self::Target {}
            }
            impl Bar { fn bar(&self) {} }
            fn test() {
                let foo = &mut Foo;
                (&**foo).bar();
            }"#,
        );
    }
}
