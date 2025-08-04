use syntax::{
    AstNode, T,
    ast::{self, edit::AstNodeEdit},
};

use crate::{AssistContext, AssistId, Assists};

// Assist: unwrap_tuple
//
// Unwrap the tuple to different variables.
//
// ```
// # //- minicore: result
// fn main() {
//     $0let (foo, bar) = ("Foo", "Bar");
// }
// ```
// ->
// ```
// fn main() {
//     let foo = "Foo";
//     let bar = "Bar";
// }
// ```
pub(crate) fn unwrap_tuple(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let let_kw = ctx.find_token_syntax_at_offset(T![let])?;
    let let_stmt = let_kw.parent().and_then(ast::LetStmt::cast)?;
    let indent_level = let_stmt.indent_level().0 as usize;
    let pat = let_stmt.pat()?;
    let ty = let_stmt.ty();
    let init = let_stmt.initializer()?;

    // This only applies for tuple patterns, types, and initializers.
    let tuple_pat = match pat {
        ast::Pat::TuplePat(pat) => pat,
        _ => return None,
    };
    let tuple_ty = ty.and_then(|it| match it {
        ast::Type::TupleType(ty) => Some(ty),
        _ => None,
    });
    let tuple_init = match init {
        ast::Expr::TupleExpr(expr) => expr,
        _ => return None,
    };

    if tuple_pat.fields().count() != tuple_init.fields().count() {
        return None;
    }
    if let Some(tys) = &tuple_ty
        && tuple_pat.fields().count() != tys.fields().count()
    {
        return None;
    }

    let parent = let_kw.parent()?;

    acc.add(
        AssistId::refactor_rewrite("unwrap_tuple"),
        "Unwrap tuple",
        let_kw.text_range(),
        |edit| {
            let indents = "    ".repeat(indent_level);

            // If there is an ascribed type, insert that type for each declaration,
            // otherwise, omit that type.
            if let Some(tys) = tuple_ty {
                let mut zipped_decls = String::new();
                for (pat, ty, expr) in
                    itertools::izip!(tuple_pat.fields(), tys.fields(), tuple_init.fields())
                {
                    zipped_decls.push_str(&format!("{indents}let {pat}: {ty} = {expr};\n"))
                }
                edit.replace(parent.text_range(), zipped_decls.trim());
            } else {
                let mut zipped_decls = String::new();
                for (pat, expr) in itertools::izip!(tuple_pat.fields(), tuple_init.fields()) {
                    zipped_decls.push_str(&format!("{indents}let {pat} = {expr};\n"));
                }
                edit.replace(parent.text_range(), zipped_decls.trim());
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn unwrap_tuples() {
        check_assist(
            unwrap_tuple,
            r#"
fn main() {
    $0let (foo, bar) = ("Foo", "Bar");
}
"#,
            r#"
fn main() {
    let foo = "Foo";
    let bar = "Bar";
}
"#,
        );

        check_assist(
            unwrap_tuple,
            r#"
fn main() {
    $0let (foo, bar, baz) = ("Foo", "Bar", "Baz");
}
"#,
            r#"
fn main() {
    let foo = "Foo";
    let bar = "Bar";
    let baz = "Baz";
}
"#,
        );
    }

    #[test]
    fn unwrap_tuple_with_types() {
        check_assist(
            unwrap_tuple,
            r#"
fn main() {
    $0let (foo, bar): (u8, i32) = (5, 10);
}
"#,
            r#"
fn main() {
    let foo: u8 = 5;
    let bar: i32 = 10;
}
"#,
        );

        check_assist(
            unwrap_tuple,
            r#"
fn main() {
    $0let (foo, bar, baz): (u8, i32, f64) = (5, 10, 17.5);
}
"#,
            r#"
fn main() {
    let foo: u8 = 5;
    let bar: i32 = 10;
    let baz: f64 = 17.5;
}
"#,
        );
    }
}
