use ide_db::defs::{Definition, NameRefClass};
use syntax::{ast, AstNode, SyntaxKind, T};
use test_utils::mark;

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: add_type_ascription
//
// Adds `: _` before the assignment operator to prompt the user for a type
//
// ```
// fn make<T>() -> T { todo!() }
// fn main() {
//     let x = make$0();
// }
// ```
// ->
// ```
// fn make<T>() -> T { todo!() }
// fn main() {
//     let x: ${0:_} = make();
// }
// ```
pub(crate) fn add_type_ascription(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let let_stmt = ctx.find_node_at_offset::<ast::LetStmt>()?;
    if let_stmt.colon_token().is_some() {
        mark::hit!(add_type_ascription_already_typed);
        return None
    }
    let type_pos = let_stmt.pat()?.syntax().last_token()?.text_range().end();

    let ident = ctx.find_token_syntax_at_offset(SyntaxKind::IDENT).or_else(|| {
        let arg_list = ctx.find_node_at_offset::<ast::ArgList>()?;
        if arg_list.args().count() > 0 {
            return None;
        }
        mark::hit!(add_type_ascription_after_call);
        arg_list.l_paren_token()?.prev_token().filter(|it| it.kind() == SyntaxKind::IDENT)
    })?;
    let next_token = ident.next_token()?;
    if next_token.kind() == T![::] {
        mark::hit!(add_type_ascription_turbofished);
        return None;
    }
    let name_ref = ast::NameRef::cast(ident.parent())?;
    let def = match NameRefClass::classify(&ctx.sema, &name_ref)? {
        NameRefClass::Definition(def) => def,
        NameRefClass::ExternCrate(_) | NameRefClass::FieldShorthand { .. } => return None,
    };
    let fun = match def {
        Definition::ModuleDef(hir::ModuleDef::Function(it)) => it,
        _ => return None,
    };
    let generics = hir::GenericDef::Function(fun).params(ctx.sema.db);
    if generics.is_empty() {
        mark::hit!(add_type_ascription_non_generic);
        return None;
    }
    acc.add(
        AssistId("add_type_ascription", AssistKind::RefactorRewrite),
        "Add `: _` before assignment operator",
        ident.text_range(),
        |builder| match ctx.config.snippet_cap {
            Some(cap) => builder.insert_snippet(cap, type_pos, ": ${0:_}"),
            None => builder.insert(type_pos, ": _"),
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;
    use test_utils::mark;

    #[test]
    fn add_type_ascription_function() {
        check_assist(
            add_type_ascription,
            r#"
fn make<T>() -> T {}
fn main() {
    let x = make$0();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let x: ${0:_} = make();
}
"#,
        );
    }

    #[test]
    fn add_type_ascription_after_call() {
        mark::check!(add_type_ascription_after_call);
        check_assist(
            add_type_ascription,
            r#"
fn make<T>() -> T {}
fn main() {
    let x = make()$0;
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let x: ${0:_} = make();
}
"#,
        );
    }

    #[test]
    fn add_type_ascription_method() {
        check_assist(
            add_type_ascription,
            r#"
struct S;
impl S {
    fn make<T>(&self) -> T {}
}
fn main() {
    let x = S.make$0();
}
"#,
            r#"
struct S;
impl S {
    fn make<T>(&self) -> T {}
}
fn main() {
    let x: ${0:_} = S.make();
}
"#,
        );
    }

    #[test]
    fn add_type_ascription_turbofished() {
        mark::check!(add_type_ascription_turbofished);
        check_assist_not_applicable(
            add_type_ascription,
            r#"
fn make<T>() -> T {}
fn main() {
    let x = make$0::<()>();
}
"#,
        );
    }

    #[test]
    fn add_type_ascription_already_typed() {
        mark::check!(add_type_ascription_already_typed);
        check_assist_not_applicable(
            add_type_ascription,
            r#"
fn make<T>() -> T {}
fn main() {
    let x: () = make$0();
}
"#,
        );
    }

    #[test]
    fn add_type_ascription_non_generic() {
        mark::check!(add_type_ascription_non_generic);
        check_assist_not_applicable(
            add_type_ascription,
            r#"
fn make() -> () {}
fn main() {
    let x = make$0();
}
"#,
        );
    }

    #[test]
    fn add_type_ascription_no_let() {
        check_assist_not_applicable(
            add_type_ascription,
            r#"
fn make<T>() -> T {}
fn main() {
    make$0();
}
"#,
        );
    }
}
