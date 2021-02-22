use ide_db::defs::{Definition, NameRefClass};
use syntax::{ast, AstNode, SyntaxKind, T};
use test_utils::mark;

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: add_turbo_fish
//
// Adds `::<_>` to a call of a generic method or function.
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
//     let x = make::<${0:_}>();
// }
// ```
pub(crate) fn add_turbo_fish(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let ident = ctx.find_token_syntax_at_offset(SyntaxKind::IDENT).or_else(|| {
        let arg_list = ctx.find_node_at_offset::<ast::ArgList>()?;
        if arg_list.args().count() > 0 {
            return None;
        }
        mark::hit!(add_turbo_fish_after_call);
        arg_list.l_paren_token()?.prev_token().filter(|it| it.kind() == SyntaxKind::IDENT)
    })?;
    let next_token = ident.next_token()?;
    if next_token.kind() == T![::] {
        mark::hit!(add_turbo_fish_one_fish_is_enough);
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
        mark::hit!(add_turbo_fish_non_generic);
        return None;
    }
    acc.add(
        AssistId("add_turbo_fish", AssistKind::RefactorRewrite),
        "Add `::<>`",
        ident.text_range(),
        |builder| match ctx.config.snippet_cap {
            Some(cap) => builder.insert_snippet(cap, ident.text_range().end(), "::<${0:_}>"),
            None => builder.insert(ident.text_range().end(), "::<_>"),
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;
    use test_utils::mark;

    #[test]
    fn add_turbo_fish_function() {
        check_assist(
            add_turbo_fish,
            r#"
fn make<T>() -> T {}
fn main() {
    make$0();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    make::<${0:_}>();
}
"#,
        );
    }

    #[test]
    fn add_turbo_fish_after_call() {
        mark::check!(add_turbo_fish_after_call);
        check_assist(
            add_turbo_fish,
            r#"
fn make<T>() -> T {}
fn main() {
    make()$0;
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    make::<${0:_}>();
}
"#,
        );
    }

    #[test]
    fn add_turbo_fish_method() {
        check_assist(
            add_turbo_fish,
            r#"
struct S;
impl S {
    fn make<T>(&self) -> T {}
}
fn main() {
    S.make$0();
}
"#,
            r#"
struct S;
impl S {
    fn make<T>(&self) -> T {}
}
fn main() {
    S.make::<${0:_}>();
}
"#,
        );
    }

    #[test]
    fn add_turbo_fish_one_fish_is_enough() {
        mark::check!(add_turbo_fish_one_fish_is_enough);
        check_assist_not_applicable(
            add_turbo_fish,
            r#"
fn make<T>() -> T {}
fn main() {
    make$0::<()>();
}
"#,
        );
    }

    #[test]
    fn add_turbo_fish_non_generic() {
        mark::check!(add_turbo_fish_non_generic);
        check_assist_not_applicable(
            add_turbo_fish,
            r#"
fn make() -> () {}
fn main() {
    make$0();
}
"#,
        );
    }
}
