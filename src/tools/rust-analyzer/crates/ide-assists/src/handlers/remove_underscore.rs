use ide_db::{
    assists::AssistId,
    defs::{Definition, NameClass, NameRefClass},
    rename::RenameDefinition,
};
use syntax::{AstNode, ast};

use crate::{AssistContext, Assists};

// Assist: remove_underscore_from_used_variables
//
// Removes underscore from used variables.
//
// ```
// fn main() {
//     let mut _$0foo = 1;
//     _foo = 2;
// }
// ```
// ->
// ```
// fn main() {
//     let mut foo = 1;
//     foo = 2;
// }
// ```
pub(crate) fn remove_underscore(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let (text, text_range, def) = if let Some(name_ref) = ctx.find_node_at_offset::<ast::Name>() {
        let text = name_ref.text();
        if !text.starts_with('_') {
            return None;
        }

        let def = match NameClass::classify(&ctx.sema, &name_ref)? {
            NameClass::Definition(def @ Definition::Local(_)) => def,
            NameClass::PatFieldShorthand { local_def, .. } => Definition::Local(local_def),
            _ => return None,
        };
        (text.to_owned(), name_ref.syntax().text_range(), def)
    } else if let Some(name_ref) = ctx.find_node_at_offset::<ast::NameRef>() {
        let text = name_ref.text();
        if !text.starts_with('_') {
            return None;
        }
        let def = match NameRefClass::classify(&ctx.sema, &name_ref)? {
            NameRefClass::Definition(def @ Definition::Local(_), _) => def,
            NameRefClass::FieldShorthand { local_ref, .. } => Definition::Local(local_ref),
            _ => return None,
        };
        (text.to_owned(), name_ref.syntax().text_range(), def)
    } else {
        return None;
    };

    if !def.usages(&ctx.sema).at_least_one() {
        return None;
    }

    let new_name = text.trim_start_matches('_');
    acc.add(
        AssistId::refactor("remove_underscore_from_used_variables"),
        "Remove underscore from a used variable",
        text_range,
        |builder| {
            let changes = def.rename(&ctx.sema, new_name, RenameDefinition::Yes).unwrap();
            builder.source_change = changes;
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn remove_underscore_from_used_variable() {
        check_assist(
            remove_underscore,
            r#"
fn main() {
    let mut _$0foo = 1;
    _foo = 2;
}
"#,
            r#"
fn main() {
    let mut foo = 1;
    foo = 2;
}
"#,
        );
    }

    #[test]
    fn not_applicable_for_unused() {
        check_assist_not_applicable(
            remove_underscore,
            r#"
fn main() {
    let _$0unused = 1;
}
"#,
        );
    }

    #[test]
    fn not_applicable_for_no_underscore() {
        check_assist_not_applicable(
            remove_underscore,
            r#"
fn main() {
    let f$0oo = 1;
    foo = 2;
}
"#,
        );
    }

    #[test]
    fn remove_multiple_underscores() {
        check_assist(
            remove_underscore,
            r#"
fn main() {
    let mut _$0_foo = 1;
    __foo = 2;
}
"#,
            r#"
fn main() {
    let mut foo = 1;
    foo = 2;
}
"#,
        );
    }

    #[test]
    fn remove_underscore_on_usage() {
        check_assist(
            remove_underscore,
            r#"
fn main() {
    let mut _foo = 1;
    _$0foo = 2;
}
"#,
            r#"
fn main() {
    let mut foo = 1;
    foo = 2;
}
"#,
        );
    }

    #[test]
    fn remove_underscore_in_function_parameter_usage() {
        check_assist(
            remove_underscore,
            r#"
fn foo(_foo: i32) {
    let bar = _$0foo + 1;
}
"#,
            r#"
fn foo(foo: i32) {
    let bar = foo + 1;
}
"#,
        )
    }

    #[test]
    fn remove_underscore_in_function_parameter() {
        check_assist(
            remove_underscore,
            r#"
fn foo(_$0foo: i32) {
    let bar = _foo + 1;
}
"#,
            r#"
fn foo(foo: i32) {
    let bar = foo + 1;
}
"#,
        )
    }
}
