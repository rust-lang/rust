use hir::{AsAssocItem, HirDisplay};
use ide_db::{
    assists::{AssistId, AssistKind},
    famous_defs::FamousDefs,
};
use syntax::{ast, AstNode};

use crate::assist_context::{AssistContext, Assists};

// Assist: into_to_qualified_from
//
// Convert an `into` method call to a fully qualified `from` call.
//
// ```
// //- minicore: from
// struct B;
// impl From<i32> for B {
//     fn from(a: i32) -> Self {
//        B
//     }
// }
//
// fn main() -> () {
//     let a = 3;
//     let b: B = a.in$0to();
// }
// ```
// ->
// ```
// struct B;
// impl From<i32> for B {
//     fn from(a: i32) -> Self {
//        B
//     }
// }
//
// fn main() -> () {
//     let a = 3;
//     let b: B = B::from(a);
// }
// ```
pub(crate) fn into_to_qualified_from(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let method_call: ast::MethodCallExpr = ctx.find_node_at_offset()?;
    let nameref = method_call.name_ref()?;
    let receiver = method_call.receiver()?;
    let db = ctx.db();
    let sema = &ctx.sema;
    let fnc = sema.resolve_method_call(&method_call)?;
    let scope = sema.scope(method_call.syntax())?;
    // Check if the method call refers to Into trait.
    if fnc.as_assoc_item(db)?.containing_trait_impl(db)?
        == FamousDefs(sema, scope.krate()).core_convert_Into()?
    {
        let type_call = sema.type_of_expr(&method_call.clone().into())?;
        let type_call_disp =
            type_call.adjusted().display_source_code(db, scope.module().into(), true).ok()?;

        acc.add(
            AssistId("into_to_qualified_from", AssistKind::Generate),
            "Convert `into` to fully qualified `from`",
            nameref.syntax().text_range(),
            |edit| {
                edit.replace(
                    method_call.syntax().text_range(),
                    format!("{}::from({})", type_call_disp, receiver),
                );
            },
        );
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::into_to_qualified_from;

    #[test]
    fn two_types_in_same_mod() {
        check_assist(
            into_to_qualified_from,
            r#"
//- minicore: from
struct A;
struct B;
impl From<A> for B {
    fn from(a: A) -> Self {
        B
    }
}

fn main() -> () {
    let a: A = A;
    let b: B = a.in$0to();
}"#,
            r#"
struct A;
struct B;
impl From<A> for B {
    fn from(a: A) -> Self {
        B
    }
}

fn main() -> () {
    let a: A = A;
    let b: B = B::from(a);
}"#,
        )
    }

    #[test]
    fn fromed_in_child_mod_imported() {
        check_assist(
            into_to_qualified_from,
            r#"
//- minicore: from
use C::B;

struct A;

mod C {
    use crate::A;

    pub(super) struct B;
    impl From<A> for B {
        fn from(a: A) -> Self {
            B
        }
    }
}

fn main() -> () {
    let a: A = A;
    let b: B = a.in$0to();
}"#,
            r#"
use C::B;

struct A;

mod C {
    use crate::A;

    pub(super) struct B;
    impl From<A> for B {
        fn from(a: A) -> Self {
            B
        }
    }
}

fn main() -> () {
    let a: A = A;
    let b: B = B::from(a);
}"#,
        )
    }

    #[test]
    fn fromed_in_child_mod_not_imported() {
        check_assist(
            into_to_qualified_from,
            r#"
//- minicore: from
struct A;

mod C {
    use crate::A;

    pub(super) struct B;
    impl From<A> for B {
        fn from(a: A) -> Self {
            B
        }
    }
}

fn main() -> () {
    let a: A = A;
    let b: C::B = a.in$0to();
}"#,
            r#"
struct A;

mod C {
    use crate::A;

    pub(super) struct B;
    impl From<A> for B {
        fn from(a: A) -> Self {
            B
        }
    }
}

fn main() -> () {
    let a: A = A;
    let b: C::B = C::B::from(a);
}"#,
        )
    }
}
