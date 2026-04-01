use hir::{AsAssocItem, HirDisplay};
use ide_db::{assists::AssistId, famous_defs::FamousDefs};
use syntax::{AstNode, ast};

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
    if fnc.as_assoc_item(db)?.implemented_trait(db)?
        == FamousDefs(sema, scope.krate()).core_convert_Into()?
    {
        let type_call = sema.type_of_expr(&method_call.clone().into())?;
        let adjusted_tc = type_call.adjusted();

        if adjusted_tc.contains_unknown() {
            return None;
        }

        let sc = adjusted_tc.display_source_code(db, scope.module().into(), true).ok()?;
        acc.add(
            AssistId::generate("into_to_qualified_from"),
            "Convert `into` to fully qualified `from`",
            nameref.syntax().text_range(),
            |edit| {
                edit.replace(
                    method_call.syntax().text_range(),
                    if sc.chars().all(|c| c.is_alphanumeric() || c == ':') {
                        format!("{sc}::from({receiver})")
                    } else {
                        format!("<{sc}>::from({receiver})")
                    },
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
    fn from_in_child_mod_imported() {
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
    fn from_in_child_mod_not_imported() {
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

    #[test]
    fn preceding_type_qualifier() {
        check_assist(
            into_to_qualified_from,
            r#"
//- minicore: from
impl From<(i32,i32)> for [i32;2] {
    fn from(value: (i32,i32)) -> Self {
        [value.0, value.1]
    }
}

fn tuple_to_array() -> [i32; 2] {
    (0,1).in$0to()
}"#,
            r#"
impl From<(i32,i32)> for [i32;2] {
    fn from(value: (i32,i32)) -> Self {
        [value.0, value.1]
    }
}

fn tuple_to_array() -> [i32; 2] {
    <[i32; 2]>::from((0,1))
}"#,
        )
    }

    #[test]
    fn type_with_gens() {
        check_assist(
            into_to_qualified_from,
            r#"
//- minicore: from
struct StructA<Gen>(Gen);

impl From<i32> for StructA<i32> {
    fn from(value: i32) -> Self {
        StructA(value + 1)
    }
}

fn main() -> () {
    let a: StructA<i32> = 3.in$0to();
}"#,
            r#"
struct StructA<Gen>(Gen);

impl From<i32> for StructA<i32> {
    fn from(value: i32) -> Self {
        StructA(value + 1)
    }
}

fn main() -> () {
    let a: StructA<i32> = <StructA<i32>>::from(3);
}"#,
        )
    }
}
