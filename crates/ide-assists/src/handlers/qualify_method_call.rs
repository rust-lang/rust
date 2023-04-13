use hir::{db::HirDatabase, AsAssocItem, AssocItem, AssocItemContainer, ItemInNs, ModuleDef};
use ide_db::assists::{AssistId, AssistKind};
use syntax::{ast, AstNode};

use crate::{
    assist_context::{AssistContext, Assists},
    handlers::qualify_path::QualifyCandidate,
};

// Assist: qualify_method_call
//
// Replaces the method call with a qualified function call.
//
// ```
// struct Foo;
// impl Foo {
//     fn foo(&self) {}
// }
// fn main() {
//     let foo = Foo;
//     foo.fo$0o();
// }
// ```
// ->
// ```
// struct Foo;
// impl Foo {
//     fn foo(&self) {}
// }
// fn main() {
//     let foo = Foo;
//     Foo::foo(&foo);
// }
// ```
pub(crate) fn qualify_method_call(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let name: ast::NameRef = ctx.find_node_at_offset()?;
    let call = name.syntax().parent().and_then(ast::MethodCallExpr::cast)?;

    let ident = name.ident_token()?;

    let range = call.syntax().text_range();
    let resolved_call = ctx.sema.resolve_method_call(&call)?;

    let current_module = ctx.sema.scope(call.syntax())?.module();
    let target_module_def = ModuleDef::from(resolved_call);
    let item_in_ns = ItemInNs::from(target_module_def);
    let receiver_path = current_module.find_use_path(
        ctx.sema.db,
        item_for_path_search(ctx.sema.db, item_in_ns)?,
        ctx.config.prefer_no_std,
    )?;

    let qualify_candidate = QualifyCandidate::ImplMethod(ctx.sema.db, call, resolved_call);

    acc.add(
        AssistId("qualify_method_call", AssistKind::RefactorRewrite),
        format!("Qualify `{ident}` method call"),
        range,
        |builder| {
            qualify_candidate.qualify(
                |replace_with: String| builder.replace(range, replace_with),
                &receiver_path,
                item_in_ns,
            )
        },
    );
    Some(())
}

fn item_for_path_search(db: &dyn HirDatabase, item: ItemInNs) -> Option<ItemInNs> {
    Some(match item {
        ItemInNs::Types(_) | ItemInNs::Values(_) => match item_as_assoc(db, item) {
            Some(assoc_item) => match assoc_item.container(db) {
                AssocItemContainer::Trait(trait_) => ItemInNs::from(ModuleDef::from(trait_)),
                AssocItemContainer::Impl(impl_) => match impl_.trait_(db) {
                    None => ItemInNs::from(ModuleDef::from(impl_.self_ty(db).as_adt()?)),
                    Some(trait_) => ItemInNs::from(ModuleDef::from(trait_)),
                },
            },
            None => item,
        },
        ItemInNs::Macros(_) => item,
    })
}

fn item_as_assoc(db: &dyn HirDatabase, item: ItemInNs) -> Option<AssocItem> {
    item.as_module_def().and_then(|module_def| module_def.as_assoc_item(db))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn struct_method() {
        check_assist(
            qualify_method_call,
            r#"
struct Foo;
impl Foo {
    fn foo(&self) {}
}

fn main() {
    let foo = Foo {};
    foo.fo$0o()
}
"#,
            r#"
struct Foo;
impl Foo {
    fn foo(&self) {}
}

fn main() {
    let foo = Foo {};
    Foo::foo(&foo)
}
"#,
        );
    }

    #[test]
    fn struct_method_multi_params() {
        check_assist(
            qualify_method_call,
            r#"
struct Foo;
impl Foo {
    fn foo(&self, p1: i32, p2: u32) {}
}

fn main() {
    let foo = Foo {};
    foo.fo$0o(9, 9u)
}
"#,
            r#"
struct Foo;
impl Foo {
    fn foo(&self, p1: i32, p2: u32) {}
}

fn main() {
    let foo = Foo {};
    Foo::foo(&foo, 9, 9u)
}
"#,
        );
    }

    #[test]
    fn struct_method_consume() {
        check_assist(
            qualify_method_call,
            r#"
struct Foo;
impl Foo {
    fn foo(self, p1: i32, p2: u32) {}
}

fn main() {
    let foo = Foo {};
    foo.fo$0o(9, 9u)
}
"#,
            r#"
struct Foo;
impl Foo {
    fn foo(self, p1: i32, p2: u32) {}
}

fn main() {
    let foo = Foo {};
    Foo::foo(foo, 9, 9u)
}
"#,
        );
    }

    #[test]
    fn struct_method_exclusive() {
        check_assist(
            qualify_method_call,
            r#"
struct Foo;
impl Foo {
    fn foo(&mut self, p1: i32, p2: u32) {}
}

fn main() {
    let foo = Foo {};
    foo.fo$0o(9, 9u)
}
"#,
            r#"
struct Foo;
impl Foo {
    fn foo(&mut self, p1: i32, p2: u32) {}
}

fn main() {
    let foo = Foo {};
    Foo::foo(&mut foo, 9, 9u)
}
"#,
        );
    }

    #[test]
    fn struct_method_cross_crate() {
        check_assist(
            qualify_method_call,
            r#"
//- /main.rs crate:main deps:dep
fn main() {
    let foo = dep::test_mod::Foo {};
    foo.fo$0o(9, 9u)
}
//- /dep.rs crate:dep
pub mod test_mod {
    pub struct Foo;
    impl Foo {
        pub fn foo(&mut self, p1: i32, p2: u32) {}
    }
}
"#,
            r#"
fn main() {
    let foo = dep::test_mod::Foo {};
    dep::test_mod::Foo::foo(&mut foo, 9, 9u)
}
"#,
        );
    }

    #[test]
    fn struct_method_generic() {
        check_assist(
            qualify_method_call,
            r#"
struct Foo;
impl Foo {
    fn foo<T>(&self) {}
}

fn main() {
    let foo = Foo {};
    foo.fo$0o::<()>()
}
"#,
            r#"
struct Foo;
impl Foo {
    fn foo<T>(&self) {}
}

fn main() {
    let foo = Foo {};
    Foo::foo::<()>(&foo)
}
"#,
        );
    }

    #[test]
    fn trait_method() {
        check_assist(
            qualify_method_call,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(&self) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    test_struct.test_meth$0od()
}
"#,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(&self) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    TestTrait::test_method(&test_struct)
}
"#,
        );
    }

    #[test]
    fn trait_method_multi_params() {
        check_assist(
            qualify_method_call,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(&self, p1: i32, p2: u32);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(&self, p1: i32, p2: u32) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    test_struct.test_meth$0od(12, 32u)
}
"#,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(&self, p1: i32, p2: u32);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(&self, p1: i32, p2: u32) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    TestTrait::test_method(&test_struct, 12, 32u)
}
"#,
        );
    }

    #[test]
    fn trait_method_consume() {
        check_assist(
            qualify_method_call,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(self, p1: i32, p2: u32);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(self, p1: i32, p2: u32) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    test_struct.test_meth$0od(12, 32u)
}
"#,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(self, p1: i32, p2: u32);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(self, p1: i32, p2: u32) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    TestTrait::test_method(test_struct, 12, 32u)
}
"#,
        );
    }

    #[test]
    fn trait_method_exclusive() {
        check_assist(
            qualify_method_call,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(&mut self, p1: i32, p2: u32);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(&mut self, p1: i32, p2: u32);
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    test_struct.test_meth$0od(12, 32u)
}
"#,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(&mut self, p1: i32, p2: u32);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(&mut self, p1: i32, p2: u32);
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    TestTrait::test_method(&mut test_struct, 12, 32u)
}
"#,
        );
    }

    #[test]
    fn trait_method_cross_crate() {
        check_assist(
            qualify_method_call,
            r#"
//- /main.rs crate:main deps:dep
fn main() {
    let foo = dep::test_mod::Foo {};
    foo.fo$0o(9, 9u)
}
//- /dep.rs crate:dep
pub mod test_mod {
    pub struct Foo;
    impl Foo {
        pub fn foo(&mut self, p1: i32, p2: u32) {}
    }
}
"#,
            r#"
fn main() {
    let foo = dep::test_mod::Foo {};
    dep::test_mod::Foo::foo(&mut foo, 9, 9u)
}
"#,
        );
    }

    #[test]
    fn trait_method_generic() {
        check_assist(
            qualify_method_call,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method<T>(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method<T>(&self) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = TestStruct {};
    test_struct.test_meth$0od::<()>()
}
"#,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method<T>(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method<T>(&self) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = TestStruct {};
    TestTrait::test_method::<()>(&test_struct)
}
"#,
        );
    }

    #[test]
    fn struct_method_over_struct_instance() {
        check_assist_not_applicable(
            qualify_method_call,
            r#"
struct Foo;
impl Foo {
    fn foo(&self) {}
}

fn main() {
    let foo = Foo {};
    f$0oo.foo()
}
"#,
        );
    }

    #[test]
    fn trait_method_over_struct_instance() {
        check_assist_not_applicable(
            qualify_method_call,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(&self) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    tes$0t_struct.test_method()
}
"#,
        );
    }
}
