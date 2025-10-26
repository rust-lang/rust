use ide_db::{famous_defs::FamousDefs, traits::resolve_target_trait};
use syntax::{
    AstNode, T,
    ast::{self, edit_in_place::Indent, make},
    ted,
};

use crate::{AssistContext, AssistId, Assists};

// FIXME: Generate proper `index_mut` method body refer to `index` method body may impossible due to the unpredictable case [#15581].
// Here just leave the `index_mut` method body be same as `index` method body, user can modify it manually to meet their need.

// Assist: generate_mut_trait_impl
//
// Adds a IndexMut impl from the `Index` trait.
//
// ```
// # //- minicore: index
// pub enum Axis { X = 0, Y = 1, Z = 2 }
//
// impl<T> core::ops::Index$0<Axis> for [T; 3] {
//     type Output = T;
//
//     fn index(&self, index: Axis) -> &Self::Output {
//         &self[index as usize]
//     }
// }
// ```
// ->
// ```
// pub enum Axis { X = 0, Y = 1, Z = 2 }
//
// $0impl<T> core::ops::IndexMut<Axis> for [T; 3] {
//     fn index_mut(&mut self, index: Axis) -> &mut Self::Output {
//         &mut self[index as usize]
//     }
// }
//
// impl<T> core::ops::Index<Axis> for [T; 3] {
//     type Output = T;
//
//     fn index(&self, index: Axis) -> &Self::Output {
//         &self[index as usize]
//     }
// }
// ```
pub(crate) fn generate_mut_trait_impl(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let impl_def = ctx.find_node_at_offset::<ast::Impl>()?.clone_for_update();
    let indent = impl_def.indent_level();

    let ast::Type::PathType(path) = impl_def.trait_()? else {
        return None;
    };
    let trait_name = path.path()?.segment()?.name_ref()?;

    let scope = ctx.sema.scope(impl_def.trait_()?.syntax())?;
    let famous = FamousDefs(&ctx.sema, scope.krate());

    let trait_ = resolve_target_trait(&ctx.sema, &impl_def)?;
    let trait_new = get_trait_mut(&trait_, famous)?;

    // Index -> IndexMut
    ted::replace(trait_name.syntax(), make::name_ref(trait_new).clone_for_update().syntax());

    // index -> index_mut
    let (trait_method_name, new_trait_method_name) = impl_def
        .syntax()
        .descendants()
        .filter_map(ast::Name::cast)
        .find_map(process_method_name)?;
    ted::replace(
        trait_method_name.syntax(),
        make::name(new_trait_method_name).clone_for_update().syntax(),
    );

    if let Some(type_alias) = impl_def.syntax().descendants().find_map(ast::TypeAlias::cast) {
        ted::remove(type_alias.syntax());
    }

    // &self -> &mut self
    let mut_self_param = make::mut_self_param();
    let self_param: ast::SelfParam =
        impl_def.syntax().descendants().find_map(ast::SelfParam::cast)?;
    ted::replace(self_param.syntax(), mut_self_param.clone_for_update().syntax());

    // &Self::Output -> &mut Self::Output
    let ret_type = impl_def.syntax().descendants().find_map(ast::RetType::cast)?;
    let new_ret_type = process_ret_type(&ret_type)?;
    ted::replace(ret_type.syntax(), make::ret_type(new_ret_type).clone_for_update().syntax());

    let fn_ = impl_def.assoc_item_list()?.assoc_items().find_map(|it| match it {
        ast::AssocItem::Fn(f) => Some(f),
        _ => None,
    })?;
    let _ = process_ref_mut(&fn_);

    let assoc_list = make::assoc_item_list(None).clone_for_update();
    ted::replace(impl_def.assoc_item_list()?.syntax(), assoc_list.syntax());
    impl_def.get_or_create_assoc_item_list().add_item(syntax::ast::AssocItem::Fn(fn_));

    let target = impl_def.syntax().text_range();
    acc.add(
        AssistId::generate("generate_mut_trait_impl"),
        format!("Generate `{trait_new}` impl from this `{trait_name}` trait"),
        target,
        |edit| {
            edit.insert(
                target.start(),
                if ctx.config.snippet_cap.is_some() {
                    format!("$0{impl_def}\n\n{indent}")
                } else {
                    format!("{impl_def}\n\n{indent}")
                },
            );
        },
    )
}

fn process_ref_mut(fn_: &ast::Fn) -> Option<()> {
    let expr = fn_.body()?.tail_expr()?;
    match &expr {
        ast::Expr::RefExpr(ref_expr) if ref_expr.mut_token().is_none() => {
            ted::insert_all_raw(
                ted::Position::after(ref_expr.amp_token()?),
                vec![make::token(T![mut]).into(), make::tokens::whitespace(" ").into()],
            );
        }
        _ => {}
    }
    None
}

fn get_trait_mut(apply_trait: &hir::Trait, famous: FamousDefs<'_, '_>) -> Option<&'static str> {
    let trait_ = Some(apply_trait);
    if trait_ == famous.core_convert_Index().as_ref() {
        return Some("IndexMut");
    }
    if trait_ == famous.core_convert_AsRef().as_ref() {
        return Some("AsMut");
    }
    if trait_ == famous.core_borrow_Borrow().as_ref() {
        return Some("BorrowMut");
    }
    if trait_ == famous.core_ops_Deref().as_ref() {
        return Some("DerefMut");
    }
    None
}

fn process_method_name(name: ast::Name) -> Option<(ast::Name, &'static str)> {
    let new_name = match &*name.text() {
        "index" => "index_mut",
        "as_ref" => "as_mut",
        "borrow" => "borrow_mut",
        "deref" => "deref_mut",
        _ => return None,
    };
    Some((name, new_name))
}

fn process_ret_type(ref_ty: &ast::RetType) -> Option<ast::Type> {
    let ty = ref_ty.ty()?;
    let ast::Type::RefType(ref_type) = ty else {
        return None;
    };
    Some(make::ty_ref(ref_type.ty()?, true))
}

#[cfg(test)]
mod tests {
    use crate::{
        AssistConfig,
        tests::{TEST_CONFIG, check_assist, check_assist_not_applicable, check_assist_with_config},
    };

    use super::*;

    #[test]
    fn test_generate_mut_trait_impl() {
        check_assist(
            generate_mut_trait_impl,
            r#"
//- minicore: index
pub enum Axis { X = 0, Y = 1, Z = 2 }

impl<T> core::ops::Index$0<Axis> for [T; 3] {
    type Output = T;

    fn index(&self, index: Axis) -> &Self::Output {
        &self[index as usize]
    }
}
"#,
            r#"
pub enum Axis { X = 0, Y = 1, Z = 2 }

$0impl<T> core::ops::IndexMut<Axis> for [T; 3] {
    fn index_mut(&mut self, index: Axis) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

impl<T> core::ops::Index<Axis> for [T; 3] {
    type Output = T;

    fn index(&self, index: Axis) -> &Self::Output {
        &self[index as usize]
    }
}
"#,
        );

        check_assist(
            generate_mut_trait_impl,
            r#"
//- minicore: index
pub enum Axis { X = 0, Y = 1, Z = 2 }

impl<T> core::ops::Index$0<Axis> for [T; 3] where T: Copy {
    type Output = T;

    fn index(&self, index: Axis) -> &Self::Output {
        let var_name = &self[index as usize];
        var_name
    }
}
"#,
            r#"
pub enum Axis { X = 0, Y = 1, Z = 2 }

$0impl<T> core::ops::IndexMut<Axis> for [T; 3] where T: Copy {
    fn index_mut(&mut self, index: Axis) -> &mut Self::Output {
        let var_name = &self[index as usize];
        var_name
    }
}

impl<T> core::ops::Index<Axis> for [T; 3] where T: Copy {
    type Output = T;

    fn index(&self, index: Axis) -> &Self::Output {
        let var_name = &self[index as usize];
        var_name
    }
}
"#,
        );

        check_assist(
            generate_mut_trait_impl,
            r#"
//- minicore: as_ref
struct Foo(i32);

impl core::convert::AsRef$0<i32> for Foo {
    fn as_ref(&self) -> &i32 {
        &self.0
    }
}
"#,
            r#"
struct Foo(i32);

$0impl core::convert::AsMut<i32> for Foo {
    fn as_mut(&mut self) -> &mut i32 {
        &mut self.0
    }
}

impl core::convert::AsRef<i32> for Foo {
    fn as_ref(&self) -> &i32 {
        &self.0
    }
}
"#,
        );

        check_assist(
            generate_mut_trait_impl,
            r#"
//- minicore: deref
struct Foo(i32);

impl core::ops::Deref$0 for Foo {
    type Target = i32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
"#,
            r#"
struct Foo(i32);

$0impl core::ops::DerefMut for Foo {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl core::ops::Deref for Foo {
    type Target = i32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_mut_trait_impl_non_zero_indent() {
        check_assist(
            generate_mut_trait_impl,
            r#"
//- minicore: index
mod foo {
    pub enum Axis { X = 0, Y = 1, Z = 2 }

    impl<T> core::ops::Index$0<Axis> for [T; 3] where T: Copy {
        type Output = T;

        fn index(&self, index: Axis) -> &Self::Output {
            let var_name = &self[index as usize];
            var_name
        }
    }
}
"#,
            r#"
mod foo {
    pub enum Axis { X = 0, Y = 1, Z = 2 }

    $0impl<T> core::ops::IndexMut<Axis> for [T; 3] where T: Copy {
        fn index_mut(&mut self, index: Axis) -> &mut Self::Output {
            let var_name = &self[index as usize];
            var_name
        }
    }

    impl<T> core::ops::Index<Axis> for [T; 3] where T: Copy {
        type Output = T;

        fn index(&self, index: Axis) -> &Self::Output {
            let var_name = &self[index as usize];
            var_name
        }
    }
}
"#,
        );

        check_assist(
            generate_mut_trait_impl,
            r#"
//- minicore: index
mod foo {
    mod bar {
        pub enum Axis { X = 0, Y = 1, Z = 2 }

        impl<T> core::ops::Index$0<Axis> for [T; 3] where T: Copy {
            type Output = T;

            fn index(&self, index: Axis) -> &Self::Output {
                let var_name = &self[index as usize];
                var_name
            }
        }
    }
}
"#,
            r#"
mod foo {
    mod bar {
        pub enum Axis { X = 0, Y = 1, Z = 2 }

        $0impl<T> core::ops::IndexMut<Axis> for [T; 3] where T: Copy {
            fn index_mut(&mut self, index: Axis) -> &mut Self::Output {
                let var_name = &self[index as usize];
                var_name
            }
        }

        impl<T> core::ops::Index<Axis> for [T; 3] where T: Copy {
            type Output = T;

            fn index(&self, index: Axis) -> &Self::Output {
                let var_name = &self[index as usize];
                var_name
            }
        }
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_mut_trait_impl_not_applicable() {
        check_assist_not_applicable(
            generate_mut_trait_impl,
            r#"
pub trait Index<Idx: ?Sized> {}

impl<T> Index$0<i32> for [T; 3] {}
"#,
        );
        check_assist_not_applicable(
            generate_mut_trait_impl,
            r#"
pub trait AsRef<T: ?Sized> {}

impl AsRef$0<i32> for [T; 3] {}
"#,
        );
    }

    #[test]
    fn no_snippets() {
        check_assist_with_config(
            generate_mut_trait_impl,
            AssistConfig { snippet_cap: None, ..TEST_CONFIG },
            r#"
//- minicore: index
pub enum Axis { X = 0, Y = 1, Z = 2 }

impl<T> core::ops::Index$0<Axis> for [T; 3] {
    type Output = T;

    fn index(&self, index: Axis) -> &Self::Output {
        &self[index as usize]
    }
}
"#,
            r#"
pub enum Axis { X = 0, Y = 1, Z = 2 }

impl<T> core::ops::IndexMut<Axis> for [T; 3] {
    fn index_mut(&mut self, index: Axis) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

impl<T> core::ops::Index<Axis> for [T; 3] {
    type Output = T;

    fn index(&self, index: Axis) -> &Self::Output {
        &self[index as usize]
    }
}
"#,
        );
    }
}
