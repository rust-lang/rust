use ide_db::famous_defs::FamousDefs;
use syntax::{
    AstNode,
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
//         &self[index as usize]
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

    let trait_ = impl_def.trait_()?;
    if let ast::Type::PathType(trait_path) = trait_ {
        let trait_type = ctx.sema.resolve_trait(&trait_path.path()?)?;
        let scope = ctx.sema.scope(trait_path.syntax())?;
        if trait_type != FamousDefs(&ctx.sema, scope.krate()).core_convert_Index()? {
            return None;
        }
    }

    // Index -> IndexMut
    let index_trait = impl_def
        .syntax()
        .descendants()
        .filter_map(ast::NameRef::cast)
        .find(|it| it.text() == "Index")?;
    ted::replace(
        index_trait.syntax(),
        make::path_segment(make::name_ref("IndexMut")).clone_for_update().syntax(),
    );

    // index -> index_mut
    let trait_method_name = impl_def
        .syntax()
        .descendants()
        .filter_map(ast::Name::cast)
        .find(|it| it.text() == "index")?;
    ted::replace(trait_method_name.syntax(), make::name("index_mut").clone_for_update().syntax());

    let type_alias = impl_def.syntax().descendants().find_map(ast::TypeAlias::cast)?;
    ted::remove(type_alias.syntax());

    // &self -> &mut self
    let mut_self_param = make::mut_self_param();
    let self_param: ast::SelfParam =
        impl_def.syntax().descendants().find_map(ast::SelfParam::cast)?;
    ted::replace(self_param.syntax(), mut_self_param.clone_for_update().syntax());

    // &Self::Output -> &mut Self::Output
    let ret_type = impl_def.syntax().descendants().find_map(ast::RetType::cast)?;
    ted::replace(
        ret_type.syntax(),
        make::ret_type(make::ty("&mut Self::Output")).clone_for_update().syntax(),
    );

    let fn_ = impl_def.assoc_item_list()?.assoc_items().find_map(|it| match it {
        ast::AssocItem::Fn(f) => Some(f),
        _ => None,
    })?;

    let assoc_list = make::assoc_item_list().clone_for_update();
    ted::replace(impl_def.assoc_item_list()?.syntax(), assoc_list.syntax());
    impl_def.get_or_create_assoc_item_list().add_item(syntax::ast::AssocItem::Fn(fn_));

    let target = impl_def.syntax().text_range();
    acc.add(
        AssistId::generate("generate_mut_trait_impl"),
        "Generate `IndexMut` impl from this `Index` trait",
        target,
        |edit| {
            edit.insert(target.start(), format!("$0{impl_def}\n\n{indent}"));
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

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
        &self[index as usize]
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
    }
}
