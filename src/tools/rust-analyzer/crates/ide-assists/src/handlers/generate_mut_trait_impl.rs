use ide_db::{famous_defs::FamousDefs, traits::resolve_target_trait};
use syntax::{
    AstNode, SyntaxElement, SyntaxNode, T,
    ast::{self, edit::AstNodeEdit, edit_in_place::Indent, syntax_factory::SyntaxFactory},
    syntax_editor::{Element, Position, SyntaxEditor},
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
    let impl_def = ctx.find_node_at_offset::<ast::Impl>()?;
    let indent = Indent::indent_level(&impl_def);

    let ast::Type::PathType(path) = impl_def.trait_()? else {
        return None;
    };

    let trait_name = path.path()?.segment()?.name_ref()?;

    let scope = ctx.sema.scope(impl_def.trait_()?.syntax())?;
    let famous = FamousDefs(&ctx.sema, scope.krate());

    let trait_ = resolve_target_trait(&ctx.sema, &impl_def)?;
    let trait_new = get_trait_mut(&trait_, famous)?;

    let target = impl_def.syntax().text_range();

    acc.add(
        AssistId::generate("generate_mut_trait_impl"),
        format!("Generate `{trait_new}` impl from this `{trait_name}` trait"),
        target,
        |edit| {
            let impl_clone = impl_def.reset_indent().clone_subtree();
            let mut editor = SyntaxEditor::new(impl_clone.syntax().clone());
            let factory = SyntaxFactory::without_mappings();

            apply_generate_mut_impl(&mut editor, &factory, &impl_clone, trait_new);

            let new_root = editor.finish();
            let new_root = new_root.new_root();

            let new_impl = ast::Impl::cast(new_root.clone()).unwrap();

            Indent::indent(&new_impl, indent);

            let mut editor = edit.make_editor(impl_def.syntax());
            editor.insert_all(
                Position::before(impl_def.syntax()),
                vec![
                    new_impl.syntax().syntax_element(),
                    factory.whitespace(&format!("\n\n{indent}")).syntax_element(),
                ],
            );

            if let Some(cap) = ctx.config.snippet_cap {
                let tabstop_before = edit.make_tabstop_before(cap);
                editor.add_annotation(new_impl.syntax(), tabstop_before);
            }

            edit.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

fn delete_with_trivia(editor: &mut SyntaxEditor, node: &SyntaxNode) {
    let mut end: SyntaxElement = node.clone().into();

    if let Some(next) = node.next_sibling_or_token()
        && let SyntaxElement::Token(tok) = &next
        && tok.kind().is_trivia()
    {
        end = next.clone();
    }

    editor.delete_all(node.clone().into()..=end);
}

fn apply_generate_mut_impl(
    editor: &mut SyntaxEditor,
    factory: &SyntaxFactory,
    impl_def: &ast::Impl,
    trait_new: &str,
) -> Option<()> {
    let path =
        impl_def.trait_().and_then(|t| t.syntax().descendants().find_map(ast::Path::cast))?;
    let seg = path.segment()?;
    let name_ref = seg.name_ref()?;

    let new_name_ref = factory.name_ref(trait_new);
    editor.replace(name_ref.syntax(), new_name_ref.syntax());

    if let Some((name, new_name)) =
        impl_def.syntax().descendants().filter_map(ast::Name::cast).find_map(process_method_name)
    {
        let new_name_node = factory.name(new_name);
        editor.replace(name.syntax(), new_name_node.syntax());
    }

    if let Some(type_alias) = impl_def.syntax().descendants().find_map(ast::TypeAlias::cast) {
        delete_with_trivia(editor, type_alias.syntax());
    }

    if let Some(self_param) = impl_def.syntax().descendants().find_map(ast::SelfParam::cast) {
        let mut_self = factory.mut_self_param();
        editor.replace(self_param.syntax(), mut_self.syntax());
    }

    if let Some(ret_type) = impl_def.syntax().descendants().find_map(ast::RetType::cast)
        && let Some(new_ty) = process_ret_type(factory, &ret_type)
    {
        let new_ret = factory.ret_type(new_ty);
        editor.replace(ret_type.syntax(), new_ret.syntax())
    }

    if let Some(fn_) = impl_def.assoc_item_list().and_then(|l| {
        l.assoc_items().find_map(|it| match it {
            ast::AssocItem::Fn(f) => Some(f),
            _ => None,
        })
    }) {
        process_ref_mut(editor, factory, &fn_);
    }

    Some(())
}

fn process_ref_mut(editor: &mut SyntaxEditor, factory: &SyntaxFactory, fn_: &ast::Fn) {
    let Some(expr) = fn_.body().and_then(|b| b.tail_expr()) else { return };

    let ast::Expr::RefExpr(ref_expr) = expr else { return };

    if ref_expr.mut_token().is_some() {
        return;
    }

    let Some(amp) = ref_expr.amp_token() else { return };

    let mut_kw = factory.token(T![mut]);
    let space = factory.whitespace(" ");

    editor.insert(Position::after(amp.clone()), space.syntax_element());
    editor.insert(Position::after(amp), mut_kw.syntax_element());
}

fn process_ret_type(factory: &SyntaxFactory, ref_ty: &ast::RetType) -> Option<ast::Type> {
    let ty = ref_ty.ty()?;
    let ast::Type::RefType(ref_type) = ty else {
        return None;
    };

    let inner = ref_type.ty()?;
    Some(factory.ty_ref(inner, true))
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
