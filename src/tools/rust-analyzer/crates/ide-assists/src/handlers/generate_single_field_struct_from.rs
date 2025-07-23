use ast::make;
use hir::{HasCrate, ModuleDef, Semantics};
use ide_db::{
    RootDatabase, famous_defs::FamousDefs, helpers::mod_path_to_ast,
    imports::import_assets::item_for_path_search, use_trivial_constructor::use_trivial_constructor,
};
use syntax::{
    TokenText,
    ast::{self, AstNode, HasGenericParams, HasName, edit, edit_in_place::Indent},
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
    utils::add_cfg_attrs_to,
};

// Assist: generate_single_field_struct_from
//
// Implement From for a single field structure, ignore trivial types.
//
// ```
// # //- minicore: from, phantom_data
// use core::marker::PhantomData;
// struct $0Foo<T> {
//     id: i32,
//     _phantom_data: PhantomData<T>,
// }
// ```
// ->
// ```
// use core::marker::PhantomData;
// struct Foo<T> {
//     id: i32,
//     _phantom_data: PhantomData<T>,
// }
//
// impl<T> From<i32> for Foo<T> {
//     fn from(id: i32) -> Self {
//         Self { id, _phantom_data: PhantomData }
//     }
// }
// ```
pub(crate) fn generate_single_field_struct_from(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let strukt_name = ctx.find_node_at_offset::<ast::Name>()?;
    let adt = ast::Adt::cast(strukt_name.syntax().parent()?)?;
    let ast::Adt::Struct(strukt) = adt else {
        return None;
    };

    let sema = &ctx.sema;
    let (names, types) = get_fields(&strukt)?;

    let module = sema.scope(strukt.syntax())?.module();
    let constructors = make_constructors(ctx, module, &types);

    if constructors.iter().filter(|expr| expr.is_none()).count() != 1 {
        return None;
    }
    let main_field_i = constructors.iter().position(Option::is_none)?;
    if from_impl_exists(&strukt, main_field_i, &ctx.sema).is_some() {
        return None;
    }

    let main_field_name =
        names.as_ref().map_or(TokenText::borrowed("value"), |names| names[main_field_i].text());
    let main_field_ty = types[main_field_i].clone();

    acc.add(
        AssistId::generate("generate_single_field_struct_from"),
        "Generate single field `From`",
        strukt.syntax().text_range(),
        |builder| {
            let indent = strukt.indent_level();
            let ty_where_clause = strukt.where_clause();
            let type_gen_params = strukt.generic_param_list();
            let type_gen_args = type_gen_params.as_ref().map(|params| params.to_generic_args());
            let trait_gen_args = Some(make::generic_arg_list([ast::GenericArg::TypeArg(
                make::type_arg(main_field_ty.clone()),
            )]));

            let ty = make::ty(&strukt_name.text());

            let constructor =
                make_adt_constructor(names.as_deref(), constructors, &main_field_name);
            let body = make::block_expr([], Some(constructor));

            let fn_ = make::fn_(
                None,
                make::name("from"),
                None,
                None,
                make::param_list(
                    None,
                    [make::param(
                        make::path_pat(make::path_from_text(&main_field_name)),
                        main_field_ty,
                    )],
                ),
                body,
                Some(make::ret_type(make::ty("Self"))),
                false,
                false,
                false,
                false,
            )
            .clone_for_update();

            fn_.indent(1.into());

            let impl_ = make::impl_trait(
                false,
                None,
                trait_gen_args,
                type_gen_params,
                type_gen_args,
                false,
                make::ty("From"),
                ty.clone(),
                None,
                ty_where_clause.map(|wc| edit::AstNodeEdit::reset_indent(&wc)),
                None,
            )
            .clone_for_update();

            impl_.get_or_create_assoc_item_list().add_item(fn_.into());

            add_cfg_attrs_to(&strukt, &impl_);

            impl_.reindent_to(indent);

            builder.insert(strukt.syntax().text_range().end(), format!("\n\n{indent}{impl_}"));
        },
    )
}

fn make_adt_constructor(
    names: Option<&[ast::Name]>,
    constructors: Vec<Option<ast::Expr>>,
    main_field_name: &TokenText<'_>,
) -> ast::Expr {
    if let Some(names) = names {
        let fields = make::record_expr_field_list(names.iter().zip(constructors).map(
            |(name, initializer)| {
                make::record_expr_field(make::name_ref(&name.text()), initializer)
            },
        ));
        make::record_expr(make::path_from_text("Self"), fields).into()
    } else {
        let arg_list = make::arg_list(constructors.into_iter().map(|expr| {
            expr.unwrap_or_else(|| make::expr_path(make::path_from_text(main_field_name)))
        }));
        make::expr_call(make::expr_path(make::path_from_text("Self")), arg_list).into()
    }
}

fn make_constructors(
    ctx: &AssistContext<'_>,
    module: hir::Module,
    types: &[ast::Type],
) -> Vec<Option<ast::Expr>> {
    let (db, sema) = (ctx.db(), &ctx.sema);
    types
        .iter()
        .map(|ty| {
            let ty = sema.resolve_type(ty)?;
            if ty.is_unit() {
                return Some(make::expr_tuple([]).into());
            }
            let item_in_ns = ModuleDef::Adt(ty.as_adt()?).into();
            let edition = module.krate().edition(db);

            let ty_path = module.find_path(
                db,
                item_for_path_search(db, item_in_ns)?,
                ctx.config.import_path_config(),
            )?;

            use_trivial_constructor(db, mod_path_to_ast(&ty_path, edition), &ty, edition)
        })
        .collect()
}

fn get_fields(strukt: &ast::Struct) -> Option<(Option<Vec<ast::Name>>, Vec<ast::Type>)> {
    Some(match strukt.kind() {
        ast::StructKind::Unit => return None,
        ast::StructKind::Record(fields) => {
            let names = fields.fields().map(|field| field.name()).collect::<Option<_>>()?;
            let types = fields.fields().map(|field| field.ty()).collect::<Option<_>>()?;
            (Some(names), types)
        }
        ast::StructKind::Tuple(fields) => {
            (None, fields.fields().map(|field| field.ty()).collect::<Option<_>>()?)
        }
    })
}

fn from_impl_exists(
    strukt: &ast::Struct,
    main_field_i: usize,
    sema: &Semantics<'_, RootDatabase>,
) -> Option<()> {
    let db = sema.db;
    let strukt = sema.to_def(strukt)?;
    let krate = strukt.krate(db);
    let from_trait = FamousDefs(sema, krate).core_convert_From()?;
    let ty = strukt.fields(db).get(main_field_i)?.ty(db);

    strukt.ty(db).impls_trait(db, from_trait, &[ty]).then_some(())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::generate_single_field_struct_from;

    #[test]
    fn works() {
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo {
                foo: i32,
            }
            "#,
            r#"
            struct Foo {
                foo: i32,
            }

            impl From<i32> for Foo {
                fn from(foo: i32) -> Self {
                    Self { foo }
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from, phantom_data
            struct $0Foo {
                b1: (),
                b2: core::marker::PhantomData,
                foo: i32,
                a1: (),
                a2: core::marker::PhantomData,
            }
            "#,
            r#"
            struct Foo {
                b1: (),
                b2: core::marker::PhantomData,
                foo: i32,
                a1: (),
                a2: core::marker::PhantomData,
            }

            impl From<i32> for Foo {
                fn from(foo: i32) -> Self {
                    Self { b1: (), b2: core::marker::PhantomData, foo, a1: (), a2: core::marker::PhantomData }
                }
            }
            "#,
        );
    }

    #[test]
    fn cfgs() {
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            #[cfg(feature = "foo")]
            #[cfg(test)]
            struct $0Foo {
                foo: i32,
            }
            "#,
            r#"
            #[cfg(feature = "foo")]
            #[cfg(test)]
            struct Foo {
                foo: i32,
            }

            #[cfg(feature = "foo")]
            #[cfg(test)]
            impl From<i32> for Foo {
                fn from(foo: i32) -> Self {
                    Self { foo }
                }
            }
            "#,
        );
    }

    #[test]
    fn indent() {
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            mod foo {
                struct $0Foo {
                    foo: i32,
                }
            }
            "#,
            r#"
            mod foo {
                struct Foo {
                    foo: i32,
                }

                impl From<i32> for Foo {
                    fn from(foo: i32) -> Self {
                        Self { foo }
                    }
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            mod foo {
                mod bar {
                    struct $0Foo {
                        foo: i32,
                    }
                }
            }
            "#,
            r#"
            mod foo {
                mod bar {
                    struct Foo {
                        foo: i32,
                    }

                    impl From<i32> for Foo {
                        fn from(foo: i32) -> Self {
                            Self { foo }
                        }
                    }
                }
            }
            "#,
        );
    }

    #[test]
    fn where_clause_indent() {
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            mod foo {
                mod bar {
                    trait Trait {}
                    struct $0Foo<T>
                    where
                        T: Trait,
                    {
                        foo: T,
                    }
                }
            }
            "#,
            r#"
            mod foo {
                mod bar {
                    trait Trait {}
                    struct Foo<T>
                    where
                        T: Trait,
                    {
                        foo: T,
                    }

                    impl<T> From<T> for Foo<T>
                    where
                        T: Trait,
                    {
                        fn from(foo: T) -> Self {
                            Self { foo }
                        }
                    }
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            mod foo {
                mod bar {
                    trait Trait<const B: bool> {}
                    struct $0Foo<T>
                    where
                        T: Trait<{
                            true
                        }>
                    {
                        foo: T,
                    }
                }
            }
            "#,
            r#"
            mod foo {
                mod bar {
                    trait Trait<const B: bool> {}
                    struct Foo<T>
                    where
                        T: Trait<{
                            true
                        }>
                    {
                        foo: T,
                    }

                    impl<T> From<T> for Foo<T>
                    where
                        T: Trait<{
                            true
                        }>
                    {
                        fn from(foo: T) -> Self {
                            Self { foo }
                        }
                    }
                }
            }
            "#,
        );
    }

    #[test]
    fn generics() {
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T> {
                foo: T,
            }
            "#,
            r#"
            struct Foo<T> {
                foo: T,
            }

            impl<T> From<T> for Foo<T> {
                fn from(foo: T) -> Self {
                    Self { foo }
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T: Send> {
                foo: T,
            }
            "#,
            r#"
            struct Foo<T: Send> {
                foo: T,
            }

            impl<T: Send> From<T> for Foo<T> {
                fn from(foo: T) -> Self {
                    Self { foo }
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T: Send> where T: Sync,{
                foo: T,
            }
            "#,
            r#"
            struct Foo<T: Send> where T: Sync,{
                foo: T,
            }

            impl<T: Send> From<T> for Foo<T>
            where T: Sync,
            {
                fn from(foo: T) -> Self {
                    Self { foo }
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T: Send> where T: Sync {
                foo: T,
            }
            "#,
            r#"
            struct Foo<T: Send> where T: Sync {
                foo: T,
            }

            impl<T: Send> From<T> for Foo<T>
            where T: Sync
            {
                fn from(foo: T) -> Self {
                    Self { foo }
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T: Send> where T: Sync, Self: Send {
                foo: T,
            }
            "#,
            r#"
            struct Foo<T: Send> where T: Sync, Self: Send {
                foo: T,
            }

            impl<T: Send> From<T> for Foo<T>
            where T: Sync, Self: Send
            {
                fn from(foo: T) -> Self {
                    Self { foo }
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T: Send>
            where T: Sync, Self: Send
            {
                foo: T,
            }
            "#,
            r#"
            struct Foo<T: Send>
            where T: Sync, Self: Send
            {
                foo: T,
            }

            impl<T: Send> From<T> for Foo<T>
            where T: Sync, Self: Send
            {
                fn from(foo: T) -> Self {
                    Self { foo }
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T: Send>
            where T: Sync, Self: Send,
            {
                foo: T,
            }
            "#,
            r#"
            struct Foo<T: Send>
            where T: Sync, Self: Send,
            {
                foo: T,
            }

            impl<T: Send> From<T> for Foo<T>
            where T: Sync, Self: Send,
            {
                fn from(foo: T) -> Self {
                    Self { foo }
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T: Send>
            where T: Sync,
                  Self: Send,
            {
                foo: T,
            }
            "#,
            r#"
            struct Foo<T: Send>
            where T: Sync,
                  Self: Send,
            {
                foo: T,
            }

            impl<T: Send> From<T> for Foo<T>
            where T: Sync,
                  Self: Send,
            {
                fn from(foo: T) -> Self {
                    Self { foo }
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T: Send>
            where
                T: Sync,
                Self: Send,
            {
                foo: T,
            }
            "#,
            r#"
            struct Foo<T: Send>
            where
                T: Sync,
                Self: Send,
            {
                foo: T,
            }

            impl<T: Send> From<T> for Foo<T>
            where
                T: Sync,
                Self: Send,
            {
                fn from(foo: T) -> Self {
                    Self { foo }
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T: Send + Sync>
            where
                T: Sync,
                Self: Send,
            {
                foo: T,
            }
            "#,
            r#"
            struct Foo<T: Send + Sync>
            where
                T: Sync,
                Self: Send,
            {
                foo: T,
            }

            impl<T: Send + Sync> From<T> for Foo<T>
            where
                T: Sync,
                Self: Send,
            {
                fn from(foo: T) -> Self {
                    Self { foo }
                }
            }
            "#,
        );
    }

    #[test]
    fn tuple() {
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo(i32);
            "#,
            r#"
            struct Foo(i32);

            impl From<i32> for Foo {
                fn from(value: i32) -> Self {
                    Self(value)
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T>(T);
            "#,
            r#"
            struct Foo<T>(T);

            impl<T> From<T> for Foo<T> {
                fn from(value: T) -> Self {
                    Self(value)
                }
            }
            "#,
        );
    }

    #[test]
    fn trivial() {
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from, phantom_data
            use core::marker::PhantomData;
            struct $0Foo(i32, PhantomData<i32>);
            "#,
            r#"
            use core::marker::PhantomData;
            struct Foo(i32, PhantomData<i32>);

            impl From<i32> for Foo {
                fn from(value: i32) -> Self {
                    Self(value, PhantomData)
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from, phantom_data
            use core::marker::PhantomData;
            struct $0Foo(i32, PhantomData<()>);
            "#,
            r#"
            use core::marker::PhantomData;
            struct Foo(i32, PhantomData<()>);

            impl From<i32> for Foo {
                fn from(value: i32) -> Self {
                    Self(value, PhantomData)
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from, phantom_data
            use core::marker::PhantomData;
            struct $0Foo(PhantomData<()>, i32, PhantomData<()>);
            "#,
            r#"
            use core::marker::PhantomData;
            struct Foo(PhantomData<()>, i32, PhantomData<()>);

            impl From<i32> for Foo {
                fn from(value: i32) -> Self {
                    Self(PhantomData, value, PhantomData)
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from, phantom_data
            use core::marker::PhantomData;
            struct $0Foo<T>(PhantomData<T>, i32, PhantomData<()>);
            "#,
            r#"
            use core::marker::PhantomData;
            struct Foo<T>(PhantomData<T>, i32, PhantomData<()>);

            impl<T> From<i32> for Foo<T> {
                fn from(value: i32) -> Self {
                    Self(PhantomData, value, PhantomData)
                }
            }
            "#,
        );
    }

    #[test]
    fn unit() {
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo(i32, ());
            "#,
            r#"
            struct Foo(i32, ());

            impl From<i32> for Foo {
                fn from(value: i32) -> Self {
                    Self(value, ())
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo((), i32, ());
            "#,
            r#"
            struct Foo((), i32, ());

            impl From<i32> for Foo {
                fn from(value: i32) -> Self {
                    Self((), value, ())
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo((), (), i32, ());
            "#,
            r#"
            struct Foo((), (), i32, ());

            impl From<i32> for Foo {
                fn from(value: i32) -> Self {
                    Self((), (), value, ())
                }
            }
            "#,
        );
    }

    #[test]
    fn invalid_multiple_main_field() {
        check_assist_not_applicable(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo(i32, i32);
            "#,
        );
        check_assist_not_applicable(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T>(i32, T);
            "#,
        );
        check_assist_not_applicable(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T>(T, T);
            "#,
        );
        check_assist_not_applicable(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo<T> { foo: T, bar: i32 }
            "#,
        );
        check_assist_not_applicable(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo { foo: i32, bar: i64 }
            "#,
        );
    }

    #[test]
    fn exists_other_from() {
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo(i32);

            impl From<&i32> for Foo {
                fn from(value: &i32) -> Self {
                    todo!()
                }
            }
            "#,
            r#"
            struct Foo(i32);

            impl From<i32> for Foo {
                fn from(value: i32) -> Self {
                    Self(value)
                }
            }

            impl From<&i32> for Foo {
                fn from(value: &i32) -> Self {
                    todo!()
                }
            }
            "#,
        );
        check_assist(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo(i32);

            type X = i32;

            impl From<&X> for Foo {
                fn from(value: &X) -> Self {
                    todo!()
                }
            }
            "#,
            r#"
            struct Foo(i32);

            impl From<i32> for Foo {
                fn from(value: i32) -> Self {
                    Self(value)
                }
            }

            type X = i32;

            impl From<&X> for Foo {
                fn from(value: &X) -> Self {
                    todo!()
                }
            }
            "#,
        );
    }

    #[test]
    fn exists_from() {
        check_assist_not_applicable(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo(i32);

            impl From<i32> for Foo {
                fn from(_: i32) -> Self {
                    todo!()
                }
            }
            "#,
        );
        check_assist_not_applicable(
            generate_single_field_struct_from,
            r#"
            //- minicore: from
            struct $0Foo(i32);

            type X = i32;

            impl From<X> for Foo {
                fn from(_: X) -> Self {
                    todo!()
                }
            }
            "#,
        );
    }
}
