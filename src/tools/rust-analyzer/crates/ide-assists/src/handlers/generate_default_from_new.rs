use ide_db::famous_defs::FamousDefs;
use stdx::format_to;
use syntax::{
    AstNode,
    ast::{self, HasGenericParams, HasName, HasTypeBounds, Impl, make},
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
};

// Assist: generate_default_from_new
//
// Generates default implementation from new method.
//
// ```
// # //- minicore: default
// struct Example { _inner: () }
//
// impl Example {
//     pub fn n$0ew() -> Self {
//         Self { _inner: () }
//     }
// }
// ```
// ->
// ```
// struct Example { _inner: () }
//
// impl Example {
//     pub fn new() -> Self {
//         Self { _inner: () }
//     }
// }
//
// impl Default for Example {
//     fn default() -> Self {
//         Self::new()
//     }
// }
// ```
pub(crate) fn generate_default_from_new(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let fn_node = ctx.find_node_at_offset::<ast::Fn>()?;
    let fn_name = fn_node.name()?;

    if fn_name.text() != "new" {
        cov_mark::hit!(other_function_than_new);
        return None;
    }

    if fn_node.param_list()?.params().next().is_some() {
        cov_mark::hit!(new_function_with_parameters);
        return None;
    }

    let impl_ = fn_node.syntax().ancestors().find_map(ast::Impl::cast)?;
    let self_ty = impl_.self_ty()?;
    if is_default_implemented(ctx, &impl_) {
        cov_mark::hit!(default_block_is_already_present);
        cov_mark::hit!(struct_in_module_with_default);
        return None;
    }

    let insert_location = impl_.syntax().text_range();

    acc.add(
        AssistId::generate("generate_default_from_new"),
        "Generate a Default impl from a new fn",
        insert_location,
        move |builder| {
            let default_code = "    fn default() -> Self {
        Self::new()
    }";
            let code = generate_trait_impl_text_from_impl(&impl_, self_ty, "Default", default_code);
            builder.insert(insert_location.end(), code);
        },
    )
}

// FIXME: based on from utils::generate_impl_text_inner
fn generate_trait_impl_text_from_impl(
    impl_: &ast::Impl,
    self_ty: ast::Type,
    trait_text: &str,
    code: &str,
) -> String {
    let generic_params = impl_.generic_param_list().map(|generic_params| {
        let lifetime_params =
            generic_params.lifetime_params().map(ast::GenericParam::LifetimeParam);
        let ty_or_const_params = generic_params.type_or_const_params().filter_map(|param| {
            // remove defaults since they can't be specified in impls
            let param = match param {
                ast::TypeOrConstParam::Type(param) => {
                    let param = make::type_param(param.name()?, param.type_bound_list());
                    ast::GenericParam::TypeParam(param)
                }
                ast::TypeOrConstParam::Const(param) => {
                    let param = make::const_param(param.name()?, param.ty()?);
                    ast::GenericParam::ConstParam(param)
                }
            };
            Some(param)
        });

        make::generic_param_list(itertools::chain(lifetime_params, ty_or_const_params))
    });

    let mut buf = String::with_capacity(code.len());
    buf.push_str("\n\n");

    // `impl{generic_params} {trait_text} for {impl_.self_ty()}`
    buf.push_str("impl");
    if let Some(generic_params) = &generic_params {
        format_to!(buf, "{generic_params}")
    }
    format_to!(buf, " {trait_text} for {self_ty}");

    match impl_.where_clause() {
        Some(where_clause) => {
            format_to!(buf, "\n{where_clause}\n{{\n{code}\n}}");
        }
        None => {
            format_to!(buf, " {{\n{code}\n}}");
        }
    }

    buf
}

fn is_default_implemented(ctx: &AssistContext<'_>, impl_: &Impl) -> bool {
    let db = ctx.sema.db;
    let impl_ = ctx.sema.to_def(impl_);
    let impl_def = match impl_ {
        Some(value) => value,
        None => return false,
    };

    let ty = impl_def.self_ty(db);
    let krate = impl_def.module(db).krate();
    let default = FamousDefs(&ctx.sema, krate).core_default_Default();
    let default_trait = match default {
        Some(value) => value,
        // Return `true` to avoid providing the assist because it makes no sense
        // to impl `Default` when it's missing.
        None => return true,
    };

    ty.impls_trait(db, default_trait, &[])
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn generate_default() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
struct Example { _inner: () }

impl Example {
    pub fn ne$0w() -> Self {
        Self { _inner: () }
    }
}

fn main() {}
"#,
            r#"
struct Example { _inner: () }

impl Example {
    pub fn new() -> Self {
        Self { _inner: () }
    }
}

impl Default for Example {
    fn default() -> Self {
        Self::new()
    }
}

fn main() {}
"#,
        );
    }

    #[test]
    fn generate_default2() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
struct Test { value: u32 }

impl Test {
    pub fn ne$0w() -> Self {
        Self { value: 0 }
    }
}
"#,
            r#"
struct Test { value: u32 }

impl Test {
    pub fn new() -> Self {
        Self { value: 0 }
    }
}

impl Default for Test {
    fn default() -> Self {
        Self::new()
    }
}
"#,
        );
    }

    #[test]
    fn new_function_with_generic() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
pub struct Foo<T> {
    _bar: *mut T,
}

impl<T> Foo<T> {
    pub fn ne$0w() -> Self {
        unimplemented!()
    }
}
"#,
            r#"
pub struct Foo<T> {
    _bar: *mut T,
}

impl<T> Foo<T> {
    pub fn new() -> Self {
        unimplemented!()
    }
}

impl<T> Default for Foo<T> {
    fn default() -> Self {
        Self::new()
    }
}
"#,
        );
    }

    #[test]
    fn new_function_with_generics() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
pub struct Foo<T, B> {
    _tars: *mut T,
    _bar: *mut B,
}

impl<T, B> Foo<T, B> {
    pub fn ne$0w() -> Self {
        unimplemented!()
    }
}
"#,
            r#"
pub struct Foo<T, B> {
    _tars: *mut T,
    _bar: *mut B,
}

impl<T, B> Foo<T, B> {
    pub fn new() -> Self {
        unimplemented!()
    }
}

impl<T, B> Default for Foo<T, B> {
    fn default() -> Self {
        Self::new()
    }
}
"#,
        );
    }

    #[test]
    fn new_function_with_generic_and_bound() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
pub struct Foo<T> {
    t: T,
}

impl<T: From<i32>> Foo<T> {
    pub fn ne$0w() -> Self {
        Foo { t: 0.into() }
    }
}
"#,
            r#"
pub struct Foo<T> {
    t: T,
}

impl<T: From<i32>> Foo<T> {
    pub fn new() -> Self {
        Foo { t: 0.into() }
    }
}

impl<T: From<i32>> Default for Foo<T> {
    fn default() -> Self {
        Self::new()
    }
}
"#,
        );
    }

    #[test]
    fn new_function_with_generics_and_bounds() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
pub struct Foo<T, B> {
    _tars: T,
    _bar: B,
}

impl<T: From<i32>, B: From<i64>> Foo<T, B> {
    pub fn ne$0w() -> Self {
        unimplemented!()
    }
}
"#,
            r#"
pub struct Foo<T, B> {
    _tars: T,
    _bar: B,
}

impl<T: From<i32>, B: From<i64>> Foo<T, B> {
    pub fn new() -> Self {
        unimplemented!()
    }
}

impl<T: From<i32>, B: From<i64>> Default for Foo<T, B> {
    fn default() -> Self {
        Self::new()
    }
}
"#,
        );
    }

    #[test]
    fn new_function_with_generic_and_where() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
pub struct Foo<T> {
    t: T,
}

impl<T: From<i32>> Foo<T>
where
    Option<T>: Debug
{
    pub fn ne$0w() -> Self {
        Foo { t: 0.into() }
    }
}
"#,
            r#"
pub struct Foo<T> {
    t: T,
}

impl<T: From<i32>> Foo<T>
where
    Option<T>: Debug
{
    pub fn new() -> Self {
        Foo { t: 0.into() }
    }
}

impl<T: From<i32>> Default for Foo<T>
where
    Option<T>: Debug
{
    fn default() -> Self {
        Self::new()
    }
}
"#,
        );
    }

    #[test]
    fn new_function_with_generics_and_where() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
pub struct Foo<T, B> {
    _tars: T,
    _bar: B,
}

impl<T: From<i32>, B: From<i64>> Foo<T, B>
where
    Option<T>: Debug, Option<B>: Debug,
{
    pub fn ne$0w() -> Self {
        unimplemented!()
    }
}
"#,
            r#"
pub struct Foo<T, B> {
    _tars: T,
    _bar: B,
}

impl<T: From<i32>, B: From<i64>> Foo<T, B>
where
    Option<T>: Debug, Option<B>: Debug,
{
    pub fn new() -> Self {
        unimplemented!()
    }
}

impl<T: From<i32>, B: From<i64>> Default for Foo<T, B>
where
    Option<T>: Debug, Option<B>: Debug,
{
    fn default() -> Self {
        Self::new()
    }
}
"#,
        );
    }

    #[test]
    fn new_function_with_parameters() {
        cov_mark::check!(new_function_with_parameters);
        check_assist_not_applicable(
            generate_default_from_new,
            r#"
//- minicore: default
struct Example { _inner: () }

impl Example {
    pub fn $0new(value: ()) -> Self {
        Self { _inner: value }
    }
}
"#,
        );
    }

    #[test]
    fn other_function_than_new() {
        cov_mark::check!(other_function_than_new);
        check_assist_not_applicable(
            generate_default_from_new,
            r#"
//- minicore: default
struct Example { _inner: () }

impl Example {
    pub fn a$0dd() -> Self {
        Self { _inner: () }
    }
}

"#,
        );
    }

    #[test]
    fn default_block_is_already_present() {
        cov_mark::check!(default_block_is_already_present);
        check_assist_not_applicable(
            generate_default_from_new,
            r#"
//- minicore: default
struct Example { _inner: () }

impl Example {
    pub fn n$0ew() -> Self {
        Self { _inner: () }
    }
}

impl Default for Example {
    fn default() -> Self {
        Self::new()
    }
}
"#,
        );
    }

    #[test]
    fn standalone_new_function() {
        check_assist_not_applicable(
            generate_default_from_new,
            r#"
fn n$0ew() -> u32 {
    0
}
"#,
        );
    }

    #[test]
    fn multiple_struct_blocks() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
struct Example { _inner: () }
struct Test { value: u32 }

impl Example {
    pub fn new$0() -> Self {
        Self { _inner: () }
    }
}
"#,
            r#"
struct Example { _inner: () }
struct Test { value: u32 }

impl Example {
    pub fn new() -> Self {
        Self { _inner: () }
    }
}

impl Default for Example {
    fn default() -> Self {
        Self::new()
    }
}
"#,
        );
    }

    #[test]
    fn when_struct_is_after_impl() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
impl Example {
    pub fn $0new() -> Self {
        Self { _inner: () }
    }
}

struct Example { _inner: () }
"#,
            r#"
impl Example {
    pub fn new() -> Self {
        Self { _inner: () }
    }
}

impl Default for Example {
    fn default() -> Self {
        Self::new()
    }
}

struct Example { _inner: () }
"#,
        );
    }

    #[test]
    fn struct_in_module() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
mod test {
    struct Example { _inner: () }

    impl Example {
        pub fn n$0ew() -> Self {
            Self { _inner: () }
        }
    }
}
"#,
            r#"
mod test {
    struct Example { _inner: () }

    impl Example {
        pub fn new() -> Self {
            Self { _inner: () }
        }
    }

impl Default for Example {
    fn default() -> Self {
        Self::new()
    }
}
}
"#,
        );
    }

    #[test]
    fn struct_in_module_with_default() {
        cov_mark::check!(struct_in_module_with_default);
        check_assist_not_applicable(
            generate_default_from_new,
            r#"
//- minicore: default
mod test {
    struct Example { _inner: () }

    impl Example {
        pub fn n$0ew() -> Self {
            Self { _inner: () }
        }
    }

    impl Default for Example {
        fn default() -> Self {
            Self::new()
        }
    }
}
"#,
        );
    }

    #[test]
    fn not_applicable_when_default_lang_item_is_missing() {
        check_assist_not_applicable(
            generate_default_from_new,
            r#"
struct S;
impl S {
    fn new$0() -> Self {}
}
"#,
        );
    }

    #[test]
    fn not_applicable_for_missing_self_ty() {
        // Regression test for #15398.
        check_assist_not_applicable(generate_default_from_new, "impl { fn new$0() -> Self {} }");
    }
}
