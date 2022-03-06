use ide_db::famous_defs::FamousDefs;
use itertools::Itertools;
use stdx::format_to;
use syntax::{
    ast::{self, HasGenericParams, HasName, HasTypeBounds, Impl},
    AstNode,
};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId,
};

// Assist: generate_default_from_new
//
// Generates default implementation from new method.
//
// ```
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
pub(crate) fn generate_default_from_new(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
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

    let impl_ = fn_node.syntax().ancestors().into_iter().find_map(ast::Impl::cast)?;
    if is_default_implemented(ctx, &impl_) {
        cov_mark::hit!(default_block_is_already_present);
        cov_mark::hit!(struct_in_module_with_default);
        return None;
    }

    let insert_location = impl_.syntax().text_range();

    acc.add(
        AssistId("generate_default_from_new", crate::AssistKind::Generate),
        "Generate a Default impl from a new fn",
        insert_location,
        move |builder| {
            let default_code = "    fn default() -> Self {
        Self::new()
    }";
            let code = generate_trait_impl_text_from_impl(&impl_, "Default", default_code);
            builder.insert(insert_location.end(), code);
        },
    )
}

fn generate_trait_impl_text_from_impl(impl_: &ast::Impl, trait_text: &str, code: &str) -> String {
    let generic_params = impl_.generic_param_list();
    let mut buf = String::with_capacity(code.len());
    buf.push_str("\n\n");
    buf.push_str("impl");

    if let Some(generic_params) = &generic_params {
        let lifetimes = generic_params.lifetime_params().map(|lt| format!("{}", lt.syntax()));
        let toc_params = generic_params.type_or_const_params().map(|toc_param| match toc_param {
            ast::TypeOrConstParam::Type(type_param) => {
                let mut buf = String::new();
                if let Some(it) = type_param.name() {
                    format_to!(buf, "{}", it.syntax());
                }
                if let Some(it) = type_param.colon_token() {
                    format_to!(buf, "{} ", it);
                }
                if let Some(it) = type_param.type_bound_list() {
                    format_to!(buf, "{}", it.syntax());
                }
                buf
            }
            ast::TypeOrConstParam::Const(const_param) => const_param.syntax().to_string(),
        });
        let generics = lifetimes.chain(toc_params).format(", ");
        format_to!(buf, "<{}>", generics);
    }

    buf.push(' ');
    buf.push_str(trait_text);
    buf.push_str(" for ");
    buf.push_str(&impl_.self_ty().unwrap().syntax().text().to_string());

    match impl_.where_clause() {
        Some(where_clause) => {
            format_to!(buf, "\n{}\n{{\n{}\n}}", where_clause, code);
        }
        None => {
            format_to!(buf, " {{\n{}\n}}", code);
        }
    }

    buf
}

fn is_default_implemented(ctx: &AssistContext, impl_: &Impl) -> bool {
    let db = ctx.sema.db;
    let impl_ = ctx.sema.to_def(impl_);
    let impl_def = match impl_ {
        Some(value) => value,
        None => return false,
    };

    let ty = impl_def.self_ty(db);
    let krate = impl_def.module(db).krate();
    let default = FamousDefs(&ctx.sema, Some(krate)).core_default_Default();
    let default_trait = match default {
        Some(value) => value,
        None => return false,
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
    fn new_function_with_generics_and_wheres() {
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
}
