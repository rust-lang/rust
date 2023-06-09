use hir::{known, HasSource, Name};
use syntax::{
    ast::{self, HasName},
    AstNode,
};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: generate_is_empty_from_len
//
// Generates is_empty implementation from the len method.
//
// ```
// struct MyStruct { data: Vec<String> }
//
// impl MyStruct {
//     #[must_use]
//     p$0ub fn len(&self) -> usize {
//         self.data.len()
//     }
// }
// ```
// ->
// ```
// struct MyStruct { data: Vec<String> }
//
// impl MyStruct {
//     #[must_use]
//     pub fn len(&self) -> usize {
//         self.data.len()
//     }
//
//     #[must_use]
//     pub fn is_empty(&self) -> bool {
//         self.len() == 0
//     }
// }
// ```
pub(crate) fn generate_is_empty_from_len(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let fn_node = ctx.find_node_at_offset::<ast::Fn>()?;
    let fn_name = fn_node.name()?;

    if fn_name.text() != "len" {
        cov_mark::hit!(len_function_not_present);
        return None;
    }

    if fn_node.param_list()?.params().next().is_some() {
        cov_mark::hit!(len_function_with_parameters);
        return None;
    }

    let impl_ = fn_node.syntax().ancestors().find_map(ast::Impl::cast)?;
    let len_fn = get_impl_method(ctx, &impl_, &known::len)?;
    if !len_fn.ret_type(ctx.sema.db).is_usize() {
        cov_mark::hit!(len_fn_different_return_type);
        return None;
    }

    if get_impl_method(ctx, &impl_, &known::is_empty).is_some() {
        cov_mark::hit!(is_empty_already_implemented);
        return None;
    }

    let node = len_fn.source(ctx.sema.db)?;
    let range = node.syntax().value.text_range();

    acc.add(
        AssistId("generate_is_empty_from_len", AssistKind::Generate),
        "Generate a is_empty impl from a len function",
        range,
        |builder| {
            let code = r#"

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }"#
            .to_string();
            builder.insert(range.end(), code)
        },
    )
}

fn get_impl_method(
    ctx: &AssistContext<'_>,
    impl_: &ast::Impl,
    fn_name: &Name,
) -> Option<hir::Function> {
    let db = ctx.sema.db;
    let impl_def: hir::Impl = ctx.sema.to_def(impl_)?;

    let scope = ctx.sema.scope(impl_.syntax())?;
    let ty = impl_def.self_ty(db);
    ty.iterate_method_candidates(db, &scope, None, Some(fn_name), |func| Some(func))
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn len_function_not_present() {
        cov_mark::check!(len_function_not_present);
        check_assist_not_applicable(
            generate_is_empty_from_len,
            r#"
struct MyStruct { data: Vec<String> }

impl MyStruct {
    p$0ub fn test(&self) -> usize {
            self.data.len()
        }
    }
"#,
        );
    }

    #[test]
    fn len_function_with_parameters() {
        cov_mark::check!(len_function_with_parameters);
        check_assist_not_applicable(
            generate_is_empty_from_len,
            r#"
struct MyStruct { data: Vec<String> }

impl MyStruct {
    #[must_use]
    p$0ub fn len(&self, _i: bool) -> usize {
        self.data.len()
    }
}
"#,
        );
    }

    #[test]
    fn is_empty_already_implemented() {
        cov_mark::check!(is_empty_already_implemented);
        check_assist_not_applicable(
            generate_is_empty_from_len,
            r#"
struct MyStruct { data: Vec<String> }

impl MyStruct {
    #[must_use]
    p$0ub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
"#,
        );
    }

    #[test]
    fn len_fn_different_return_type() {
        cov_mark::check!(len_fn_different_return_type);
        check_assist_not_applicable(
            generate_is_empty_from_len,
            r#"
struct MyStruct { data: Vec<String> }

impl MyStruct {
    #[must_use]
    p$0ub fn len(&self) -> u32 {
        self.data.len()
    }
}
"#,
        );
    }

    #[test]
    fn generate_is_empty() {
        check_assist(
            generate_is_empty_from_len,
            r#"
struct MyStruct { data: Vec<String> }

impl MyStruct {
    #[must_use]
    p$0ub fn len(&self) -> usize {
        self.data.len()
    }
}
"#,
            r#"
struct MyStruct { data: Vec<String> }

impl MyStruct {
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
"#,
        );
    }

    #[test]
    fn multiple_functions_in_impl() {
        check_assist(
            generate_is_empty_from_len,
            r#"
struct MyStruct { data: Vec<String> }

impl MyStruct {
    #[must_use]
    pub fn new() -> Self {
        Self { data: 0 }
    }

    #[must_use]
    p$0ub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn work(&self) -> Option<usize> {

    }
}
"#,
            r#"
struct MyStruct { data: Vec<String> }

impl MyStruct {
    #[must_use]
    pub fn new() -> Self {
        Self { data: 0 }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn work(&self) -> Option<usize> {

    }
}
"#,
        );
    }

    #[test]
    fn multiple_impls() {
        check_assist_not_applicable(
            generate_is_empty_from_len,
            r#"
struct MyStruct { data: Vec<String> }

impl MyStruct {
    #[must_use]
    p$0ub fn len(&self) -> usize {
        self.data.len()
    }
}

impl MyStruct {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
"#,
        );
    }
}
