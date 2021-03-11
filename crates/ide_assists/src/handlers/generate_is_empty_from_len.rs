use hir::{AssocItem, HasSource, Impl};
use syntax::{
    ast::{self, NameOwner},
    AstNode, TextRange,
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
// impl MyStruct {
//     p$0ub fn len(&self) -> usize {
//         self.data.len()
//     }
// }
// ```
// ->
// ```
// impl MyStruct {
//     pub fn len(&self) -> usize {
//         self.data.len()
//     }
//
//     pub fn is_empty(&self) -> bool {
//         self.len() == 0
//     }
// }
// ```
pub(crate) fn generate_is_empty_from_len(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
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

    let impl_ = fn_node.syntax().ancestors().into_iter().find_map(ast::Impl::cast)?;
    let impl_def = ctx.sema.to_def(&impl_)?;
    if is_empty_implemented(ctx, &impl_def) {
        cov_mark::hit!(is_empty_already_implemented);
        return None;
    }

    let range = get_text_range_of_len_function(ctx, &impl_def)?;

    acc.add(
        AssistId("generate_is_empty_from_len", AssistKind::Generate),
        "Generate a is_empty impl from a len function",
        range,
        |builder| {
            let code = get_is_empty_code();
            builder.insert(range.end(), code)
        },
    )
}

fn get_function_from_impl(ctx: &AssistContext, impl_def: &Impl, name: &str) -> Option<AssocItem> {
    let db = ctx.sema.db;
    impl_def.items(db).into_iter().filter(|item| matches!(item, AssocItem::Function(_value))).find(
        |func| match func.name(db) {
            Some(fn_name) => fn_name.to_string() == name,
            None => false,
        },
    )
}

fn is_empty_implemented(ctx: &AssistContext, impl_def: &Impl) -> bool {
    get_function_from_impl(ctx, impl_def, "is_empty").is_some()
}

fn get_text_range_of_len_function(ctx: &AssistContext, impl_def: &Impl) -> Option<TextRange> {
    let db = ctx.sema.db;
    let len_fn = get_function_from_impl(ctx, impl_def, "len")?;

    let mut range = None;
    if let AssocItem::Function(node) = len_fn {
        let node = node.source(db)?;
        range = Some(node.syntax().value.text_range());
    }

    range
}

fn get_is_empty_code() -> String {
    r#"

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }"#
    .to_string()
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
impl MyStruct {
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
impl MyStruct {
    p$0ub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
impl MyStruct {
    p$0ub fn len(&self) -> usize {
        self.data.len()
    }
}
"#,
            r#"
impl MyStruct {
    pub fn len(&self) -> usize {
        self.data.len()
    }

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
impl MyStruct {
    pub fn new() -> Self {
        Self { data: 0 }
    }

    p$0ub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn work(&self) -> Option<usize> {
        // do some work
    }
}
"#,
            r#"
impl MyStruct {
    pub fn new() -> Self {
        Self { data: 0 }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn work(&self) -> Option<usize> {
        // do some work
    }
}
"#,
        );
    }
}
