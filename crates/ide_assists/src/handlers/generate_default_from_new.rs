use crate::{
    assist_context::{AssistContext, Assists},
    AssistId,
};
use syntax::{
    ast::{self, NameOwner},
    AstNode, SyntaxKind, SyntaxNode, SyntaxText,
};
use test_utils::mark;

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
    let fn_node: ast::Fn = ctx.find_node_at_offset()?;
    let fn_name = fn_node.name()?.to_string();

    if !fn_name.eq("new") {
        mark::hit!(other_function_than_new);
        return None;
    }

    if fn_node.param_list()?.params().count() != 0 {
        mark::hit!(new_function_with_parameters);
        return None;
    }

    let insert_after = scope_for_fn_insertion_node(&fn_node.syntax())?;
    let impl_obj = ast::Impl::cast(insert_after)?;
    let struct_name = impl_obj.self_ty()?.syntax().text();

    let default_fn_syntax = default_fn_node_for_new(struct_name);

    acc.add(
        AssistId("generate_default_from_new", crate::AssistKind::Generate),
        "Generate a Default impl from a new fn",
        impl_obj.syntax().text_range(),
        move |builder| {
            // FIXME: indentation logic can also go here.
            // let new_indent = IndentLevel::from_node(&insert_after);
            let insert_location = impl_obj.syntax().text_range().end();
            builder.insert(insert_location, default_fn_syntax);
        },
    )
}

fn scope_for_fn_insertion_node(node: &SyntaxNode) -> Option<SyntaxNode> {
    node.ancestors().into_iter().find(|node| node.kind() == SyntaxKind::IMPL)
}

fn default_fn_node_for_new(struct_name: SyntaxText) -> String {
    // FIXME: Update the implementation to consider the code indentation.
    format!(
        r#"

impl Default for {} {{
    fn default() -> Self {{
        Self::new()
    }}
}}"#,
        struct_name
    )
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
    fn new_function_with_parameters() {
        mark::check!(new_function_with_parameters);
        check_assist_not_applicable(
            generate_default_from_new,
            r#"
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
        mark::check!(other_function_than_new);
        check_assist_not_applicable(
            generate_default_from_new,
            r#"
struct Example { _inner: () }

impl Exmaple {
    pub fn a$0dd() -> Self {
        Self { _inner: () }
    }
}

"#,
        );
    }

    //     #[test]
    //     fn default_block_is_already_present() {
    //         check_assist_not_applicable(generate_default_from_new,
    //         r#"
    // struct Example { _inner: () }

    // impl Exmaple {
    //     pub fn n$0ew() -> Self {
    //         Self { _inner: () }
    //     }
    // }

    // impl Default for Example {
    //     fn default() -> Self {
    //         Self::new()
    //     }
    // }
    // "#,
    //         );
    //     }

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
}
