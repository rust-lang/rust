use crate::{
    assist_context::{AssistContext, Assists},
    utils, AssistId,
};
use syntax::{
    ast::{self, Adt, Impl, NameOwner},
    AstNode, Direction,
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
    let fn_node = ctx.find_node_at_offset::<ast::Fn>()?;
    let fn_name = fn_node.name()?;

    if fn_name.text() != "new" {
        mark::hit!(other_function_than_new);
        return None;
    }

    if fn_node.param_list()?.params().next().is_some() {
        mark::hit!(new_function_with_parameters);
        return None;
    }

    let impl_ = fn_node.syntax().ancestors().into_iter().find_map(ast::Impl::cast)?;

    let insert_location = impl_.syntax().text_range();

    acc.add(
        AssistId("generate_default_from_new", crate::AssistKind::Generate),
        "Generate a Default impl from a new fn",
        insert_location,
        move |builder| {
            let default_fn_syntax = default_fn_node_for_new(impl_);
            if let Some(code) = default_fn_syntax {
                builder.insert(insert_location.end(), code)
            }
        },
    )
}

fn default_fn_node_for_new(impl_: Impl) -> Option<String> {
    // the code string is this way due to formatting reason
    let code = r#"    fn default() -> Self {
        Self::new()
    }"#;
    let struct_name = impl_.self_ty()?.syntax().to_string();
    let struct_ = impl_
        .syntax()
        .siblings(Direction::Prev)
        .filter_map(ast::Struct::cast)
        .find(|struct_| struct_.name().unwrap().text() == struct_name)?;

    let adt = Adt::cast(struct_.syntax().clone())?;
    Some(utils::generate_trait_impl_text(&adt, "Default", code))
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

    #[test]
    fn multiple_struct_blocks() {
        check_assist(
            generate_default_from_new,
            r#"
struct Example { _inner: () }
struct Test { value: u32 }

impl Example {
    pub fn new$0 () -> Self {
        Self { _inner: () }
    }
}
"#,
            r#"
struct Example { _inner: () }
struct Test { value: u32 }

impl Example {
    pub fn new () -> Self {
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
}
