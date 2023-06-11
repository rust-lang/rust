use hir::{AsAssocItem, HasVisibility, ModuleDef, Visibility};
use ide_db::assists::{AssistId, AssistKind};
use itertools::Itertools;
use stdx::{format_to, to_lower_snake_case};
use syntax::{
    algo::skip_whitespace_token,
    ast::{self, edit::IndentLevel, HasDocComments, HasName},
    match_ast, AstNode, AstToken,
};

use crate::assist_context::{AssistContext, Assists};

// Assist: generate_documentation_template
//
// Adds a documentation template above a function definition / declaration.
//
// ```
// pub struct S;
// impl S {
//     pub unsafe fn set_len$0(&mut self, len: usize) -> Result<(), std::io::Error> {
//         /* ... */
//     }
// }
// ```
// ->
// ```
// pub struct S;
// impl S {
//     /// Sets the length of this [`S`].
//     ///
//     /// # Errors
//     ///
//     /// This function will return an error if .
//     ///
//     /// # Safety
//     ///
//     /// .
//     pub unsafe fn set_len(&mut self, len: usize) -> Result<(), std::io::Error> {
//         /* ... */
//     }
// }
// ```
pub(crate) fn generate_documentation_template(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let name = ctx.find_node_at_offset::<ast::Name>()?;
    let ast_func = name.syntax().parent().and_then(ast::Fn::cast)?;
    if is_in_trait_impl(&ast_func, ctx) || ast_func.doc_comments().next().is_some() {
        return None;
    }

    let parent_syntax = ast_func.syntax();
    let text_range = parent_syntax.text_range();
    let indent_level = IndentLevel::from_node(parent_syntax);

    acc.add(
        AssistId("generate_documentation_template", AssistKind::Generate),
        "Generate a documentation template",
        text_range,
        |builder| {
            // Introduction / short function description before the sections
            let mut doc_lines = vec![introduction_builder(&ast_func, ctx).unwrap_or(".".into())];
            // Then come the sections
            for section_builder in [panics_builder, errors_builder, safety_builder] {
                if let Some(mut lines) = section_builder(&ast_func) {
                    doc_lines.push("".into());
                    doc_lines.append(&mut lines);
                }
            }
            builder.insert(text_range.start(), documentation_from_lines(doc_lines, indent_level));
        },
    )
}

// Assist: generate_doc_example
//
// Generates a rustdoc example when editing an item's documentation.
//
// ```
// /// Adds two numbers.$0
// pub fn add(a: i32, b: i32) -> i32 { a + b }
// ```
// ->
// ```
// /// Adds two numbers.
// ///
// /// # Examples
// ///
// /// ```
// /// use test::add;
// ///
// /// assert_eq!(add(a, b), );
// /// ```
// pub fn add(a: i32, b: i32) -> i32 { a + b }
// ```
pub(crate) fn generate_doc_example(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let tok: ast::Comment = ctx.find_token_at_offset()?;
    let node = tok.syntax().parent()?;
    let last_doc_token =
        ast::AnyHasDocComments::cast(node.clone())?.doc_comments().last()?.syntax().clone();
    let next_token = skip_whitespace_token(last_doc_token.next_token()?, syntax::Direction::Next)?;

    let example = match_ast! {
        match node {
            ast::Fn(it) => make_example_for_fn(&it, ctx)?,
            _ => return None,
        }
    };

    let mut lines = string_vec_from(&["", "# Examples", "", "```"]);
    lines.extend(example.lines().map(String::from));
    lines.push("```".into());
    let indent_level = IndentLevel::from_node(&node);

    acc.add(
        AssistId("generate_doc_example", AssistKind::Generate),
        "Generate a documentation example",
        node.text_range(),
        |builder| {
            builder.insert(
                next_token.text_range().start(),
                documentation_from_lines(lines, indent_level),
            );
        },
    )
}

fn make_example_for_fn(ast_func: &ast::Fn, ctx: &AssistContext<'_>) -> Option<String> {
    if !is_public(ast_func, ctx)? {
        // Doctests for private items can't actually name the item, so they're pretty useless.
        return None;
    }

    if is_in_trait_def(ast_func, ctx) {
        // This is not yet implemented.
        return None;
    }

    let mut example = String::new();

    let use_path = build_path(ast_func, ctx)?;
    let is_unsafe = ast_func.unsafe_token().is_some();
    let param_list = ast_func.param_list()?;
    let ref_mut_params = ref_mut_params(&param_list);
    let self_name = self_name(ast_func);

    format_to!(example, "use {use_path};\n\n");
    if let Some(self_name) = &self_name {
        if let Some(mut_) = is_ref_mut_self(ast_func) {
            let mut_ = if mut_ == true { "mut " } else { "" };
            format_to!(example, "let {mut_}{self_name} = ;\n");
        }
    }
    for param_name in &ref_mut_params {
        format_to!(example, "let mut {param_name} = ;\n");
    }
    // Call the function, check result
    let function_call = function_call(ast_func, &param_list, self_name.as_deref(), is_unsafe)?;
    if returns_a_value(ast_func, ctx) {
        if count_parameters(&param_list) < 3 {
            format_to!(example, "assert_eq!({function_call}, );\n");
        } else {
            format_to!(example, "let result = {function_call};\n");
            example.push_str("assert_eq!(result, );\n");
        }
    } else {
        format_to!(example, "{function_call};\n");
    }
    // Check the mutated values
    if let Some(self_name) = &self_name {
        if is_ref_mut_self(ast_func) == Some(true) {
            format_to!(example, "assert_eq!({self_name}, );");
        }
    }
    for param_name in &ref_mut_params {
        format_to!(example, "assert_eq!({param_name}, );");
    }

    Some(example)
}

fn introduction_builder(ast_func: &ast::Fn, ctx: &AssistContext<'_>) -> Option<String> {
    let hir_func = ctx.sema.to_def(ast_func)?;
    let container = hir_func.as_assoc_item(ctx.db())?.container(ctx.db());
    if let hir::AssocItemContainer::Impl(imp) = container {
        let ret_ty = hir_func.ret_type(ctx.db());
        let self_ty = imp.self_ty(ctx.db());
        let name = ast_func.name()?.to_string();
        let linkable_self_ty = self_type_without_lifetimes(ast_func);
        let linkable_self_ty = linkable_self_ty.as_deref();

        let intro_for_new = || {
            let is_new = name == "new";
            if is_new && ret_ty == self_ty {
                let self_ty = linkable_self_ty?;
                Some(format!("Creates a new [`{self_ty}`]."))
            } else {
                None
            }
        };

        let intro_for_getter = || match (
            hir_func.self_param(ctx.sema.db),
            &*hir_func.params_without_self(ctx.sema.db),
        ) {
            (Some(self_param), []) if self_param.access(ctx.sema.db) != hir::Access::Owned => {
                if name.starts_with("as_") || name.starts_with("to_") || name == "get" {
                    return None;
                }
                let mut what = name.trim_end_matches("_mut").replace('_', " ");
                if what == "len" {
                    what = "length".into()
                }
                let reference = if ret_ty.is_mutable_reference() {
                    " a mutable reference to"
                } else if ret_ty.is_reference() {
                    " a reference to"
                } else {
                    ""
                };

                let self_ty = linkable_self_ty?;
                Some(format!("Returns{reference} the {what} of this [`{self_ty}`]."))
            }
            _ => None,
        };

        let intro_for_setter = || {
            if !name.starts_with("set_") {
                return None;
            }

            let mut what = name.trim_start_matches("set_").replace('_', " ");
            if what == "len" {
                what = "length".into()
            };

            let self_ty = linkable_self_ty?;
            Some(format!("Sets the {what} of this [`{self_ty}`]."))
        };

        if let Some(intro) = intro_for_new() {
            return Some(intro);
        }
        if let Some(intro) = intro_for_getter() {
            return Some(intro);
        }
        if let Some(intro) = intro_for_setter() {
            return Some(intro);
        }
    }
    None
}

/// Builds an optional `# Panics` section
fn panics_builder(ast_func: &ast::Fn) -> Option<Vec<String>> {
    match can_panic(ast_func) {
        Some(true) => Some(string_vec_from(&["# Panics", "", "Panics if ."])),
        _ => None,
    }
}

/// Builds an optional `# Errors` section
fn errors_builder(ast_func: &ast::Fn) -> Option<Vec<String>> {
    match return_type(ast_func)?.to_string().contains("Result") {
        true => Some(string_vec_from(&["# Errors", "", "This function will return an error if ."])),
        false => None,
    }
}

/// Builds an optional `# Safety` section
fn safety_builder(ast_func: &ast::Fn) -> Option<Vec<String>> {
    let is_unsafe = ast_func.unsafe_token().is_some();
    match is_unsafe {
        true => Some(string_vec_from(&["# Safety", "", "."])),
        false => None,
    }
}

/// Checks if the function is public / exported
fn is_public(ast_func: &ast::Fn, ctx: &AssistContext<'_>) -> Option<bool> {
    let hir_func = ctx.sema.to_def(ast_func)?;
    Some(
        hir_func.visibility(ctx.db()) == Visibility::Public
            && all_parent_mods_public(&hir_func, ctx),
    )
}

/// Checks that all parent modules of the function are public / exported
fn all_parent_mods_public(hir_func: &hir::Function, ctx: &AssistContext<'_>) -> bool {
    let mut module = hir_func.module(ctx.db());
    loop {
        if let Some(parent) = module.parent(ctx.db()) {
            match ModuleDef::from(module).visibility(ctx.db()) {
                Visibility::Public => module = parent,
                _ => break false,
            }
        } else {
            break true;
        }
    }
}

/// Returns the name of the current crate
fn crate_name(ast_func: &ast::Fn, ctx: &AssistContext<'_>) -> Option<String> {
    let krate = ctx.sema.scope(ast_func.syntax())?.krate();
    Some(krate.display_name(ctx.db())?.to_string())
}

/// `None` if function without a body; some bool to guess if function can panic
fn can_panic(ast_func: &ast::Fn) -> Option<bool> {
    let body = ast_func.body()?.to_string();
    let can_panic = body.contains("panic!(")
        // FIXME it would be better to not match `debug_assert*!` macro invocations
        || body.contains("assert!(")
        || body.contains(".unwrap()")
        || body.contains(".expect(");
    Some(can_panic)
}

/// Helper function to get the name that should be given to `self` arguments
fn self_name(ast_func: &ast::Fn) -> Option<String> {
    self_partial_type(ast_func).map(|name| to_lower_snake_case(&name))
}

/// Helper function to get the name of the type of `self`
fn self_type(ast_func: &ast::Fn) -> Option<ast::Type> {
    ast_func.syntax().ancestors().find_map(ast::Impl::cast).and_then(|i| i.self_ty())
}

/// Output the real name of `Self` like `MyType<T>`, without the lifetimes.
fn self_type_without_lifetimes(ast_func: &ast::Fn) -> Option<String> {
    let path_segment = match self_type(ast_func)? {
        ast::Type::PathType(path_type) => path_type.path()?.segment()?,
        _ => return None,
    };
    let mut name = path_segment.name_ref()?.to_string();
    let generics = path_segment.generic_arg_list().into_iter().flat_map(|list| {
        list.generic_args()
            .filter(|generic| matches!(generic, ast::GenericArg::TypeArg(_)))
            .map(|generic| generic.to_string())
    });
    let generics: String = generics.format(", ").to_string();
    if !generics.is_empty() {
        name.push('<');
        name.push_str(&generics);
        name.push('>');
    }
    Some(name)
}

/// Helper function to get the name of the type of `self` without generic arguments
fn self_partial_type(ast_func: &ast::Fn) -> Option<String> {
    let mut self_type = self_type(ast_func)?.to_string();
    if let Some(idx) = self_type.find(|c| ['<', ' '].contains(&c)) {
        self_type.truncate(idx);
    }
    Some(self_type)
}

/// Helper function to determine if the function is in a trait implementation
fn is_in_trait_impl(ast_func: &ast::Fn, ctx: &AssistContext<'_>) -> bool {
    ctx.sema
        .to_def(ast_func)
        .and_then(|hir_func| hir_func.as_assoc_item(ctx.db()))
        .and_then(|assoc_item| assoc_item.containing_trait_impl(ctx.db()))
        .is_some()
}

/// Helper function to determine if the function definition is in a trait definition
fn is_in_trait_def(ast_func: &ast::Fn, ctx: &AssistContext<'_>) -> bool {
    ctx.sema
        .to_def(ast_func)
        .and_then(|hir_func| hir_func.as_assoc_item(ctx.db()))
        .and_then(|assoc_item| assoc_item.containing_trait(ctx.db()))
        .is_some()
}

/// Returns `None` if no `self` at all, `Some(true)` if there is `&mut self` else `Some(false)`
fn is_ref_mut_self(ast_func: &ast::Fn) -> Option<bool> {
    let self_param = ast_func.param_list()?.self_param()?;
    Some(self_param.mut_token().is_some() && self_param.amp_token().is_some())
}

/// Helper function to determine if a parameter is `&mut`
fn is_a_ref_mut_param(param: &ast::Param) -> bool {
    match param.ty() {
        Some(ast::Type::RefType(param_ref)) => param_ref.mut_token().is_some(),
        _ => false,
    }
}

/// Helper function to build the list of `&mut` parameters
fn ref_mut_params(param_list: &ast::ParamList) -> Vec<String> {
    param_list
        .params()
        .filter_map(|param| match is_a_ref_mut_param(&param) {
            // Maybe better filter the param name (to do this maybe extract a function from
            // `arguments_from_params`?) in case of a `mut a: &mut T`. Anyway managing most (not
            // all) cases might be enough, the goal is just to produce a template.
            true => Some(param.pat()?.to_string()),
            false => None,
        })
        .collect()
}

/// Helper function to build the comma-separated list of arguments of the function
fn arguments_from_params(param_list: &ast::ParamList) -> String {
    let args_iter = param_list.params().map(|param| match param.pat() {
        // To avoid `mut` in the function call (which would be a nonsense), `Pat` should not be
        // written as is so its variants must be managed independently. Other variants (for
        // instance `TuplePat`) could be managed later.
        Some(ast::Pat::IdentPat(ident_pat)) => match ident_pat.name() {
            Some(name) => match is_a_ref_mut_param(&param) {
                true => format!("&mut {name}"),
                false => name.to_string(),
            },
            None => "_".to_string(),
        },
        _ => "_".to_string(),
    });
    args_iter.format(", ").to_string()
}

/// Helper function to build a function call. `None` if expected `self_name` was not provided
fn function_call(
    ast_func: &ast::Fn,
    param_list: &ast::ParamList,
    self_name: Option<&str>,
    is_unsafe: bool,
) -> Option<String> {
    let name = ast_func.name()?;
    let arguments = arguments_from_params(param_list);
    let function_call = if param_list.self_param().is_some() {
        let self_ = self_name?;
        format!("{self_}.{name}({arguments})")
    } else if let Some(implementation) = self_partial_type(ast_func) {
        format!("{implementation}::{name}({arguments})")
    } else {
        format!("{name}({arguments})")
    };
    match is_unsafe {
        true => Some(format!("unsafe {{ {function_call} }}")),
        false => Some(function_call),
    }
}

/// Helper function to count the parameters including `self`
fn count_parameters(param_list: &ast::ParamList) -> usize {
    param_list.params().count() + if param_list.self_param().is_some() { 1 } else { 0 }
}

/// Helper function to transform lines of documentation into a Rust code documentation
fn documentation_from_lines(doc_lines: Vec<String>, indent_level: IndentLevel) -> String {
    let mut result = String::new();
    for doc_line in doc_lines {
        result.push_str("///");
        if !doc_line.is_empty() {
            result.push(' ');
            result.push_str(&doc_line);
        }
        result.push('\n');
        result.push_str(&indent_level.to_string());
    }
    result
}

/// Helper function to transform an array of borrowed strings to an owned `Vec<String>`
fn string_vec_from(string_array: &[&str]) -> Vec<String> {
    string_array.iter().map(|&s| s.to_owned()).collect()
}

/// Helper function to build the path of the module in the which is the node
fn build_path(ast_func: &ast::Fn, ctx: &AssistContext<'_>) -> Option<String> {
    let crate_name = crate_name(ast_func, ctx)?;
    let leaf = self_partial_type(ast_func)
        .or_else(|| ast_func.name().map(|n| n.to_string()))
        .unwrap_or_else(|| "*".into());
    let module_def: ModuleDef = ctx.sema.to_def(ast_func)?.module(ctx.db()).into();
    match module_def.canonical_path(ctx.db()) {
        Some(path) => Some(format!("{crate_name}::{path}::{leaf}")),
        None => Some(format!("{crate_name}::{leaf}")),
    }
}

/// Helper function to get the return type of a function
fn return_type(ast_func: &ast::Fn) -> Option<ast::Type> {
    ast_func.ret_type()?.ty()
}

/// Helper function to determine if the function returns some data
fn returns_a_value(ast_func: &ast::Fn, ctx: &AssistContext<'_>) -> bool {
    ctx.sema
        .to_def(ast_func)
        .map(|hir_func| hir_func.ret_type(ctx.db()))
        .map(|ret_ty| !ret_ty.is_unit() && !ret_ty.is_never())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn not_applicable_on_function_calls() {
        check_assist_not_applicable(
            generate_documentation_template,
            r#"
fn hello_world() {}
fn calls_hello_world() {
    hello_world$0();
}
"#,
        )
    }

    #[test]
    fn not_applicable_in_trait_impl() {
        check_assist_not_applicable(
            generate_documentation_template,
            r#"
trait MyTrait {}
struct MyStruct;
impl MyTrait for MyStruct {
    fn hello_world$0();
}
"#,
        )
    }

    #[test]
    fn not_applicable_if_function_already_documented() {
        check_assist_not_applicable(
            generate_documentation_template,
            r#"
/// Some documentation here
pub fn $0documented_function() {}
"#,
        );
    }

    #[test]
    fn supports_noop_function() {
        check_assist(
            generate_documentation_template,
            r#"
pub fn no$0op() {}
"#,
            r#"
/// .
pub fn noop() {}
"#,
        );
    }

    #[test]
    fn is_applicable_if_function_is_private() {
        check_assist(
            generate_documentation_template,
            r#"
fn priv$0ate() {}
"#,
            r#"
/// .
fn private() {}
"#,
        );
    }

    #[test]
    fn no_doc_example_for_private_fn() {
        check_assist_not_applicable(
            generate_doc_example,
            r#"
///$0
fn private() {}
"#,
        );
    }

    #[test]
    fn supports_a_parameter() {
        check_assist(
            generate_doc_example,
            r#"
/// $0.
pub fn noop_with_param(_a: i32) {}
"#,
            r#"
/// .
///
/// # Examples
///
/// ```
/// use test::noop_with_param;
///
/// noop_with_param(_a);
/// ```
pub fn noop_with_param(_a: i32) {}
"#,
        );
    }

    #[test]
    fn detects_unsafe_function() {
        check_assist(
            generate_documentation_template,
            r#"
pub unsafe fn no$0op_unsafe() {}
"#,
            r#"
/// .
///
/// # Safety
///
/// .
pub unsafe fn noop_unsafe() {}
"#,
        );
        check_assist(
            generate_doc_example,
            r#"
/// .
///
/// # Safety$0
///
/// .
pub unsafe fn noop_unsafe() {}
"#,
            r#"
/// .
///
/// # Safety
///
/// .
///
/// # Examples
///
/// ```
/// use test::noop_unsafe;
///
/// unsafe { noop_unsafe() };
/// ```
pub unsafe fn noop_unsafe() {}
"#,
        );
    }

    #[test]
    fn guesses_panic_macro_can_panic() {
        check_assist(
            generate_documentation_template,
            r#"
pub fn panic$0s_if(a: bool) {
    if a {
        panic!();
    }
}
"#,
            r#"
/// .
///
/// # Panics
///
/// Panics if .
pub fn panics_if(a: bool) {
    if a {
        panic!();
    }
}
"#,
        );
    }

    #[test]
    fn guesses_assert_macro_can_panic() {
        check_assist(
            generate_documentation_template,
            r#"
pub fn $0panics_if_not(a: bool) {
    assert!(a == true);
}
"#,
            r#"
/// .
///
/// # Panics
///
/// Panics if .
pub fn panics_if_not(a: bool) {
    assert!(a == true);
}
"#,
        );
    }

    #[test]
    fn guesses_unwrap_can_panic() {
        check_assist(
            generate_documentation_template,
            r#"
pub fn $0panics_if_none(a: Option<()>) {
    a.unwrap();
}
"#,
            r#"
/// .
///
/// # Panics
///
/// Panics if .
pub fn panics_if_none(a: Option<()>) {
    a.unwrap();
}
"#,
        );
    }

    #[test]
    fn guesses_expect_can_panic() {
        check_assist(
            generate_documentation_template,
            r#"
pub fn $0panics_if_none2(a: Option<()>) {
    a.expect("Bouh!");
}
"#,
            r#"
/// .
///
/// # Panics
///
/// Panics if .
pub fn panics_if_none2(a: Option<()>) {
    a.expect("Bouh!");
}
"#,
        );
    }

    #[test]
    fn checks_output_in_example() {
        check_assist(
            generate_doc_example,
            r#"
///$0
pub fn returns_a_value$0() -> i32 {
    0
}
"#,
            r#"
///
///
/// # Examples
///
/// ```
/// use test::returns_a_value;
///
/// assert_eq!(returns_a_value(), );
/// ```
pub fn returns_a_value() -> i32 {
    0
}
"#,
        );
    }

    #[test]
    fn detects_result_output() {
        check_assist(
            generate_documentation_template,
            r#"
pub fn returns_a_result$0() -> Result<i32, std::io::Error> {
    Ok(0)
}
"#,
            r#"
/// .
///
/// # Errors
///
/// This function will return an error if .
pub fn returns_a_result() -> Result<i32, std::io::Error> {
    Ok(0)
}
"#,
        );
    }

    #[test]
    fn checks_ref_mut_in_example() {
        check_assist(
            generate_doc_example,
            r#"
///$0
pub fn modifies_a_value$0(a: &mut i32) {
    *a = 0;
}
"#,
            r#"
///
///
/// # Examples
///
/// ```
/// use test::modifies_a_value;
///
/// let mut a = ;
/// modifies_a_value(&mut a);
/// assert_eq!(a, );
/// ```
pub fn modifies_a_value(a: &mut i32) {
    *a = 0;
}
"#,
        );
    }

    #[test]
    fn stores_result_if_at_least_3_params() {
        check_assist(
            generate_doc_example,
            r#"
///$0
pub fn sum3$0(a: i32, b: i32, c: i32) -> i32 {
    a + b + c
}
"#,
            r#"
///
///
/// # Examples
///
/// ```
/// use test::sum3;
///
/// let result = sum3(a, b, c);
/// assert_eq!(result, );
/// ```
pub fn sum3(a: i32, b: i32, c: i32) -> i32 {
    a + b + c
}
"#,
        );
    }

    #[test]
    fn supports_fn_in_mods() {
        check_assist(
            generate_doc_example,
            r#"
pub mod a {
    pub mod b {
        ///$0
        pub fn noop() {}
    }
}
"#,
            r#"
pub mod a {
    pub mod b {
        ///
        ///
        /// # Examples
        ///
        /// ```
        /// use test::a::b::noop;
        ///
        /// noop();
        /// ```
        pub fn noop() {}
    }
}
"#,
        );
    }

    #[test]
    fn supports_fn_in_impl() {
        check_assist(
            generate_doc_example,
            r#"
pub struct MyStruct;
impl MyStruct {
    ///$0
    pub fn noop() {}
}
"#,
            r#"
pub struct MyStruct;
impl MyStruct {
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use test::MyStruct;
    ///
    /// MyStruct::noop();
    /// ```
    pub fn noop() {}
}
"#,
        );
    }

    #[test]
    fn supports_unsafe_fn_in_trait() {
        check_assist(
            generate_documentation_template,
            r#"
pub trait MyTrait {
    unsafe fn unsafe_funct$0ion_trait();
}
"#,
            r#"
pub trait MyTrait {
    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn unsafe_function_trait();
}
"#,
        );
    }

    #[test]
    fn supports_fn_in_trait_with_default_panicking() {
        check_assist(
            generate_documentation_template,
            r#"
pub trait MyTrait {
    fn function_trait_with_$0default_panicking() {
        panic!()
    }
}
"#,
            r#"
pub trait MyTrait {
    /// .
    ///
    /// # Panics
    ///
    /// Panics if .
    fn function_trait_with_default_panicking() {
        panic!()
    }
}
"#,
        );
    }

    #[test]
    fn supports_fn_in_trait_returning_result() {
        check_assist(
            generate_documentation_template,
            r#"
pub trait MyTrait {
    fn function_tr$0ait_returning_result() -> Result<(), std::io::Error>;
}
"#,
            r#"
pub trait MyTrait {
    /// .
    ///
    /// # Errors
    ///
    /// This function will return an error if .
    fn function_trait_returning_result() -> Result<(), std::io::Error>;
}
"#,
        );
    }

    #[test]
    fn detects_new() {
        check_assist(
            generate_documentation_template,
            r#"
pub struct String(u8);
impl String {
    pub fn new$0(x: u8) -> String {
        String(x)
    }
}
"#,
            r#"
pub struct String(u8);
impl String {
    /// Creates a new [`String`].
    pub fn new(x: u8) -> String {
        String(x)
    }
}
"#,
        );
        check_assist(
            generate_documentation_template,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct<T> {
    pub x: T,
}
impl<T> MyGenericStruct<T> {
    pub fn new$0(x: T) -> MyGenericStruct<T> {
        MyGenericStruct { x }
    }
}
"#,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct<T> {
    pub x: T,
}
impl<T> MyGenericStruct<T> {
    /// Creates a new [`MyGenericStruct<T>`].
    pub fn new(x: T) -> MyGenericStruct<T> {
        MyGenericStruct { x }
    }
}
"#,
        );
    }

    #[test]
    fn removes_one_lifetime_from_description() {
        check_assist(
            generate_documentation_template,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct<'a, T> {
    pub x: &'a T,
}
impl<'a, T> MyGenericStruct<'a, T> {
    pub fn new$0(x: &'a T) -> Self {
        MyGenericStruct { x }
    }
}
"#,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct<'a, T> {
    pub x: &'a T,
}
impl<'a, T> MyGenericStruct<'a, T> {
    /// Creates a new [`MyGenericStruct<T>`].
    pub fn new(x: &'a T) -> Self {
        MyGenericStruct { x }
    }
}
"#,
        );
    }

    #[test]
    fn removes_all_lifetimes_from_description() {
        check_assist(
            generate_documentation_template,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct<'a, 'b, T> {
    pub x: &'a T,
    pub y: &'b T,
}
impl<'a, 'b, T> MyGenericStruct<'a, 'b, T> {
    pub fn new$0(x: &'a T, y: &'b T) -> Self {
        MyGenericStruct { x, y }
    }
}
"#,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct<'a, 'b, T> {
    pub x: &'a T,
    pub y: &'b T,
}
impl<'a, 'b, T> MyGenericStruct<'a, 'b, T> {
    /// Creates a new [`MyGenericStruct<T>`].
    pub fn new(x: &'a T, y: &'b T) -> Self {
        MyGenericStruct { x, y }
    }
}
"#,
        );
    }

    #[test]
    fn removes_all_lifetimes_and_brackets_from_description() {
        check_assist(
            generate_documentation_template,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct<'a, 'b> {
    pub x: &'a usize,
    pub y: &'b usize,
}
impl<'a, 'b> MyGenericStruct<'a, 'b> {
    pub fn new$0(x: &'a usize, y: &'b usize) -> Self {
        MyGenericStruct { x, y }
    }
}
"#,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct<'a, 'b> {
    pub x: &'a usize,
    pub y: &'b usize,
}
impl<'a, 'b> MyGenericStruct<'a, 'b> {
    /// Creates a new [`MyGenericStruct`].
    pub fn new(x: &'a usize, y: &'b usize) -> Self {
        MyGenericStruct { x, y }
    }
}
"#,
        );
    }

    #[test]
    fn detects_new_with_self() {
        check_assist(
            generate_documentation_template,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct2<T> {
    pub x: T,
}
impl<T> MyGenericStruct2<T> {
    pub fn new$0(x: T) -> Self {
        MyGenericStruct2 { x }
    }
}
"#,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct2<T> {
    pub x: T,
}
impl<T> MyGenericStruct2<T> {
    /// Creates a new [`MyGenericStruct2<T>`].
    pub fn new(x: T) -> Self {
        MyGenericStruct2 { x }
    }
}
"#,
        );
    }

    #[test]
    fn supports_method_call() {
        check_assist(
            generate_doc_example,
            r#"
impl<T> MyGenericStruct<T> {
    ///$0
    pub fn consume(self) {}
}
"#,
            r#"
impl<T> MyGenericStruct<T> {
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use test::MyGenericStruct;
    ///
    /// let my_generic_struct = ;
    /// my_generic_struct.consume();
    /// ```
    pub fn consume(self) {}
}
"#,
        );
    }

    #[test]
    fn checks_modified_self_param() {
        check_assist(
            generate_doc_example,
            r#"
impl<T> MyGenericStruct<T> {
    ///$0
    pub fn modify(&mut self, new_value: T) {
        self.x = new_value;
    }
}
"#,
            r#"
impl<T> MyGenericStruct<T> {
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use test::MyGenericStruct;
    ///
    /// let mut my_generic_struct = ;
    /// my_generic_struct.modify(new_value);
    /// assert_eq!(my_generic_struct, );
    /// ```
    pub fn modify(&mut self, new_value: T) {
        self.x = new_value;
    }
}
"#,
        );
    }

    #[test]
    fn generates_intro_for_getters() {
        check_assist(
            generate_documentation_template,
            r#"
pub struct S;
impl S {
    pub fn speed$0(&self) -> f32 { 0.0 }
}
"#,
            r#"
pub struct S;
impl S {
    /// Returns the speed of this [`S`].
    pub fn speed(&self) -> f32 { 0.0 }
}
"#,
        );
        check_assist(
            generate_documentation_template,
            r#"
pub struct S;
impl S {
    pub fn data$0(&self) -> &[u8] { &[] }
}
"#,
            r#"
pub struct S;
impl S {
    /// Returns a reference to the data of this [`S`].
    pub fn data(&self) -> &[u8] { &[] }
}
"#,
        );
        check_assist(
            generate_documentation_template,
            r#"
pub struct S;
impl S {
    pub fn data$0(&mut self) -> &mut [u8] { &mut [] }
}
"#,
            r#"
pub struct S;
impl S {
    /// Returns a mutable reference to the data of this [`S`].
    pub fn data(&mut self) -> &mut [u8] { &mut [] }
}
"#,
        );
        check_assist(
            generate_documentation_template,
            r#"
pub struct S;
impl S {
    pub fn data_mut$0(&mut self) -> &mut [u8] { &mut [] }
}
"#,
            r#"
pub struct S;
impl S {
    /// Returns a mutable reference to the data of this [`S`].
    pub fn data_mut(&mut self) -> &mut [u8] { &mut [] }
}
"#,
        );
    }

    #[test]
    fn no_getter_intro_for_prefixed_methods() {
        check_assist(
            generate_documentation_template,
            r#"
pub struct S;
impl S {
    pub fn as_bytes$0(&self) -> &[u8] { &[] }
}
"#,
            r#"
pub struct S;
impl S {
    /// .
    pub fn as_bytes(&self) -> &[u8] { &[] }
}
"#,
        );
    }

    #[test]
    fn generates_intro_for_setters() {
        check_assist(
            generate_documentation_template,
            r#"
pub struct S;
impl S {
    pub fn set_data$0(&mut self, data: Vec<u8>) {}
}
"#,
            r#"
pub struct S;
impl S {
    /// Sets the data of this [`S`].
    pub fn set_data(&mut self, data: Vec<u8>) {}
}
"#,
        );
        check_assist(
            generate_documentation_template,
            r#"
pub struct S;
impl S {
    pub fn set_domain_name$0(&mut self, name: String) {}
}
"#,
            r#"
pub struct S;
impl S {
    /// Sets the domain name of this [`S`].
    pub fn set_domain_name(&mut self, name: String) {}
}
"#,
        );
    }
}
