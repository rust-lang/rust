use ra_syntax::{
    ast::{self, AstNode},
    SmolStr, SyntaxKind, SyntaxNode, TextUnit,
};

use crate::{Assist, AssistCtx, AssistId};
use ast::{ArgListOwner, CallExpr, Expr};
use hir::HirDisplay;
use ra_fmt::leading_indent;
use rustc_hash::{FxHashMap, FxHashSet};

// Assist: add_function
//
// Adds a stub function with a signature matching the function under the cursor.
//
// ```
// struct Baz;
// fn baz() -> Baz { Baz }
// fn foo() {
//      bar<|>("", baz());
// }
//
// ```
// ->
// ```
// struct Baz;
// fn baz() -> Baz { Baz }
// fn foo() {
//      bar("", baz());
// }
//
// fn bar(arg: &str, baz: Baz) {
//     unimplemented!()
// }
//
// ```
pub(crate) fn add_function(ctx: AssistCtx) -> Option<Assist> {
    let path_expr: ast::PathExpr = ctx.find_node_at_offset()?;
    let call = path_expr.syntax().parent().and_then(ast::CallExpr::cast)?;
    let path = path_expr.path()?;

    if path.qualifier().is_some() {
        return None;
    }

    if ctx.sema.resolve_path(&path).is_some() {
        // The function call already resolves, no need to add a function
        return None;
    }

    let function_builder = FunctionBuilder::from_call(&ctx, &call)?;

    ctx.add_assist(AssistId("add_function"), "Add function", |edit| {
        edit.target(call.syntax().text_range());

        let function_template = function_builder.render();
        edit.set_cursor(function_template.cursor_offset);
        edit.insert(function_template.insert_offset, function_template.fn_text);
    })
}

struct FunctionTemplate {
    insert_offset: TextUnit,
    cursor_offset: TextUnit,
    fn_text: String,
}

struct FunctionBuilder {
    start_offset: TextUnit,
    fn_name: String,
    fn_generics: String,
    fn_args: String,
    indent: String,
}

impl FunctionBuilder {
    fn from_call(ctx: &AssistCtx, call: &ast::CallExpr) -> Option<Self> {
        let (start, indent) = next_space_for_fn(&call)?;
        let fn_name = fn_name(&call)?;
        let fn_generics = fn_generics(&call)?;
        let fn_args = fn_args(ctx, &call)?;
        let indent = if let Some(i) = &indent { i.to_string() } else { String::new() };
        Some(Self { start_offset: start, fn_name, fn_generics, fn_args, indent })
    }
    fn render(&self) -> FunctionTemplate {
        let mut fn_buf = String::with_capacity(128);
        fn_buf.push_str("\n\n");
        fn_buf.push_str(&self.indent);
        fn_buf.push_str("fn ");
        fn_buf.push_str(&self.fn_name);
        fn_buf.push_str(&self.fn_generics);
        fn_buf.push_str(&self.fn_args);
        fn_buf.push_str(" {\n");
        fn_buf.push_str(&self.indent);
        fn_buf.push_str("    ");

        // We take the offset here to put the cursor in front of the `unimplemented!()` body
        let offset = TextUnit::of_str(&fn_buf);

        fn_buf.push_str("unimplemented!()\n");
        fn_buf.push_str(&self.indent);
        fn_buf.push_str("}");

        let cursor_pos = self.start_offset + offset;
        FunctionTemplate {
            fn_text: fn_buf,
            cursor_offset: cursor_pos,
            insert_offset: self.start_offset,
        }
    }
}

fn fn_name(call: &CallExpr) -> Option<String> {
    Some(call.expr()?.syntax().to_string())
}

fn fn_generics(_call: &CallExpr) -> Option<String> {
    // TODO
    Some("".into())
}

fn fn_args(ctx: &AssistCtx, call: &CallExpr) -> Option<String> {
    let mut arg_names = Vec::new();
    let mut arg_types = Vec::new();
    for arg in call.arg_list()?.args() {
        let arg_name = match fn_arg_name(&arg) {
            Some(name) => name,
            None => String::from("arg"),
        };
        arg_names.push(arg_name);
        arg_types.push(match fn_arg_type(ctx, &arg) {
            Some(ty) => ty,
            None => String::from("()"),
        });
    }
    deduplicate_arg_names(&mut arg_names);
    Some(format!(
        "({})",
        arg_names
            .into_iter()
            .zip(arg_types)
            .map(|(name, ty)| format!("{}: {}", name, ty))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Makes duplicate argument names unique by appending incrementing numbers.
///
/// ```
/// let mut names: Vec<String> =
///     vec!["foo".into(), "foo".into(), "bar".into(), "baz".into(), "bar".into()];
/// deduplicate_arg_names(&mut names);
/// let expected: Vec<String> =
///     vec!["foo_1".into(), "foo_2".into(), "bar_1".into(), "baz".into(), "bar_2".into()];
/// assert_eq!(names, expected);
/// ```
fn deduplicate_arg_names(arg_names: &mut Vec<String>) {
    let arg_name_counts = arg_names.iter().fold(FxHashMap::default(), |mut m, name| {
        *m.entry(name).or_insert(0) += 1;
        m
    });
    let duplicate_arg_names: FxHashSet<String> = arg_name_counts
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .map(|(name, _)| name.clone())
        .collect();

    let mut counter_per_name = FxHashMap::default();
    for arg_name in arg_names.iter_mut() {
        if duplicate_arg_names.contains(arg_name) {
            let counter = counter_per_name.entry(arg_name.clone()).or_insert(1);
            arg_name.push('_');
            arg_name.push_str(&counter.to_string());
            *counter += 1;
        }
    }
}

fn fn_arg_name(fn_arg: &Expr) -> Option<String> {
    match fn_arg {
        Expr::CastExpr(cast_expr) => fn_arg_name(&cast_expr.expr()?),
        _ => Some(
            fn_arg
                .syntax()
                .descendants()
                .filter(|d| ast::NameRef::can_cast(d.kind()))
                .last()?
                .to_string(),
        ),
    }
}

fn fn_arg_type(ctx: &AssistCtx, fn_arg: &Expr) -> Option<String> {
    let ty = ctx.sema.type_of_expr(fn_arg)?;
    if ty.is_unknown() {
        return None;
    }
    Some(ty.display(ctx.sema.db).to_string())
}

/// Returns the position inside the current mod or file
/// directly after the current block
/// We want to write the generated function directly after
/// fns, impls or macro calls, but inside mods
fn next_space_for_fn(expr: &CallExpr) -> Option<(TextUnit, Option<SmolStr>)> {
    let mut ancestors = expr.syntax().ancestors().peekable();
    let mut last_ancestor: Option<SyntaxNode> = None;
    while let Some(next_ancestor) = ancestors.next() {
        match next_ancestor.kind() {
            SyntaxKind::SOURCE_FILE => {
                break;
            }
            SyntaxKind::ITEM_LIST => {
                if ancestors.peek().map(|a| a.kind()) == Some(SyntaxKind::MODULE) {
                    break;
                }
            }
            _ => {}
        }
        last_ancestor = Some(next_ancestor);
    }
    last_ancestor.map(|a| (a.text_range().end(), leading_indent(&a)))
}

#[cfg(test)]
mod tests {
    use crate::helpers::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn add_function_with_no_args() {
        check_assist(
            add_function,
            r"
fn foo() {
    bar<|>();
}
",
            r"
fn foo() {
    bar();
}

fn bar() {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    fn add_function_from_method() {
        // This ensures that the function is correctly generated
        // in the next outer mod or file
        check_assist(
            add_function,
            r"
impl Foo {
    fn foo() {
        bar<|>();
    }
}
",
            r"
impl Foo {
    fn foo() {
        bar();
    }
}

fn bar() {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    fn add_function_directly_after_current_block() {
        // The new fn should not be created at the end of the file or module
        check_assist(
            add_function,
            r"
fn foo1() {
    bar<|>();
}

fn foo2() {}
",
            r"
fn foo1() {
    bar();
}

fn bar() {
    <|>unimplemented!()
}

fn foo2() {}
",
        )
    }

    #[test]
    fn add_function_with_no_args_in_same_module() {
        check_assist(
            add_function,
            r"
mod baz {
    fn foo() {
        bar<|>();
    }
}
",
            r"
mod baz {
    fn foo() {
        bar();
    }

    fn bar() {
        <|>unimplemented!()
    }
}
",
        )
    }

    #[test]
    fn add_function_with_function_call_arg() {
        check_assist(
            add_function,
            r"
struct Baz;
fn baz() -> Baz { unimplemented!() }
fn foo() {
    bar<|>(baz());
}
",
            r"
struct Baz;
fn baz() -> Baz { unimplemented!() }
fn foo() {
    bar(baz());
}

fn bar(baz: Baz) {
    <|>unimplemented!()
}
",
        );
    }

    #[test]
    fn add_function_with_method_call_arg() {
        check_assist(
            add_function,
            r"
struct Baz;
impl Baz {
    fn foo(&self) -> Baz {
        ba<|>r(self.baz())
    }
    fn baz(&self) -> Baz {
        Baz
    }
}
",
            r"
struct Baz;
impl Baz {
    fn foo(&self) -> Baz {
        bar(self.baz())
    }
    fn baz(&self) -> Baz {
        Baz
    }
}

fn bar(baz: Baz) {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    fn add_function_with_string_literal_arg() {
        check_assist(
            add_function,
            r#"
fn foo() {
    <|>bar("bar")
}
"#,
            r#"
fn foo() {
    bar("bar")
}

fn bar(arg: &str) {
    <|>unimplemented!()
}
"#,
        )
    }

    #[test]
    fn add_function_with_char_literal_arg() {
        check_assist(
            add_function,
            r#"
fn foo() {
    <|>bar('x')
}
"#,
            r#"
fn foo() {
    bar('x')
}

fn bar(arg: char) {
    <|>unimplemented!()
}
"#,
        )
    }

    #[test]
    fn add_function_with_int_literal_arg() {
        check_assist(
            add_function,
            r"
fn foo() {
    <|>bar(42)
}
",
            r"
fn foo() {
    bar(42)
}

fn bar(arg: i32) {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    fn add_function_with_cast_int_literal_arg() {
        check_assist(
            add_function,
            r"
fn foo() {
    <|>bar(42 as u8)
}
",
            r"
fn foo() {
    bar(42 as u8)
}

fn bar(arg: u8) {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    fn name_of_cast_variable_is_used() {
        // Ensures that the name of the cast type isn't used
        // in the generated function signature.
        check_assist(
            add_function,
            r"
fn foo() {
    let x = 42;
    bar<|>(x as u8)
}
",
            r"
fn foo() {
    let x = 42;
    bar(x as u8)
}

fn bar(x: u8) {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    fn add_function_with_variable_arg() {
        check_assist(
            add_function,
            r"
fn foo() {
    let worble = ();
    <|>bar(worble)
}
",
            r"
fn foo() {
    let worble = ();
    bar(worble)
}

fn bar(worble: ()) {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    fn add_function_with_impl_trait_arg() {
        check_assist(
            add_function,
            r"
trait Foo {}
fn foo() -> impl Foo {
    unimplemented!()
}
fn baz() {
    <|>bar(foo())
}
",
            r"
trait Foo {}
fn foo() -> impl Foo {
    unimplemented!()
}
fn baz() {
    bar(foo())
}

fn bar(foo: impl Foo) {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    #[ignore]
    // FIXME print paths properly to make this test pass
    fn add_function_with_qualified_path_arg() {
        check_assist(
            add_function,
            r"
mod Baz {
    pub struct Bof;
    pub fn baz() -> Bof { Bof }
}
mod Foo {
    fn foo() {
        <|>bar(super::Baz::baz())
    }
}
",
            r"
mod Baz {
    pub struct Bof;
    pub fn baz() -> Bof { Bof }
}
mod Foo {
    fn foo() {
        bar(super::Baz::baz())
    }

    fn bar(baz: super::Baz::Bof) {
        <|>unimplemented!()
    }
}
",
        )
    }

    #[test]
    #[ignore]
    // FIXME fix printing the generics of a `Ty` to make this test pass
    fn add_function_with_generic_arg() {
        check_assist(
            add_function,
            r"
fn foo<T>(t: T) {
    <|>bar(t)
}
",
            r"
fn foo<T>(t: T) {
    bar(t)
}

fn bar<T>(t: T) {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    #[ignore]
    // FIXME Fix function type printing to make this test pass
    fn add_function_with_fn_arg() {
        check_assist(
            add_function,
            r"
struct Baz;
impl Baz {
    fn new() -> Self { Baz }
}
fn foo() {
    <|>bar(Baz::new);
}
",
            r"
struct Baz;
impl Baz {
    fn new() -> Self { Baz }
}
fn foo() {
    bar(Baz::new);
}

fn bar(arg: fn() -> Baz) {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    #[ignore]
    // FIXME Fix closure type printing to make this test pass
    fn add_function_with_closure_arg() {
        check_assist(
            add_function,
            r"
fn foo() {
    let closure = |x: i64| x - 1;
    <|>bar(closure)
}
",
            r"
fn foo() {
    let closure = |x: i64| x - 1;
    bar(closure)
}

fn bar(closure: impl Fn(i64) -> i64) {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    fn unresolveable_types_default_to_unit() {
        check_assist(
            add_function,
            r"
fn foo() {
    <|>bar(baz)
}
",
            r"
fn foo() {
    bar(baz)
}

fn bar(baz: ()) {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    fn arg_names_dont_overlap() {
        check_assist(
            add_function,
            r"
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    <|>bar(baz(), baz())
}
",
            r"
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    bar(baz(), baz())
}

fn bar(baz_1: Baz, baz_2: Baz) {
    <|>unimplemented!()
}
",
        )
    }

    #[test]
    fn arg_name_counters_start_at_1_per_name() {
        check_assist(
            add_function,
            r#"
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    <|>bar(baz(), baz(), "foo", "bar")
}
"#,
            r#"
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    bar(baz(), baz(), "foo", "bar")
}

fn bar(baz_1: Baz, baz_2: Baz, arg_1: &str, arg_2: &str) {
    <|>unimplemented!()
}
"#,
        )
    }

    #[test]
    fn add_function_not_applicable_if_function_already_exists() {
        check_assist_not_applicable(
            add_function,
            r"
fn foo() {
    bar<|>();
}

fn bar() {}
",
        )
    }

    #[test]
    fn add_function_not_applicable_if_unresolved_variable_in_call_is_selected() {
        check_assist_not_applicable(
            // bar is resolved, but baz isn't.
            // The assist is only active if the cursor is on an unresolved path,
            // but the assist should only be offered if the path is a function call.
            add_function,
            r"
fn foo() {
    bar(b<|>az);
}

fn bar(baz: ()) {}
",
        )
    }

    #[test]
    fn add_function_not_applicable_if_function_path_not_singleton() {
        // In the future this assist could be extended to generate functions
        // if the path is in the same crate (or even the same workspace).
        // For the beginning, I think this is fine.
        check_assist_not_applicable(
            add_function,
            r"
fn foo() {
    other_crate::bar<|>();
}
        ",
        )
    }

    #[test]
    #[ignore]
    fn create_method_with_no_args() {
        check_assist(
            add_function,
            r"
struct Foo;
impl Foo {
    fn foo(&self) {
        self.bar()<|>;
    }
}
        ",
            r"
struct Foo;
impl Foo {
    fn foo(&self) {
        self.bar();
    }
    fn bar(&self) {
        unimplemented!();
    }
}
        ",
        )
    }
}
