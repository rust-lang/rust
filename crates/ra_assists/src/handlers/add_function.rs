use ra_syntax::{
    ast::{self, AstNode},
    SyntaxKind, SyntaxNode, TextSize,
};

use crate::{Assist, AssistCtx, AssistFile, AssistId};
use ast::{edit::IndentLevel, ArgListOwner, ModuleItemOwner};
use hir::HirDisplay;
use rustc_hash::{FxHashMap, FxHashSet};

// Assist: add_function
//
// Adds a stub function with a signature matching the function under the cursor.
//
// ```
// struct Baz;
// fn baz() -> Baz { Baz }
// fn foo() {
//     bar<|>("", baz());
// }
//
// ```
// ->
// ```
// struct Baz;
// fn baz() -> Baz { Baz }
// fn foo() {
//     bar("", baz());
// }
//
// fn bar(arg: &str, baz: Baz) {
//     todo!()
// }
//
// ```
pub(crate) fn add_function(ctx: AssistCtx) -> Option<Assist> {
    let path_expr: ast::PathExpr = ctx.find_node_at_offset()?;
    let call = path_expr.syntax().parent().and_then(ast::CallExpr::cast)?;
    let path = path_expr.path()?;

    if ctx.sema.resolve_path(&path).is_some() {
        // The function call already resolves, no need to add a function
        return None;
    }

    let target_module = if let Some(qualifier) = path.qualifier() {
        if let Some(hir::PathResolution::Def(hir::ModuleDef::Module(module))) =
            ctx.sema.resolve_path(&qualifier)
        {
            Some(module.definition_source(ctx.sema.db))
        } else {
            return None;
        }
    } else {
        None
    };

    let function_builder = FunctionBuilder::from_call(&ctx, &call, &path, target_module)?;

    let target = call.syntax().text_range();
    ctx.add_assist(AssistId("add_function"), "Add function", target, |edit| {
        let function_template = function_builder.render();
        edit.set_file(function_template.file);
        edit.set_cursor(function_template.cursor_offset);
        edit.insert(function_template.insert_offset, function_template.fn_def.to_string());
    })
}

struct FunctionTemplate {
    insert_offset: TextSize,
    cursor_offset: TextSize,
    fn_def: ast::SourceFile,
    file: AssistFile,
}

struct FunctionBuilder {
    target: GeneratedFunctionTarget,
    fn_name: ast::Name,
    type_params: Option<ast::TypeParamList>,
    params: ast::ParamList,
    file: AssistFile,
    needs_pub: bool,
}

impl FunctionBuilder {
    /// Prepares a generated function that matches `call` in `generate_in`
    /// (or as close to `call` as possible, if `generate_in` is `None`)
    fn from_call(
        ctx: &AssistCtx,
        call: &ast::CallExpr,
        path: &ast::Path,
        target_module: Option<hir::InFile<hir::ModuleSource>>,
    ) -> Option<Self> {
        let needs_pub = target_module.is_some();
        let mut file = AssistFile::default();
        let target = if let Some(target_module) = target_module {
            let (in_file, target) = next_space_for_fn_in_module(ctx.sema.db, target_module)?;
            file = in_file;
            target
        } else {
            next_space_for_fn_after_call_site(&call)?
        };
        let fn_name = fn_name(&path)?;
        let (type_params, params) = fn_args(ctx, &call)?;
        Some(Self { target, fn_name, type_params, params, file, needs_pub })
    }

    fn render(self) -> FunctionTemplate {
        let placeholder_expr = ast::make::expr_todo();
        let fn_body = ast::make::block_expr(vec![], Some(placeholder_expr));
        let mut fn_def = ast::make::fn_def(self.fn_name, self.type_params, self.params, fn_body);
        if self.needs_pub {
            fn_def = ast::make::add_pub_crate_modifier(fn_def);
        }

        let (fn_def, insert_offset) = match self.target {
            GeneratedFunctionTarget::BehindItem(it) => {
                let with_leading_blank_line = ast::make::add_leading_newlines(2, fn_def);
                let indented = IndentLevel::from_node(&it).increase_indent(with_leading_blank_line);
                (indented, it.text_range().end())
            }
            GeneratedFunctionTarget::InEmptyItemList(it) => {
                let indent_once = IndentLevel(1);
                let indent = IndentLevel::from_node(it.syntax());

                let fn_def = ast::make::add_leading_newlines(1, fn_def);
                let fn_def = indent_once.increase_indent(fn_def);
                let fn_def = ast::make::add_trailing_newlines(1, fn_def);
                let fn_def = indent.increase_indent(fn_def);
                (fn_def, it.syntax().text_range().start() + TextSize::of('{'))
            }
        };

        let placeholder_expr =
            fn_def.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();
        let cursor_offset_from_fn_start = placeholder_expr.syntax().text_range().start();
        let cursor_offset = insert_offset + cursor_offset_from_fn_start;
        FunctionTemplate { insert_offset, cursor_offset, fn_def, file: self.file }
    }
}

enum GeneratedFunctionTarget {
    BehindItem(SyntaxNode),
    InEmptyItemList(ast::ItemList),
}

fn fn_name(call: &ast::Path) -> Option<ast::Name> {
    let name = call.segment()?.syntax().to_string();
    Some(ast::make::name(&name))
}

/// Computes the type variables and arguments required for the generated function
fn fn_args(
    ctx: &AssistCtx,
    call: &ast::CallExpr,
) -> Option<(Option<ast::TypeParamList>, ast::ParamList)> {
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
    let params = arg_names.into_iter().zip(arg_types).map(|(name, ty)| ast::make::param(name, ty));
    Some((None, ast::make::param_list(params)))
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

fn fn_arg_name(fn_arg: &ast::Expr) -> Option<String> {
    match fn_arg {
        ast::Expr::CastExpr(cast_expr) => fn_arg_name(&cast_expr.expr()?),
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

fn fn_arg_type(ctx: &AssistCtx, fn_arg: &ast::Expr) -> Option<String> {
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
fn next_space_for_fn_after_call_site(expr: &ast::CallExpr) -> Option<GeneratedFunctionTarget> {
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
    last_ancestor.map(GeneratedFunctionTarget::BehindItem)
}

fn next_space_for_fn_in_module(
    db: &dyn hir::db::AstDatabase,
    module: hir::InFile<hir::ModuleSource>,
) -> Option<(AssistFile, GeneratedFunctionTarget)> {
    let file = module.file_id.original_file(db);
    let assist_file = AssistFile::TargetFile(file);
    let assist_item = match module.value {
        hir::ModuleSource::SourceFile(it) => {
            if let Some(last_item) = it.items().last() {
                GeneratedFunctionTarget::BehindItem(last_item.syntax().clone())
            } else {
                GeneratedFunctionTarget::BehindItem(it.syntax().clone())
            }
        }
        hir::ModuleSource::Module(it) => {
            if let Some(last_item) = it.item_list().and_then(|it| it.items().last()) {
                GeneratedFunctionTarget::BehindItem(last_item.syntax().clone())
            } else {
                GeneratedFunctionTarget::InEmptyItemList(it.item_list()?)
            }
        }
    };
    Some((assist_file, assist_item))
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

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
    <|>todo!()
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
    <|>todo!()
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
    <|>todo!()
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
        <|>todo!()
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
fn baz() -> Baz { todo!() }
fn foo() {
    bar<|>(baz());
}
",
            r"
struct Baz;
fn baz() -> Baz { todo!() }
fn foo() {
    bar(baz());
}

fn bar(baz: Baz) {
    <|>todo!()
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
    <|>todo!()
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
    <|>todo!()
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
    <|>todo!()
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
    <|>todo!()
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
    <|>todo!()
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
    <|>todo!()
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
    <|>todo!()
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
    todo!()
}
fn baz() {
    <|>bar(foo())
}
",
            r"
trait Foo {}
fn foo() -> impl Foo {
    todo!()
}
fn baz() {
    bar(foo())
}

fn bar(foo: impl Foo) {
    <|>todo!()
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
        <|>todo!()
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
    <|>todo!()
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
    <|>todo!()
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
    <|>todo!()
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
    <|>todo!()
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
    <|>todo!()
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
    <|>todo!()
}
"#,
        )
    }

    #[test]
    fn add_function_in_module() {
        check_assist(
            add_function,
            r"
mod bar {}

fn foo() {
    bar::my_fn<|>()
}
",
            r"
mod bar {
    pub(crate) fn my_fn() {
        <|>todo!()
    }
}

fn foo() {
    bar::my_fn()
}
",
        )
    }

    #[test]
    fn add_function_in_module_containing_other_items() {
        check_assist(
            add_function,
            r"
mod bar {
    fn something_else() {}
}

fn foo() {
    bar::my_fn<|>()
}
",
            r"
mod bar {
    fn something_else() {}

    pub(crate) fn my_fn() {
        <|>todo!()
    }
}

fn foo() {
    bar::my_fn()
}
",
        )
    }

    #[test]
    fn add_function_in_nested_module() {
        check_assist(
            add_function,
            r"
mod bar {
    mod baz {}
}

fn foo() {
    bar::baz::my_fn<|>()
}
",
            r"
mod bar {
    mod baz {
        pub(crate) fn my_fn() {
            <|>todo!()
        }
    }
}

fn foo() {
    bar::baz::my_fn()
}
",
        )
    }

    #[test]
    fn add_function_in_another_file() {
        check_assist(
            add_function,
            r"
//- /main.rs
mod foo;

fn main() {
    foo::bar<|>()
}
//- /foo.rs
",
            r"


pub(crate) fn bar() {
    <|>todo!()
}",
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
        todo!();
    }
}
        ",
        )
    }
}
