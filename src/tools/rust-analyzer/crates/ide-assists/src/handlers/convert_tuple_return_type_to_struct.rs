use either::Either;
use hir::ModuleDef;
use ide_db::{
    FxHashSet,
    assists::AssistId,
    defs::Definition,
    helpers::mod_path_to_ast,
    imports::insert_use::{ImportScope, insert_use},
    search::{FileReference, UsageSearchResult},
    source_change::SourceChangeBuilder,
    syntax_helpers::node_ext::{for_each_tail_expr, walk_expr},
};
use syntax::{
    AstNode, SyntaxNode,
    ast::{self, HasName, edit::IndentLevel, edit_in_place::Indent, make},
    match_ast, ted,
};

use crate::assist_context::{AssistContext, Assists};

// Assist: convert_tuple_return_type_to_struct
//
// This converts the return type of a function from a tuple type
// into a tuple struct and updates the body accordingly.
//
// ```
// fn bar() {
//     let (a, b, c) = foo();
// }
//
// fn foo() -> ($0u32, u32, u32) {
//     (1, 2, 3)
// }
// ```
// ->
// ```
// fn bar() {
//     let FooResult(a, b, c) = foo();
// }
//
// struct FooResult(u32, u32, u32);
//
// fn foo() -> FooResult {
//     FooResult(1, 2, 3)
// }
// ```
pub(crate) fn convert_tuple_return_type_to_struct(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let ret_type = ctx.find_node_at_offset::<ast::RetType>()?;
    let type_ref = ret_type.ty()?;

    let ast::Type::TupleType(tuple_ty) = &type_ref else { return None };
    if tuple_ty.fields().any(|field| matches!(field, ast::Type::ImplTraitType(_))) {
        return None;
    }

    let fn_ = ret_type.syntax().parent().and_then(ast::Fn::cast)?;
    let fn_def = ctx.sema.to_def(&fn_)?;
    let fn_name = fn_.name()?;
    let target_module = ctx.sema.scope(fn_.syntax())?.module().nearest_non_block_module(ctx.db());

    let target = type_ref.syntax().text_range();
    acc.add(
        AssistId::refactor_rewrite("convert_tuple_return_type_to_struct"),
        "Convert tuple return type to tuple struct",
        target,
        move |edit| {
            let ret_type = edit.make_mut(ret_type);
            let fn_ = edit.make_mut(fn_);

            let usages = Definition::Function(fn_def).usages(&ctx.sema).all();
            let struct_name = format!("{}Result", stdx::to_camel_case(&fn_name.to_string()));
            let parent = fn_.syntax().ancestors().find_map(<Either<ast::Impl, ast::Trait>>::cast);
            add_tuple_struct_def(
                edit,
                ctx,
                &usages,
                parent.as_ref().map(|it| it.syntax()).unwrap_or(fn_.syntax()),
                tuple_ty,
                &struct_name,
                &target_module,
            );

            ted::replace(
                ret_type.syntax(),
                make::ret_type(make::ty(&struct_name)).syntax().clone_for_update(),
            );

            if let Some(fn_body) = fn_.body() {
                replace_body_return_values(ast::Expr::BlockExpr(fn_body), &struct_name);
            }

            replace_usages(edit, ctx, &usages, &struct_name, &target_module);
        },
    )
}

/// Replaces tuple usages with the corresponding tuple struct pattern.
fn replace_usages(
    edit: &mut SourceChangeBuilder,
    ctx: &AssistContext<'_>,
    usages: &UsageSearchResult,
    struct_name: &str,
    target_module: &hir::Module,
) {
    for (file_id, references) in usages.iter() {
        edit.edit_file(file_id.file_id(ctx.db()));

        let refs_with_imports =
            augment_references_with_imports(edit, ctx, references, struct_name, target_module);

        refs_with_imports.into_iter().rev().for_each(|(name, import_data)| {
            if let Some(fn_) = name.syntax().parent().and_then(ast::Fn::cast) {
                cov_mark::hit!(replace_trait_impl_fns);

                if let Some(ret_type) = fn_.ret_type() {
                    ted::replace(
                        ret_type.syntax(),
                        make::ret_type(make::ty(struct_name)).syntax().clone_for_update(),
                    );
                }

                if let Some(fn_body) = fn_.body() {
                    replace_body_return_values(ast::Expr::BlockExpr(fn_body), struct_name);
                }
            } else {
                // replace tuple patterns
                let pats = name
                    .syntax()
                    .ancestors()
                    .find(|node| {
                        ast::CallExpr::can_cast(node.kind())
                            || ast::MethodCallExpr::can_cast(node.kind())
                    })
                    .and_then(|node| node.parent())
                    .and_then(node_to_pats)
                    .unwrap_or(Vec::new());

                let tuple_pats = pats.iter().filter_map(|pat| match pat {
                    ast::Pat::TuplePat(tuple_pat) => Some(tuple_pat),
                    _ => None,
                });
                for tuple_pat in tuple_pats {
                    ted::replace(
                        tuple_pat.syntax(),
                        make::tuple_struct_pat(
                            make::path_from_text(struct_name),
                            tuple_pat.fields(),
                        )
                        .clone_for_update()
                        .syntax(),
                    );
                }
            }
            // add imports across modules where needed
            if let Some((import_scope, path)) = import_data {
                insert_use(&import_scope, path, &ctx.config.insert_use);
            }
        })
    }
}

fn node_to_pats(node: SyntaxNode) -> Option<Vec<ast::Pat>> {
    match_ast! {
        match node {
            ast::LetStmt(it) => it.pat().map(|pat| vec![pat]),
            ast::LetExpr(it) => it.pat().map(|pat| vec![pat]),
            ast::MatchExpr(it) => it.match_arm_list().map(|arm_list| {
                arm_list.arms().filter_map(|arm| arm.pat()).collect()
            }),
            _ => None,
        }
    }
}

fn augment_references_with_imports(
    edit: &mut SourceChangeBuilder,
    ctx: &AssistContext<'_>,
    references: &[FileReference],
    struct_name: &str,
    target_module: &hir::Module,
) -> Vec<(ast::NameLike, Option<(ImportScope, ast::Path)>)> {
    let mut visited_modules = FxHashSet::default();

    let cfg = ctx.config.import_path_config();

    references
        .iter()
        .filter_map(|FileReference { name, .. }| {
            let name = name.clone().into_name_like()?;
            ctx.sema.scope(name.syntax()).map(|scope| (name, scope.module()))
        })
        .map(|(name, ref_module)| {
            let new_name = edit.make_mut(name);

            // if the referenced module is not the same as the target one and has not been seen before, add an import
            let import_data = if ref_module.nearest_non_block_module(ctx.db()) != *target_module
                && !visited_modules.contains(&ref_module)
            {
                visited_modules.insert(ref_module);

                let import_scope =
                    ImportScope::find_insert_use_container(new_name.syntax(), &ctx.sema);
                let path = ref_module
                    .find_use_path(
                        ctx.sema.db,
                        ModuleDef::Module(*target_module),
                        ctx.config.insert_use.prefix_kind,
                        cfg,
                    )
                    .map(|mod_path| {
                        make::path_concat(
                            mod_path_to_ast(&mod_path, target_module.krate().edition(ctx.db())),
                            make::path_from_text(struct_name),
                        )
                    });

                import_scope.zip(path)
            } else {
                None
            };

            (new_name, import_data)
        })
        .collect()
}

// Adds the definition of the tuple struct before the parent function.
fn add_tuple_struct_def(
    edit: &mut SourceChangeBuilder,
    ctx: &AssistContext<'_>,
    usages: &UsageSearchResult,
    parent: &SyntaxNode,
    tuple_ty: &ast::TupleType,
    struct_name: &str,
    target_module: &hir::Module,
) {
    let make_struct_pub = usages
        .iter()
        .flat_map(|(_, refs)| refs)
        .filter_map(|FileReference { name, .. }| {
            let name = name.clone().into_name_like()?;
            ctx.sema.scope(name.syntax()).map(|scope| scope.module())
        })
        .any(|module| module.nearest_non_block_module(ctx.db()) != *target_module);
    let visibility = if make_struct_pub { Some(make::visibility_pub()) } else { None };

    let field_list = ast::FieldList::TupleFieldList(make::tuple_field_list(
        tuple_ty.fields().map(|ty| make::tuple_field(visibility.clone(), ty)),
    ));
    let struct_name = make::name(struct_name);
    let struct_def = make::struct_(visibility, struct_name, None, field_list).clone_for_update();

    let indent = IndentLevel::from_node(parent);
    struct_def.reindent_to(indent);

    edit.insert(parent.text_range().start(), format!("{struct_def}\n\n{indent}"));
}

/// Replaces each returned tuple in `body` with the constructor of the tuple struct named `struct_name`.
fn replace_body_return_values(body: ast::Expr, struct_name: &str) {
    let mut exprs_to_wrap = Vec::new();

    let tail_cb = &mut |e: &_| tail_cb_impl(&mut exprs_to_wrap, e);
    walk_expr(&body, &mut |expr| {
        if let ast::Expr::ReturnExpr(ret_expr) = expr
            && let Some(ret_expr_arg) = &ret_expr.expr()
        {
            for_each_tail_expr(ret_expr_arg, tail_cb);
        }
    });
    for_each_tail_expr(&body, tail_cb);

    for ret_expr in exprs_to_wrap {
        if let ast::Expr::TupleExpr(tuple_expr) = &ret_expr {
            let struct_constructor = make::expr_call(
                make::expr_path(make::ext::ident_path(struct_name)),
                make::arg_list(tuple_expr.fields()),
            )
            .clone_for_update();
            ted::replace(ret_expr.syntax(), struct_constructor.syntax());
        }
    }
}

fn tail_cb_impl(acc: &mut Vec<ast::Expr>, e: &ast::Expr) {
    match e {
        ast::Expr::BreakExpr(break_expr) => {
            if let Some(break_expr_arg) = break_expr.expr() {
                for_each_tail_expr(&break_expr_arg, &mut |e| tail_cb_impl(acc, e))
            }
        }
        ast::Expr::ReturnExpr(_) => {
            // all return expressions have already been handled by the walk loop
        }
        e => acc.push(e.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn function_basic() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar() -> $0(&'static str, bool) {
    ("bar", true)
}
"#,
            r#"
struct BarResult(&'static str, bool);

fn bar() -> BarResult {
    BarResult("bar", true)
}
"#,
        )
    }

    #[test]
    fn struct_and_usages_indented() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
mod foo {
    pub(crate) fn foo() {
        let (bar, baz) = bar();
        println!("{bar} {baz}");
    }

    pub(crate) fn bar() -> $0(usize, bool) {
        (42, true)
    }
}
"#,
            r#"
mod foo {
    pub(crate) fn foo() {
        let BarResult(bar, baz) = bar();
        println!("{bar} {baz}");
    }

    struct BarResult(usize, bool);

    pub(crate) fn bar() -> BarResult {
        BarResult(42, true)
    }
}
"#,
        )
    }

    #[test]
    fn field_usage() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar() -> $0(usize, bool) {
    (42, true)
}

fn main() {
    let bar_result = bar();
    println!("{} {}", bar_result.1, bar().0);
}
"#,
            r#"
struct BarResult(usize, bool);

fn bar() -> BarResult {
    BarResult(42, true)
}

fn main() {
    let bar_result = bar();
    println!("{} {}", bar_result.1, bar().0);
}
"#,
        )
    }

    #[test]
    fn method_usage() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
struct Foo;

impl Foo {
    fn foo(&self, x: usize) -> $0(usize, usize) {
        (x, x)
    }
}

fn main() {
    let foo = Foo {};
    let (x, y) = foo.foo(2);
}
"#,
            r#"
struct Foo;

struct FooResult(usize, usize);

impl Foo {
    fn foo(&self, x: usize) -> FooResult {
        FooResult(x, x)
    }
}

fn main() {
    let foo = Foo {};
    let FooResult(x, y) = foo.foo(2);
}
"#,
        )
    }

    #[test]
    fn method_usage_within_same_impl() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
struct Foo;

impl Foo {
    fn new() -> $0(usize, usize) {
        (0, 0)
    }

    fn foo() {
        let (mut foo1, mut foo2) = Self::new();
    }
}
"#,
            r#"
struct Foo;

struct NewResult(usize, usize);

impl Foo {
    fn new() -> NewResult {
        NewResult(0, 0)
    }

    fn foo() {
        let NewResult(mut foo1, mut foo2) = Self::new();
    }
}
"#,
        )
    }

    #[test]
    fn multiple_usages() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar() -> $0(usize, usize) {
    (42, 24)
}

fn main() {
    let bar_result = bar();
    let (foo, b) = bar();
    let (b, baz) = bar();

    if foo == b && b == baz {
        println!("{} {}", bar_result.1, bar().0);
    }
}
"#,
            r#"
struct BarResult(usize, usize);

fn bar() -> BarResult {
    BarResult(42, 24)
}

fn main() {
    let bar_result = bar();
    let BarResult(foo, b) = bar();
    let BarResult(b, baz) = bar();

    if foo == b && b == baz {
        println!("{} {}", bar_result.1, bar().0);
    }
}
"#,
        )
    }

    #[test]
    fn usage_match_tuple_pat() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar() -> $0(usize, bool) {
    (42, true)
}

fn main() {
    match bar() {
        x if x.0 == 0 => println!("0"),
        (x, false) => println!("{x}"),
        (42, true) => println!("bar"),
        _ => println!("foo"),
    }
}
"#,
            r#"
struct BarResult(usize, bool);

fn bar() -> BarResult {
    BarResult(42, true)
}

fn main() {
    match bar() {
        x if x.0 == 0 => println!("0"),
        BarResult(x, false) => println!("{x}"),
        BarResult(42, true) => println!("bar"),
        _ => println!("foo"),
    }
}
"#,
        )
    }

    #[test]
    fn usage_if_let_tuple_pat() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar() -> $0(usize, bool) {
    (42, true)
}

fn main() {
    if let (42, true) = bar() {
        println!("bar")
    }
}
"#,
            r#"
struct BarResult(usize, bool);

fn bar() -> BarResult {
    BarResult(42, true)
}

fn main() {
    if let BarResult(42, true) = bar() {
        println!("bar")
    }
}
"#,
        )
    }

    #[test]
    fn function_nested_outer() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar() -> $0(usize, bool) {
    fn foo() -> (usize, bool) {
        (42, true)
    }

    foo()
}
"#,
            r#"
struct BarResult(usize, bool);

fn bar() -> BarResult {
    fn foo() -> (usize, bool) {
        (42, true)
    }

    foo()
}
"#,
        )
    }

    #[test]
    fn function_nested_inner() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar() -> (usize, bool) {
    fn foo() -> $0(usize, bool) {
        (42, true)
    }

    foo()
}
"#,
            r#"
fn bar() -> (usize, bool) {
    struct FooResult(usize, bool);

    fn foo() -> FooResult {
        FooResult(42, true)
    }

    foo()
}
"#,
        )
    }

    #[test]
    fn trait_impl_and_usage() {
        cov_mark::check!(replace_trait_impl_fns);
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
struct Struct;

trait Foo {
    fn foo(&self) -> $0(usize, bool);
}

impl Foo for Struct {
    fn foo(&self) -> (usize, bool) {
        (0, true)
    }
}

fn main() {
    let s = Struct {};
    let (foo, bar) = s.foo();
    let (foo, bar) = Struct::foo(&s);
    println!("{foo} {bar}");
}
"#,
            r#"
struct Struct;

struct FooResult(usize, bool);

trait Foo {
    fn foo(&self) -> FooResult;
}

impl Foo for Struct {
    fn foo(&self) -> FooResult {
        FooResult(0, true)
    }
}

fn main() {
    let s = Struct {};
    let FooResult(foo, bar) = s.foo();
    let FooResult(foo, bar) = Struct::foo(&s);
    println!("{foo} {bar}");
}
"#,
        )
    }

    #[test]
    fn body_wraps_nested() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn foo() -> $0(u8, usize, u32) {
    if true {
        match 3 {
            0 => (1, 2, 3),
            _ => return (4, 5, 6),
        }
    } else {
        (2, 1, 3)
    }
}
"#,
            r#"
struct FooResult(u8, usize, u32);

fn foo() -> FooResult {
    if true {
        match 3 {
            0 => FooResult(1, 2, 3),
            _ => return FooResult(4, 5, 6),
        }
    } else {
        FooResult(2, 1, 3)
    }
}
"#,
        )
    }

    #[test]
    fn body_wraps_break_and_return() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn foo(mut i: isize) -> (usize, $0u32, u8) {
    if i < 0 {
        return (0, 0, 0);
    }

    loop {
        if i == 2 {
            println!("foo");
            break (1, 2, 3);
        }
        i += 1;
    }
}
"#,
            r#"
struct FooResult(usize, u32, u8);

fn foo(mut i: isize) -> FooResult {
    if i < 0 {
        return FooResult(0, 0, 0);
    }

    loop {
        if i == 2 {
            println!("foo");
            break FooResult(1, 2, 3);
        }
        i += 1;
    }
}
"#,
        )
    }

    #[test]
    fn body_doesnt_wrap_identifier() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn foo() -> $0(u8, usize, u32) {
    let tuple = (1, 2, 3);
    tuple
}
"#,
            r#"
struct FooResult(u8, usize, u32);

fn foo() -> FooResult {
    let tuple = (1, 2, 3);
    tuple
}
"#,
        )
    }

    #[test]
    fn body_doesnt_wrap_other_exprs() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar(num: usize) -> (u8, usize, u32) {
    (1, num, 3)
}

fn foo() -> $0(u8, usize, u32) {
    bar(2)
}
"#,
            r#"
fn bar(num: usize) -> (u8, usize, u32) {
    (1, num, 3)
}

struct FooResult(u8, usize, u32);

fn foo() -> FooResult {
    bar(2)
}
"#,
        )
    }

    #[test]
    fn cross_file_and_module() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
//- /main.rs
mod foo;

fn main() {
    use foo::bar;

    let (bar, baz) = bar::bar();
    println!("{}", bar == baz);
}

//- /foo.rs
pub mod bar {
    pub fn bar() -> $0(usize, usize) {
        (1, 3)
    }
}
"#,
            r#"
//- /main.rs
use foo::bar::BarResult;

mod foo;

fn main() {
    use foo::bar;

    let BarResult(bar, baz) = bar::bar();
    println!("{}", bar == baz);
}

//- /foo.rs
pub mod bar {
    pub struct BarResult(pub usize, pub usize);

    pub fn bar() -> BarResult {
        BarResult(1, 3)
    }
}
"#,
        )
    }

    #[test]
    fn does_not_replace_nested_usage() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar() -> $0(usize, bool) {
    (42, true)
}

fn main() {
    let ((bar1, bar2), foo) = (bar(), 3);
    println!("{bar1} {bar2} {foo}");
}
"#,
            r#"
struct BarResult(usize, bool);

fn bar() -> BarResult {
    BarResult(42, true)
}

fn main() {
    let ((bar1, bar2), foo) = (bar(), 3);
    println!("{bar1} {bar2} {foo}");
}
"#,
        )
    }

    #[test]
    fn function_with_non_tuple_return_type() {
        check_assist_not_applicable(
            convert_tuple_return_type_to_struct,
            r#"
fn bar() -> $0usize {
    0
}
"#,
        )
    }

    #[test]
    fn function_with_impl_type() {
        check_assist_not_applicable(
            convert_tuple_return_type_to_struct,
            r#"
fn bar() -> $0(impl Clone, usize) {
    ("bar", 0)
}
"#,
        )
    }
}
