use hir::{HasSource, HirDisplay, Module, Semantics, TypeInfo};
use ide_db::{
    base_db::FileId,
    defs::{Definition, NameRefClass},
    famous_defs::FamousDefs,
    FxHashMap, FxHashSet, RootDatabase, SnippetCap,
};
use stdx::to_lower_snake_case;
use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make, AstNode, CallExpr, HasArgList, HasModuleItem,
    },
    SyntaxKind, SyntaxNode, TextRange, TextSize,
};

use crate::{
    utils::convert_reference_type,
    utils::{find_struct_impl, render_snippet, Cursor},
    AssistContext, AssistId, AssistKind, Assists,
};

// Assist: generate_function
//
// Adds a stub function with a signature matching the function under the cursor.
//
// ```
// struct Baz;
// fn baz() -> Baz { Baz }
// fn foo() {
//     bar$0("", baz());
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
// fn bar(arg: &str, baz: Baz) ${0:-> _} {
//     todo!()
// }
//
// ```
pub(crate) fn generate_function(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    gen_fn(acc, ctx).or_else(|| gen_method(acc, ctx))
}

fn gen_fn(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let path_expr: ast::PathExpr = ctx.find_node_at_offset()?;
    let call = path_expr.syntax().parent().and_then(ast::CallExpr::cast)?;
    let path = path_expr.path()?;
    let name_ref = path.segment()?.name_ref()?;
    if ctx.sema.resolve_path(&path).is_some() {
        // The function call already resolves, no need to add a function
        return None;
    }

    let fn_name = &*name_ref.text();
    let TargetInfo { target_module, adt_name, target, file, insert_offset } =
        fn_target_info(ctx, path, &call, fn_name)?;
    let function_builder = FunctionBuilder::from_call(ctx, &call, fn_name, target_module, target)?;
    let text_range = call.syntax().text_range();
    let label = format!("Generate {} function", function_builder.fn_name);
    add_func_to_accumulator(
        acc,
        ctx,
        text_range,
        function_builder,
        insert_offset,
        file,
        adt_name,
        label,
    )
}

struct TargetInfo {
    target_module: Option<Module>,
    adt_name: Option<hir::Name>,
    target: GeneratedFunctionTarget,
    file: FileId,
    insert_offset: TextSize,
}

impl TargetInfo {
    fn new(
        target_module: Option<Module>,
        adt_name: Option<hir::Name>,
        target: GeneratedFunctionTarget,
        file: FileId,
        insert_offset: TextSize,
    ) -> Self {
        Self { target_module, adt_name, target, file, insert_offset }
    }
}

fn fn_target_info(
    ctx: &AssistContext<'_>,
    path: ast::Path,
    call: &CallExpr,
    fn_name: &str,
) -> Option<TargetInfo> {
    match path.qualifier() {
        Some(qualifier) => match ctx.sema.resolve_path(&qualifier) {
            Some(hir::PathResolution::Def(hir::ModuleDef::Module(module))) => {
                get_fn_target_info(ctx, &Some(module), call.clone())
            }
            Some(hir::PathResolution::Def(hir::ModuleDef::Adt(adt))) => {
                if let hir::Adt::Enum(_) = adt {
                    // Don't suggest generating function if the name starts with an uppercase letter
                    if fn_name.starts_with(char::is_uppercase) {
                        return None;
                    }
                }

                assoc_fn_target_info(ctx, call, adt, fn_name)
            }
            Some(hir::PathResolution::SelfType(impl_)) => {
                let adt = impl_.self_ty(ctx.db()).as_adt()?;
                assoc_fn_target_info(ctx, call, adt, fn_name)
            }
            _ => None,
        },
        _ => get_fn_target_info(ctx, &None, call.clone()),
    }
}

fn gen_method(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let call: ast::MethodCallExpr = ctx.find_node_at_offset()?;
    if ctx.sema.resolve_method_call(&call).is_some() {
        return None;
    }

    let fn_name = call.name_ref()?;
    let adt = ctx.sema.type_of_expr(&call.receiver()?)?.original().strip_references().as_adt()?;

    let current_module = ctx.sema.scope(call.syntax())?.module();
    let target_module = adt.module(ctx.sema.db);

    if current_module.krate() != target_module.krate() {
        return None;
    }
    let (impl_, file) = get_adt_source(ctx, &adt, fn_name.text().as_str())?;
    let (target, insert_offset) = get_method_target(ctx, &target_module, &impl_)?;
    let function_builder =
        FunctionBuilder::from_method_call(ctx, &call, &fn_name, target_module, target)?;
    let text_range = call.syntax().text_range();
    let adt_name = if impl_.is_none() { Some(adt.name(ctx.sema.db)) } else { None };
    let label = format!("Generate {} method", function_builder.fn_name);
    add_func_to_accumulator(
        acc,
        ctx,
        text_range,
        function_builder,
        insert_offset,
        file,
        adt_name,
        label,
    )
}

fn add_func_to_accumulator(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    text_range: TextRange,
    function_builder: FunctionBuilder,
    insert_offset: TextSize,
    file: FileId,
    adt_name: Option<hir::Name>,
    label: String,
) -> Option<()> {
    acc.add(AssistId("generate_function", AssistKind::Generate), label, text_range, |builder| {
        let function_template = function_builder.render();
        let mut func = function_template.to_string(ctx.config.snippet_cap);
        if let Some(name) = adt_name {
            func = format!("\nimpl {} {{\n{}\n}}", name, func);
        }
        builder.edit_file(file);
        match ctx.config.snippet_cap {
            Some(cap) => builder.insert_snippet(cap, insert_offset, func),
            None => builder.insert(insert_offset, func),
        }
    })
}

fn get_adt_source(
    ctx: &AssistContext<'_>,
    adt: &hir::Adt,
    fn_name: &str,
) -> Option<(Option<ast::Impl>, FileId)> {
    let range = adt.source(ctx.sema.db)?.syntax().original_file_range(ctx.sema.db);
    let file = ctx.sema.parse(range.file_id);
    let adt_source =
        ctx.sema.find_node_at_offset_with_macros(file.syntax(), range.range.start())?;
    find_struct_impl(ctx, &adt_source, fn_name).map(|impl_| (impl_, range.file_id))
}

struct FunctionTemplate {
    leading_ws: String,
    fn_def: ast::Fn,
    ret_type: Option<ast::RetType>,
    should_focus_return_type: bool,
    trailing_ws: String,
    tail_expr: ast::Expr,
}

impl FunctionTemplate {
    fn to_string(&self, cap: Option<SnippetCap>) -> String {
        let f = match cap {
            Some(cap) => {
                let cursor = if self.should_focus_return_type {
                    // Focus the return type if there is one
                    match self.ret_type {
                        Some(ref ret_type) => ret_type.syntax(),
                        None => self.tail_expr.syntax(),
                    }
                } else {
                    self.tail_expr.syntax()
                };
                render_snippet(cap, self.fn_def.syntax(), Cursor::Replace(cursor))
            }
            None => self.fn_def.to_string(),
        };

        format!("{}{}{}", self.leading_ws, f, self.trailing_ws)
    }
}

struct FunctionBuilder {
    target: GeneratedFunctionTarget,
    fn_name: ast::Name,
    type_params: Option<ast::GenericParamList>,
    params: ast::ParamList,
    ret_type: Option<ast::RetType>,
    should_focus_return_type: bool,
    needs_pub: bool,
    is_async: bool,
}

impl FunctionBuilder {
    /// Prepares a generated function that matches `call`.
    /// The function is generated in `target_module` or next to `call`
    fn from_call(
        ctx: &AssistContext<'_>,
        call: &ast::CallExpr,
        fn_name: &str,
        target_module: Option<hir::Module>,
        target: GeneratedFunctionTarget,
    ) -> Option<Self> {
        let needs_pub = target_module.is_some();
        let target_module =
            target_module.or_else(|| ctx.sema.scope(target.syntax()).map(|it| it.module()))?;
        let fn_name = make::name(fn_name);
        let (type_params, params) =
            fn_args(ctx, target_module, ast::CallableExpr::Call(call.clone()))?;

        let await_expr = call.syntax().parent().and_then(ast::AwaitExpr::cast);
        let is_async = await_expr.is_some();

        let (ret_type, should_focus_return_type) =
            make_return_type(ctx, &ast::Expr::CallExpr(call.clone()), target_module);

        Some(Self {
            target,
            fn_name,
            type_params,
            params,
            ret_type,
            should_focus_return_type,
            needs_pub,
            is_async,
        })
    }

    fn from_method_call(
        ctx: &AssistContext<'_>,
        call: &ast::MethodCallExpr,
        name: &ast::NameRef,
        target_module: Module,
        target: GeneratedFunctionTarget,
    ) -> Option<Self> {
        let needs_pub =
            !module_is_descendant(&ctx.sema.scope(call.syntax())?.module(), &target_module, ctx);
        let fn_name = make::name(&name.text());
        let (type_params, params) =
            fn_args(ctx, target_module, ast::CallableExpr::MethodCall(call.clone()))?;

        let await_expr = call.syntax().parent().and_then(ast::AwaitExpr::cast);
        let is_async = await_expr.is_some();

        let (ret_type, should_focus_return_type) =
            make_return_type(ctx, &ast::Expr::MethodCallExpr(call.clone()), target_module);

        Some(Self {
            target,
            fn_name,
            type_params,
            params,
            ret_type,
            should_focus_return_type,
            needs_pub,
            is_async,
        })
    }

    fn render(self) -> FunctionTemplate {
        let placeholder_expr = make::ext::expr_todo();
        let fn_body = make::block_expr(vec![], Some(placeholder_expr));
        let visibility = if self.needs_pub { Some(make::visibility_pub_crate()) } else { None };
        let mut fn_def = make::fn_(
            visibility,
            self.fn_name,
            self.type_params,
            self.params,
            fn_body,
            self.ret_type,
            self.is_async,
        );
        let leading_ws;
        let trailing_ws;

        match self.target {
            GeneratedFunctionTarget::BehindItem(it) => {
                let indent = IndentLevel::from_node(&it);
                leading_ws = format!("\n\n{}", indent);
                fn_def = fn_def.indent(indent);
                trailing_ws = String::new();
            }
            GeneratedFunctionTarget::InEmptyItemList(it) => {
                let indent = IndentLevel::from_node(&it);
                leading_ws = format!("\n{}", indent + 1);
                fn_def = fn_def.indent(indent + 1);
                trailing_ws = format!("\n{}", indent);
            }
        };

        FunctionTemplate {
            leading_ws,
            ret_type: fn_def.ret_type(),
            // PANIC: we guarantee we always create a function body with a tail expr
            tail_expr: fn_def.body().unwrap().tail_expr().unwrap(),
            should_focus_return_type: self.should_focus_return_type,
            fn_def,
            trailing_ws,
        }
    }
}

/// Makes an optional return type along with whether the return type should be focused by the cursor.
/// If we cannot infer what the return type should be, we create a placeholder type.
///
/// The rule for whether we focus a return type or not (and thus focus the function body),
/// is rather simple:
/// * If we could *not* infer what the return type should be, focus it (so the user can fill-in
/// the correct return type).
/// * If we could infer the return type, don't focus it (and thus focus the function body) so the
/// user can change the `todo!` function body.
fn make_return_type(
    ctx: &AssistContext<'_>,
    call: &ast::Expr,
    target_module: Module,
) -> (Option<ast::RetType>, bool) {
    let (ret_ty, should_focus_return_type) = {
        match ctx.sema.type_of_expr(call).map(TypeInfo::original) {
            Some(ty) if ty.is_unknown() => (Some(make::ty_placeholder()), true),
            None => (Some(make::ty_placeholder()), true),
            Some(ty) if ty.is_unit() => (None, false),
            Some(ty) => {
                let rendered = ty.display_source_code(ctx.db(), target_module.into());
                match rendered {
                    Ok(rendered) => (Some(make::ty(&rendered)), false),
                    Err(_) => (Some(make::ty_placeholder()), true),
                }
            }
        }
    };
    let ret_type = ret_ty.map(make::ret_type);
    (ret_type, should_focus_return_type)
}

fn get_fn_target_info(
    ctx: &AssistContext<'_>,
    target_module: &Option<Module>,
    call: CallExpr,
) -> Option<TargetInfo> {
    let (target, file, insert_offset) = get_fn_target(ctx, target_module, call)?;
    Some(TargetInfo::new(*target_module, None, target, file, insert_offset))
}

fn get_fn_target(
    ctx: &AssistContext<'_>,
    target_module: &Option<Module>,
    call: CallExpr,
) -> Option<(GeneratedFunctionTarget, FileId, TextSize)> {
    let mut file = ctx.file_id();
    let target = match target_module {
        Some(target_module) => {
            let module_source = target_module.definition_source(ctx.db());
            let (in_file, target) = next_space_for_fn_in_module(ctx.sema.db, &module_source)?;
            file = in_file;
            target
        }
        None => next_space_for_fn_after_call_site(ast::CallableExpr::Call(call))?,
    };
    Some((target.clone(), file, get_insert_offset(&target)))
}

fn get_method_target(
    ctx: &AssistContext<'_>,
    target_module: &Module,
    impl_: &Option<ast::Impl>,
) -> Option<(GeneratedFunctionTarget, TextSize)> {
    let target = match impl_ {
        Some(impl_) => next_space_for_fn_in_impl(impl_)?,
        None => {
            next_space_for_fn_in_module(ctx.sema.db, &target_module.definition_source(ctx.sema.db))?
                .1
        }
    };
    Some((target.clone(), get_insert_offset(&target)))
}

fn assoc_fn_target_info(
    ctx: &AssistContext<'_>,
    call: &CallExpr,
    adt: hir::Adt,
    fn_name: &str,
) -> Option<TargetInfo> {
    let current_module = ctx.sema.scope(call.syntax())?.module();
    let module = adt.module(ctx.sema.db);
    let target_module = if current_module == module { None } else { Some(module) };
    if current_module.krate() != module.krate() {
        return None;
    }
    let (impl_, file) = get_adt_source(ctx, &adt, fn_name)?;
    let (target, insert_offset) = get_method_target(ctx, &module, &impl_)?;
    let adt_name = if impl_.is_none() { Some(adt.name(ctx.sema.db)) } else { None };
    Some(TargetInfo::new(target_module, adt_name, target, file, insert_offset))
}

fn get_insert_offset(target: &GeneratedFunctionTarget) -> TextSize {
    match &target {
        GeneratedFunctionTarget::BehindItem(it) => it.text_range().end(),
        GeneratedFunctionTarget::InEmptyItemList(it) => it.text_range().start() + TextSize::of('{'),
    }
}

#[derive(Clone)]
enum GeneratedFunctionTarget {
    BehindItem(SyntaxNode),
    InEmptyItemList(SyntaxNode),
}

impl GeneratedFunctionTarget {
    fn syntax(&self) -> &SyntaxNode {
        match self {
            GeneratedFunctionTarget::BehindItem(it) => it,
            GeneratedFunctionTarget::InEmptyItemList(it) => it,
        }
    }
}

/// Computes the type variables and arguments required for the generated function
fn fn_args(
    ctx: &AssistContext<'_>,
    target_module: hir::Module,
    call: ast::CallableExpr,
) -> Option<(Option<ast::GenericParamList>, ast::ParamList)> {
    let mut arg_names = Vec::new();
    let mut arg_types = Vec::new();
    for arg in call.arg_list()?.args() {
        arg_names.push(fn_arg_name(&ctx.sema, &arg));
        arg_types.push(fn_arg_type(ctx, target_module, &arg));
    }
    deduplicate_arg_names(&mut arg_names);
    let params = arg_names.into_iter().zip(arg_types).map(|(name, ty)| {
        make::param(make::ext::simple_ident_pat(make::name(&name)).into(), make::ty(&ty))
    });

    Some((
        None,
        make::param_list(
            match call {
                ast::CallableExpr::Call(_) => None,
                ast::CallableExpr::MethodCall(_) => Some(make::self_param()),
            },
            params,
        ),
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
    let mut arg_name_counts = FxHashMap::default();
    for name in arg_names.iter() {
        *arg_name_counts.entry(name).or_insert(0) += 1;
    }
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

fn fn_arg_name(sema: &Semantics<'_, RootDatabase>, arg_expr: &ast::Expr) -> String {
    let name = (|| match arg_expr {
        ast::Expr::CastExpr(cast_expr) => Some(fn_arg_name(sema, &cast_expr.expr()?)),
        expr => {
            let name_ref = expr
                .syntax()
                .descendants()
                .filter_map(ast::NameRef::cast)
                .filter(|name| name.ident_token().is_some())
                .last()?;
            if let Some(NameRefClass::Definition(Definition::Const(_) | Definition::Static(_))) =
                NameRefClass::classify(sema, &name_ref)
            {
                return Some(name_ref.to_string().to_lowercase());
            };
            Some(to_lower_snake_case(&name_ref.to_string()))
        }
    })();
    match name {
        Some(mut name) if name.starts_with(|c: char| c.is_ascii_digit()) => {
            name.insert_str(0, "arg");
            name
        }
        Some(name) => name,
        None => "arg".to_string(),
    }
}

fn fn_arg_type(ctx: &AssistContext<'_>, target_module: hir::Module, fn_arg: &ast::Expr) -> String {
    fn maybe_displayed_type(
        ctx: &AssistContext<'_>,
        target_module: hir::Module,
        fn_arg: &ast::Expr,
    ) -> Option<String> {
        let ty = ctx.sema.type_of_expr(fn_arg)?.adjusted();
        if ty.is_unknown() {
            return None;
        }

        if ty.is_reference() || ty.is_mutable_reference() {
            let famous_defs = &FamousDefs(&ctx.sema, ctx.sema.scope(fn_arg.syntax())?.krate());
            convert_reference_type(ty.strip_references(), ctx.db(), famous_defs)
                .map(|conversion| conversion.convert_type(ctx.db()))
                .or_else(|| ty.display_source_code(ctx.db(), target_module.into()).ok())
        } else {
            ty.display_source_code(ctx.db(), target_module.into()).ok()
        }
    }

    maybe_displayed_type(ctx, target_module, fn_arg).unwrap_or_else(|| String::from("_"))
}

/// Returns the position inside the current mod or file
/// directly after the current block
/// We want to write the generated function directly after
/// fns, impls or macro calls, but inside mods
fn next_space_for_fn_after_call_site(expr: ast::CallableExpr) -> Option<GeneratedFunctionTarget> {
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
    module_source: &hir::InFile<hir::ModuleSource>,
) -> Option<(FileId, GeneratedFunctionTarget)> {
    let file = module_source.file_id.original_file(db);
    let assist_item = match &module_source.value {
        hir::ModuleSource::SourceFile(it) => match it.items().last() {
            Some(last_item) => GeneratedFunctionTarget::BehindItem(last_item.syntax().clone()),
            None => GeneratedFunctionTarget::BehindItem(it.syntax().clone()),
        },
        hir::ModuleSource::Module(it) => match it.item_list().and_then(|it| it.items().last()) {
            Some(last_item) => GeneratedFunctionTarget::BehindItem(last_item.syntax().clone()),
            None => GeneratedFunctionTarget::InEmptyItemList(it.item_list()?.syntax().clone()),
        },
        hir::ModuleSource::BlockExpr(it) => {
            if let Some(last_item) =
                it.statements().take_while(|stmt| matches!(stmt, ast::Stmt::Item(_))).last()
            {
                GeneratedFunctionTarget::BehindItem(last_item.syntax().clone())
            } else {
                GeneratedFunctionTarget::InEmptyItemList(it.syntax().clone())
            }
        }
    };
    Some((file, assist_item))
}

fn next_space_for_fn_in_impl(impl_: &ast::Impl) -> Option<GeneratedFunctionTarget> {
    if let Some(last_item) = impl_.assoc_item_list().and_then(|it| it.assoc_items().last()) {
        Some(GeneratedFunctionTarget::BehindItem(last_item.syntax().clone()))
    } else {
        Some(GeneratedFunctionTarget::InEmptyItemList(impl_.assoc_item_list()?.syntax().clone()))
    }
}

fn module_is_descendant(module: &hir::Module, ans: &hir::Module, ctx: &AssistContext<'_>) -> bool {
    if module == ans {
        return true;
    }
    for c in ans.children(ctx.sema.db) {
        if module_is_descendant(module, &c, ctx) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn add_function_with_no_args() {
        check_assist(
            generate_function,
            r"
fn foo() {
    bar$0();
}
",
            r"
fn foo() {
    bar();
}

fn bar() ${0:-> _} {
    todo!()
}
",
        )
    }

    #[test]
    fn add_function_from_method() {
        // This ensures that the function is correctly generated
        // in the next outer mod or file
        check_assist(
            generate_function,
            r"
impl Foo {
    fn foo() {
        bar$0();
    }
}
",
            r"
impl Foo {
    fn foo() {
        bar();
    }
}

fn bar() ${0:-> _} {
    todo!()
}
",
        )
    }

    #[test]
    fn add_function_directly_after_current_block() {
        // The new fn should not be created at the end of the file or module
        check_assist(
            generate_function,
            r"
fn foo1() {
    bar$0();
}

fn foo2() {}
",
            r"
fn foo1() {
    bar();
}

fn bar() ${0:-> _} {
    todo!()
}

fn foo2() {}
",
        )
    }

    #[test]
    fn add_function_with_no_args_in_same_module() {
        check_assist(
            generate_function,
            r"
mod baz {
    fn foo() {
        bar$0();
    }
}
",
            r"
mod baz {
    fn foo() {
        bar();
    }

    fn bar() ${0:-> _} {
        todo!()
    }
}
",
        )
    }

    #[test]
    fn add_function_with_upper_camel_case_arg() {
        check_assist(
            generate_function,
            r"
struct BazBaz;
fn foo() {
    bar$0(BazBaz);
}
",
            r"
struct BazBaz;
fn foo() {
    bar(BazBaz);
}

fn bar(baz_baz: BazBaz) ${0:-> _} {
    todo!()
}
",
        );
    }

    #[test]
    fn add_function_with_upper_camel_case_arg_as_cast() {
        check_assist(
            generate_function,
            r"
struct BazBaz;
fn foo() {
    bar$0(&BazBaz as *const BazBaz);
}
",
            r"
struct BazBaz;
fn foo() {
    bar(&BazBaz as *const BazBaz);
}

fn bar(baz_baz: *const BazBaz) ${0:-> _} {
    todo!()
}
",
        );
    }

    #[test]
    fn add_function_with_function_call_arg() {
        check_assist(
            generate_function,
            r"
struct Baz;
fn baz() -> Baz { todo!() }
fn foo() {
    bar$0(baz());
}
",
            r"
struct Baz;
fn baz() -> Baz { todo!() }
fn foo() {
    bar(baz());
}

fn bar(baz: Baz) ${0:-> _} {
    todo!()
}
",
        );
    }

    #[test]
    fn add_function_with_method_call_arg() {
        check_assist(
            generate_function,
            r"
struct Baz;
impl Baz {
    fn foo(&self) -> Baz {
        ba$0r(self.baz())
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

fn bar(baz: Baz) -> Baz {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn add_function_with_string_literal_arg() {
        check_assist(
            generate_function,
            r#"
fn foo() {
    $0bar("bar")
}
"#,
            r#"
fn foo() {
    bar("bar")
}

fn bar(arg: &str) {
    ${0:todo!()}
}
"#,
        )
    }

    #[test]
    fn add_function_with_char_literal_arg() {
        check_assist(
            generate_function,
            r#"
fn foo() {
    $0bar('x')
}
"#,
            r#"
fn foo() {
    bar('x')
}

fn bar(arg: char) {
    ${0:todo!()}
}
"#,
        )
    }

    #[test]
    fn add_function_with_int_literal_arg() {
        check_assist(
            generate_function,
            r"
fn foo() {
    $0bar(42)
}
",
            r"
fn foo() {
    bar(42)
}

fn bar(arg: i32) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn add_function_with_cast_int_literal_arg() {
        check_assist(
            generate_function,
            r"
fn foo() {
    $0bar(42 as u8)
}
",
            r"
fn foo() {
    bar(42 as u8)
}

fn bar(arg: u8) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn name_of_cast_variable_is_used() {
        // Ensures that the name of the cast type isn't used
        // in the generated function signature.
        check_assist(
            generate_function,
            r"
fn foo() {
    let x = 42;
    bar$0(x as u8)
}
",
            r"
fn foo() {
    let x = 42;
    bar(x as u8)
}

fn bar(x: u8) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn add_function_with_variable_arg() {
        check_assist(
            generate_function,
            r"
fn foo() {
    let worble = ();
    $0bar(worble)
}
",
            r"
fn foo() {
    let worble = ();
    bar(worble)
}

fn bar(worble: ()) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn add_function_with_impl_trait_arg() {
        check_assist(
            generate_function,
            r#"
//- minicore: sized
trait Foo {}
fn foo() -> impl Foo {
    todo!()
}
fn baz() {
    $0bar(foo())
}
"#,
            r#"
trait Foo {}
fn foo() -> impl Foo {
    todo!()
}
fn baz() {
    bar(foo())
}

fn bar(foo: impl Foo) {
    ${0:todo!()}
}
"#,
        )
    }

    #[test]
    fn borrowed_arg() {
        check_assist(
            generate_function,
            r"
struct Baz;
fn baz() -> Baz { todo!() }

fn foo() {
    bar$0(&baz())
}
",
            r"
struct Baz;
fn baz() -> Baz { todo!() }

fn foo() {
    bar(&baz())
}

fn bar(baz: &Baz) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn add_function_with_qualified_path_arg() {
        check_assist(
            generate_function,
            r"
mod Baz {
    pub struct Bof;
    pub fn baz() -> Bof { Bof }
}
fn foo() {
    $0bar(Baz::baz())
}
",
            r"
mod Baz {
    pub struct Bof;
    pub fn baz() -> Bof { Bof }
}
fn foo() {
    bar(Baz::baz())
}

fn bar(baz: Baz::Bof) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn add_function_with_generic_arg() {
        // FIXME: This is wrong, generated `bar` should include generic parameter.
        check_assist(
            generate_function,
            r"
fn foo<T>(t: T) {
    $0bar(t)
}
",
            r"
fn foo<T>(t: T) {
    bar(t)
}

fn bar(t: T) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn add_function_with_fn_arg() {
        // FIXME: The argument in `bar` is wrong.
        check_assist(
            generate_function,
            r"
struct Baz;
impl Baz {
    fn new() -> Self { Baz }
}
fn foo() {
    $0bar(Baz::new);
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

fn bar(new: fn) ${0:-> _} {
    todo!()
}
",
        )
    }

    #[test]
    fn add_function_with_closure_arg() {
        // FIXME: The argument in `bar` is wrong.
        check_assist(
            generate_function,
            r"
fn foo() {
    let closure = |x: i64| x - 1;
    $0bar(closure)
}
",
            r"
fn foo() {
    let closure = |x: i64| x - 1;
    bar(closure)
}

fn bar(closure: _) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn unresolveable_types_default_to_placeholder() {
        check_assist(
            generate_function,
            r"
fn foo() {
    $0bar(baz)
}
",
            r"
fn foo() {
    bar(baz)
}

fn bar(baz: _) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn arg_names_dont_overlap() {
        check_assist(
            generate_function,
            r"
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    $0bar(baz(), baz())
}
",
            r"
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    bar(baz(), baz())
}

fn bar(baz_1: Baz, baz_2: Baz) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn arg_name_counters_start_at_1_per_name() {
        check_assist(
            generate_function,
            r#"
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    $0bar(baz(), baz(), "foo", "bar")
}
"#,
            r#"
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    bar(baz(), baz(), "foo", "bar")
}

fn bar(baz_1: Baz, baz_2: Baz, arg_1: &str, arg_2: &str) {
    ${0:todo!()}
}
"#,
        )
    }

    #[test]
    fn add_function_in_module() {
        check_assist(
            generate_function,
            r"
mod bar {}

fn foo() {
    bar::my_fn$0()
}
",
            r"
mod bar {
    pub(crate) fn my_fn() {
        ${0:todo!()}
    }
}

fn foo() {
    bar::my_fn()
}
",
        )
    }

    #[test]
    fn qualified_path_uses_correct_scope() {
        check_assist(
            generate_function,
            r#"
mod foo {
    pub struct Foo;
}
fn bar() {
    use foo::Foo;
    let foo = Foo;
    baz$0(foo)
}
"#,
            r#"
mod foo {
    pub struct Foo;
}
fn bar() {
    use foo::Foo;
    let foo = Foo;
    baz(foo)
}

fn baz(foo: foo::Foo) {
    ${0:todo!()}
}
"#,
        )
    }

    #[test]
    fn add_function_in_module_containing_other_items() {
        check_assist(
            generate_function,
            r"
mod bar {
    fn something_else() {}
}

fn foo() {
    bar::my_fn$0()
}
",
            r"
mod bar {
    fn something_else() {}

    pub(crate) fn my_fn() {
        ${0:todo!()}
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
            generate_function,
            r"
mod bar {
    mod baz {}
}

fn foo() {
    bar::baz::my_fn$0()
}
",
            r"
mod bar {
    mod baz {
        pub(crate) fn my_fn() {
            ${0:todo!()}
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
            generate_function,
            r"
//- /main.rs
mod foo;

fn main() {
    foo::bar$0()
}
//- /foo.rs
",
            r"


pub(crate) fn bar() {
    ${0:todo!()}
}",
        )
    }

    #[test]
    fn add_function_with_return_type() {
        check_assist(
            generate_function,
            r"
fn main() {
    let x: u32 = foo$0();
}
",
            r"
fn main() {
    let x: u32 = foo();
}

fn foo() -> u32 {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn add_function_not_applicable_if_function_already_exists() {
        check_assist_not_applicable(
            generate_function,
            r"
fn foo() {
    bar$0();
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
            generate_function,
            r#"
fn foo() {
    bar(b$0az);
}

fn bar(baz: ()) {}
"#,
        )
    }

    #[test]
    fn create_method_with_no_args() {
        check_assist(
            generate_function,
            r#"
struct Foo;
impl Foo {
    fn foo(&self) {
        self.bar()$0;
    }
}
"#,
            r#"
struct Foo;
impl Foo {
    fn foo(&self) {
        self.bar();
    }

    fn bar(&self) ${0:-> _} {
        todo!()
    }
}
"#,
        )
    }

    #[test]
    fn create_function_with_async() {
        check_assist(
            generate_function,
            r"
fn foo() {
    $0bar(42).await();
}
",
            r"
fn foo() {
    bar(42).await();
}

async fn bar(arg: i32) ${0:-> _} {
    todo!()
}
",
        )
    }

    #[test]
    fn create_method() {
        check_assist(
            generate_function,
            r"
struct S;
fn foo() {S.bar$0();}
",
            r"
struct S;
fn foo() {S.bar();}
impl S {


fn bar(&self) ${0:-> _} {
    todo!()
}
}
",
        )
    }

    #[test]
    fn create_method_within_an_impl() {
        check_assist(
            generate_function,
            r"
struct S;
fn foo() {S.bar$0();}
impl S {}

",
            r"
struct S;
fn foo() {S.bar();}
impl S {
    fn bar(&self) ${0:-> _} {
        todo!()
    }
}

",
        )
    }

    #[test]
    fn create_method_from_different_module() {
        check_assist(
            generate_function,
            r"
mod s {
    pub struct S;
}
fn foo() {s::S.bar$0();}
",
            r"
mod s {
    pub struct S;
impl S {


    pub(crate) fn bar(&self) ${0:-> _} {
        todo!()
    }
}
}
fn foo() {s::S.bar();}
",
        )
    }

    #[test]
    fn create_method_from_descendant_module() {
        check_assist(
            generate_function,
            r"
struct S;
mod s {
    fn foo() {
        super::S.bar$0();
    }
}

",
            r"
struct S;
mod s {
    fn foo() {
        super::S.bar();
    }
}
impl S {


fn bar(&self) ${0:-> _} {
    todo!()
}
}

",
        )
    }

    #[test]
    fn create_method_with_cursor_anywhere_on_call_expresion() {
        check_assist(
            generate_function,
            r"
struct S;
fn foo() {$0S.bar();}
",
            r"
struct S;
fn foo() {S.bar();}
impl S {


fn bar(&self) ${0:-> _} {
    todo!()
}
}
",
        )
    }

    #[test]
    fn create_static_method() {
        check_assist(
            generate_function,
            r"
struct S;
fn foo() {S::bar$0();}
",
            r"
struct S;
fn foo() {S::bar();}
impl S {


fn bar() ${0:-> _} {
    todo!()
}
}
",
        )
    }

    #[test]
    fn create_static_method_within_an_impl() {
        check_assist(
            generate_function,
            r"
struct S;
fn foo() {S::bar$0();}
impl S {}

",
            r"
struct S;
fn foo() {S::bar();}
impl S {
    fn bar() ${0:-> _} {
        todo!()
    }
}

",
        )
    }

    #[test]
    fn create_static_method_from_different_module() {
        check_assist(
            generate_function,
            r"
mod s {
    pub struct S;
}
fn foo() {s::S::bar$0();}
",
            r"
mod s {
    pub struct S;
impl S {


    pub(crate) fn bar() ${0:-> _} {
        todo!()
    }
}
}
fn foo() {s::S::bar();}
",
        )
    }

    #[test]
    fn create_static_method_with_cursor_anywhere_on_call_expresion() {
        check_assist(
            generate_function,
            r"
struct S;
fn foo() {$0S::bar();}
",
            r"
struct S;
fn foo() {S::bar();}
impl S {


fn bar() ${0:-> _} {
    todo!()
}
}
",
        )
    }

    #[test]
    fn create_static_method_within_an_impl_with_self_syntax() {
        check_assist(
            generate_function,
            r"
struct S;
impl S {
    fn foo(&self) {
        Self::bar$0();
    }
}
",
            r"
struct S;
impl S {
    fn foo(&self) {
        Self::bar();
    }

    fn bar() ${0:-> _} {
        todo!()
    }
}
",
        )
    }

    #[test]
    fn no_panic_on_invalid_global_path() {
        check_assist(
            generate_function,
            r"
fn main() {
    ::foo$0();
}
",
            r"
fn main() {
    ::foo();
}

fn foo() ${0:-> _} {
    todo!()
}
",
        )
    }

    #[test]
    fn handle_tuple_indexing() {
        check_assist(
            generate_function,
            r"
fn main() {
    let a = ((),);
    foo$0(a.0);
}
",
            r"
fn main() {
    let a = ((),);
    foo(a.0);
}

fn foo(a: ()) ${0:-> _} {
    todo!()
}
",
        )
    }

    #[test]
    fn add_function_with_const_arg() {
        check_assist(
            generate_function,
            r"
const VALUE: usize = 0;
fn main() {
    foo$0(VALUE);
}
",
            r"
const VALUE: usize = 0;
fn main() {
    foo(VALUE);
}

fn foo(value: usize) ${0:-> _} {
    todo!()
}
",
        )
    }

    #[test]
    fn add_function_with_static_arg() {
        check_assist(
            generate_function,
            r"
static VALUE: usize = 0;
fn main() {
    foo$0(VALUE);
}
",
            r"
static VALUE: usize = 0;
fn main() {
    foo(VALUE);
}

fn foo(value: usize) ${0:-> _} {
    todo!()
}
",
        )
    }

    #[test]
    fn add_function_with_static_mut_arg() {
        check_assist(
            generate_function,
            r"
static mut VALUE: usize = 0;
fn main() {
    foo$0(VALUE);
}
",
            r"
static mut VALUE: usize = 0;
fn main() {
    foo(VALUE);
}

fn foo(value: usize) ${0:-> _} {
    todo!()
}
",
        )
    }

    #[test]
    fn not_applicable_for_enum_variant() {
        check_assist_not_applicable(
            generate_function,
            r"
enum Foo {}
fn main() {
    Foo::Bar$0(true)
}
",
        );
    }

    #[test]
    fn applicable_for_enum_method() {
        check_assist(
            generate_function,
            r"
enum Foo {}
fn main() {
    Foo::new$0();
}
",
            r"
enum Foo {}
fn main() {
    Foo::new();
}
impl Foo {


fn new() ${0:-> _} {
    todo!()
}
}
",
        )
    }
}
