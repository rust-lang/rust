use hir::{
    Adt, AsAssocItem, HasSource, HirDisplay, Module, PathResolution, Semantics, Type, TypeInfo,
};
use ide_db::{
    base_db::FileId,
    defs::{Definition, NameRefClass},
    famous_defs::FamousDefs,
    helpers::is_editable_crate,
    path_transform::PathTransform,
    FxHashMap, FxHashSet, RootDatabase, SnippetCap,
};
use stdx::to_lower_snake_case;
use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make, AstNode, CallExpr, HasArgList, HasGenericParams, HasModuleItem, HasTypeBounds,
    },
    SyntaxKind, SyntaxNode, TextRange, TextSize,
};

use crate::{
    utils::{convert_reference_type, find_struct_impl, render_snippet, Cursor},
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

    if let Some(m) = target_module {
        if !is_editable_crate(m.krate(), ctx.db()) {
            return None;
        }
    }

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
                get_fn_target_info(ctx, Some(module), call.clone())
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
        _ => get_fn_target_info(ctx, None, call.clone()),
    }
}

fn gen_method(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let call: ast::MethodCallExpr = ctx.find_node_at_offset()?;
    if ctx.sema.resolve_method_call(&call).is_some() {
        return None;
    }

    let fn_name = call.name_ref()?;
    let receiver_ty = ctx.sema.type_of_expr(&call.receiver()?)?.original().strip_references();
    let adt = receiver_ty.as_adt()?;

    let target_module = adt.module(ctx.sema.db);
    if !is_editable_crate(target_module.krate(), ctx.db()) {
        return None;
    }

    let (impl_, file) = get_adt_source(ctx, &adt, fn_name.text().as_str())?;
    let (target, insert_offset) = get_method_target(ctx, &impl_, &adt)?;

    let function_builder = FunctionBuilder::from_method_call(
        ctx,
        &call,
        &fn_name,
        receiver_ty,
        target_module,
        target,
    )?;
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
        let indent = IndentLevel::from_node(function_builder.target.syntax());
        let function_template = function_builder.render(adt_name.is_some());
        let mut func = function_template.to_string(ctx.config.snippet_cap);
        if let Some(name) = adt_name {
            // FIXME: adt may have generic params.
            func = format!("\n{indent}impl {} {{\n{func}\n{indent}}}", name.display(ctx.db()));
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
    find_struct_impl(ctx, &adt_source, &[fn_name.to_string()]).map(|impl_| (impl_, range.file_id))
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
        let Self { leading_ws, fn_def, ret_type, should_focus_return_type, trailing_ws, tail_expr } =
            self;

        let f = match cap {
            Some(cap) => {
                let cursor = if *should_focus_return_type {
                    // Focus the return type if there is one
                    match ret_type {
                        Some(ret_type) => ret_type.syntax(),
                        None => tail_expr.syntax(),
                    }
                } else {
                    tail_expr.syntax()
                };
                render_snippet(cap, fn_def.syntax(), Cursor::Replace(cursor))
            }
            None => fn_def.to_string(),
        };

        format!("{leading_ws}{f}{trailing_ws}")
    }
}

struct FunctionBuilder {
    target: GeneratedFunctionTarget,
    fn_name: ast::Name,
    generic_param_list: Option<ast::GenericParamList>,
    where_clause: Option<ast::WhereClause>,
    params: ast::ParamList,
    ret_type: Option<ast::RetType>,
    should_focus_return_type: bool,
    visibility: Visibility,
    is_async: bool,
}

impl FunctionBuilder {
    /// Prepares a generated function that matches `call`.
    /// The function is generated in `target_module` or next to `call`
    fn from_call(
        ctx: &AssistContext<'_>,
        call: &ast::CallExpr,
        fn_name: &str,
        target_module: Option<Module>,
        target: GeneratedFunctionTarget,
    ) -> Option<Self> {
        let target_module =
            target_module.or_else(|| ctx.sema.scope(target.syntax()).map(|it| it.module()))?;

        let current_module = ctx.sema.scope(call.syntax())?.module();
        let visibility = calculate_necessary_visibility(current_module, target_module, ctx);
        let fn_name = make::name(fn_name);
        let mut necessary_generic_params = FxHashSet::default();
        let params = fn_args(
            ctx,
            target_module,
            ast::CallableExpr::Call(call.clone()),
            &mut necessary_generic_params,
        )?;

        let await_expr = call.syntax().parent().and_then(ast::AwaitExpr::cast);
        let is_async = await_expr.is_some();

        let expr_for_ret_ty = await_expr.map_or_else(|| call.clone().into(), |it| it.into());
        let (ret_type, should_focus_return_type) =
            make_return_type(ctx, &expr_for_ret_ty, target_module, &mut necessary_generic_params);

        let (generic_param_list, where_clause) =
            fn_generic_params(ctx, necessary_generic_params, &target)?;

        Some(Self {
            target,
            fn_name,
            generic_param_list,
            where_clause,
            params,
            ret_type,
            should_focus_return_type,
            visibility,
            is_async,
        })
    }

    fn from_method_call(
        ctx: &AssistContext<'_>,
        call: &ast::MethodCallExpr,
        name: &ast::NameRef,
        receiver_ty: Type,
        target_module: Module,
        target: GeneratedFunctionTarget,
    ) -> Option<Self> {
        let current_module = ctx.sema.scope(call.syntax())?.module();
        let visibility = calculate_necessary_visibility(current_module, target_module, ctx);

        let fn_name = make::name(&name.text());
        let mut necessary_generic_params = FxHashSet::default();
        necessary_generic_params.extend(receiver_ty.generic_params(ctx.db()));
        let params = fn_args(
            ctx,
            target_module,
            ast::CallableExpr::MethodCall(call.clone()),
            &mut necessary_generic_params,
        )?;

        let await_expr = call.syntax().parent().and_then(ast::AwaitExpr::cast);
        let is_async = await_expr.is_some();

        let expr_for_ret_ty = await_expr.map_or_else(|| call.clone().into(), |it| it.into());
        let (ret_type, should_focus_return_type) =
            make_return_type(ctx, &expr_for_ret_ty, target_module, &mut necessary_generic_params);

        let (generic_param_list, where_clause) =
            fn_generic_params(ctx, necessary_generic_params, &target)?;

        Some(Self {
            target,
            fn_name,
            generic_param_list,
            where_clause,
            params,
            ret_type,
            should_focus_return_type,
            visibility,
            is_async,
        })
    }

    fn render(self, is_method: bool) -> FunctionTemplate {
        let placeholder_expr = make::ext::expr_todo();
        let fn_body = make::block_expr(vec![], Some(placeholder_expr));
        let visibility = match self.visibility {
            Visibility::None => None,
            Visibility::Crate => Some(make::visibility_pub_crate()),
            Visibility::Pub => Some(make::visibility_pub()),
        };
        let mut fn_def = make::fn_(
            visibility,
            self.fn_name,
            self.generic_param_list,
            self.where_clause,
            self.params,
            fn_body,
            self.ret_type,
            self.is_async,
            false, // FIXME : const and unsafe are not handled yet.
            false,
        );
        let leading_ws;
        let trailing_ws;

        match self.target {
            GeneratedFunctionTarget::BehindItem(it) => {
                let mut indent = IndentLevel::from_node(&it);
                if is_method {
                    indent = indent + 1;
                    leading_ws = format!("{indent}");
                } else {
                    leading_ws = format!("\n\n{indent}");
                }

                fn_def = fn_def.indent(indent);
                trailing_ws = String::new();
            }
            GeneratedFunctionTarget::InEmptyItemList(it) => {
                let indent = IndentLevel::from_node(&it);
                let leading_indent = indent + 1;
                leading_ws = format!("\n{leading_indent}");
                fn_def = fn_def.indent(leading_indent);
                trailing_ws = format!("\n{indent}");
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
    expr: &ast::Expr,
    target_module: Module,
    necessary_generic_params: &mut FxHashSet<hir::GenericParam>,
) -> (Option<ast::RetType>, bool) {
    let (ret_ty, should_focus_return_type) = {
        match ctx.sema.type_of_expr(expr).map(TypeInfo::original) {
            Some(ty) if ty.is_unknown() => (Some(make::ty_placeholder()), true),
            None => (Some(make::ty_placeholder()), true),
            Some(ty) if ty.is_unit() => (None, false),
            Some(ty) => {
                necessary_generic_params.extend(ty.generic_params(ctx.db()));
                let rendered = ty.display_source_code(ctx.db(), target_module.into(), true);
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
    target_module: Option<Module>,
    call: CallExpr,
) -> Option<TargetInfo> {
    let (target, file, insert_offset) = get_fn_target(ctx, target_module, call)?;
    Some(TargetInfo::new(target_module, None, target, file, insert_offset))
}

fn get_fn_target(
    ctx: &AssistContext<'_>,
    target_module: Option<Module>,
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
    impl_: &Option<ast::Impl>,
    adt: &Adt,
) -> Option<(GeneratedFunctionTarget, TextSize)> {
    let target = match impl_ {
        Some(impl_) => next_space_for_fn_in_impl(impl_)?,
        None => {
            GeneratedFunctionTarget::BehindItem(adt.source(ctx.sema.db)?.syntax().value.clone())
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
    let (target, insert_offset) = get_method_target(ctx, &impl_, &adt)?;
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

    fn parent(&self) -> SyntaxNode {
        match self {
            GeneratedFunctionTarget::BehindItem(it) => it.parent().expect("item without parent"),
            GeneratedFunctionTarget::InEmptyItemList(it) => it.clone(),
        }
    }
}

/// Computes parameter list for the generated function.
fn fn_args(
    ctx: &AssistContext<'_>,
    target_module: Module,
    call: ast::CallableExpr,
    necessary_generic_params: &mut FxHashSet<hir::GenericParam>,
) -> Option<ast::ParamList> {
    let mut arg_names = Vec::new();
    let mut arg_types = Vec::new();
    for arg in call.arg_list()?.args() {
        arg_names.push(fn_arg_name(&ctx.sema, &arg));
        arg_types.push(fn_arg_type(ctx, target_module, &arg, necessary_generic_params));
    }
    deduplicate_arg_names(&mut arg_names);
    let params = arg_names.into_iter().zip(arg_types).map(|(name, ty)| {
        make::param(make::ext::simple_ident_pat(make::name(&name)).into(), make::ty(&ty))
    });

    Some(make::param_list(
        match call {
            ast::CallableExpr::Call(_) => None,
            ast::CallableExpr::MethodCall(_) => Some(make::self_param()),
        },
        params,
    ))
}

/// Gets parameter bounds and where predicates in scope and filters out irrelevant ones. Returns
/// `None` when it fails to get scope information.
///
/// See comment on `filter_unnecessary_bounds()` for what bounds we consider relevant.
///
/// NOTE: Generic parameters returned from this function may cause name clash at `target`. We don't
/// currently do anything about it because it's actually easy to resolve it after the assist: just
/// use the Rename functionality.
fn fn_generic_params(
    ctx: &AssistContext<'_>,
    necessary_params: FxHashSet<hir::GenericParam>,
    target: &GeneratedFunctionTarget,
) -> Option<(Option<ast::GenericParamList>, Option<ast::WhereClause>)> {
    if necessary_params.is_empty() {
        // Not really needed but fast path.
        return Some((None, None));
    }

    // 1. Get generic parameters (with bounds) and where predicates in scope.
    let (generic_params, where_preds) = params_and_where_preds_in_scope(ctx);

    // 2. Extract type parameters included in each bound.
    let mut generic_params = generic_params
        .into_iter()
        .filter_map(|it| compute_contained_params_in_generic_param(ctx, it))
        .collect();
    let mut where_preds = where_preds
        .into_iter()
        .filter_map(|it| compute_contained_params_in_where_pred(ctx, it))
        .collect();

    // 3. Filter out unnecessary bounds.
    filter_unnecessary_bounds(&mut generic_params, &mut where_preds, necessary_params);
    filter_bounds_in_scope(&mut generic_params, &mut where_preds, ctx, target);

    let generic_params: Vec<_> =
        generic_params.into_iter().map(|it| it.node.clone_for_update()).collect();
    let where_preds: Vec<_> =
        where_preds.into_iter().map(|it| it.node.clone_for_update()).collect();

    // 4. Rewrite paths
    if let Some(param) = generic_params.first() {
        let source_scope = ctx.sema.scope(param.syntax())?;
        let target_scope = ctx.sema.scope(&target.parent())?;
        if source_scope.module() != target_scope.module() {
            let transform = PathTransform::generic_transformation(&target_scope, &source_scope);
            let generic_params = generic_params.iter().map(|it| it.syntax());
            let where_preds = where_preds.iter().map(|it| it.syntax());
            transform.apply_all(generic_params.chain(where_preds));
        }
    }

    let generic_param_list = make::generic_param_list(generic_params);
    let where_clause =
        if where_preds.is_empty() { None } else { Some(make::where_clause(where_preds)) };

    Some((Some(generic_param_list), where_clause))
}

fn params_and_where_preds_in_scope(
    ctx: &AssistContext<'_>,
) -> (Vec<ast::GenericParam>, Vec<ast::WherePred>) {
    let Some(body) = containing_body(ctx) else { return Default::default(); };

    let mut generic_params = Vec::new();
    let mut where_clauses = Vec::new();

    // There are two items where generic parameters currently in scope may be declared: the item
    // the cursor is at, and its parent (if any).
    //
    // We handle parent first so that their generic parameters appear first in the generic
    // parameter list of the function we're generating.
    let db = ctx.db();
    if let Some(parent) = body.as_assoc_item(db).map(|it| it.container(db)) {
        match parent {
            hir::AssocItemContainer::Impl(it) => {
                let (params, clauses) = get_bounds_in_scope(ctx, it);
                generic_params.extend(params);
                where_clauses.extend(clauses);
            }
            hir::AssocItemContainer::Trait(it) => {
                let (params, clauses) = get_bounds_in_scope(ctx, it);
                generic_params.extend(params);
                where_clauses.extend(clauses);
            }
        }
    }

    // Other defs with body may inherit generic parameters from its parent, but never have their
    // own generic parameters.
    if let hir::DefWithBody::Function(it) = body {
        let (params, clauses) = get_bounds_in_scope(ctx, it);
        generic_params.extend(params);
        where_clauses.extend(clauses);
    }

    (generic_params, where_clauses)
}

fn containing_body(ctx: &AssistContext<'_>) -> Option<hir::DefWithBody> {
    let item: ast::Item = ctx.find_node_at_offset()?;
    let def = match item {
        ast::Item::Fn(it) => ctx.sema.to_def(&it)?.into(),
        ast::Item::Const(it) => ctx.sema.to_def(&it)?.into(),
        ast::Item::Static(it) => ctx.sema.to_def(&it)?.into(),
        _ => return None,
    };
    Some(def)
}

fn get_bounds_in_scope<D>(
    ctx: &AssistContext<'_>,
    def: D,
) -> (impl Iterator<Item = ast::GenericParam>, impl Iterator<Item = ast::WherePred>)
where
    D: HasSource,
    D::Ast: HasGenericParams,
{
    // This function should be only called with `Impl`, `Trait`, or `Function`, for which it's
    // infallible to get source ast.
    let node = ctx.sema.source(def).unwrap().value;
    let generic_params = node.generic_param_list().into_iter().flat_map(|it| it.generic_params());
    let where_clauses = node.where_clause().into_iter().flat_map(|it| it.predicates());
    (generic_params, where_clauses)
}

#[derive(Debug)]
struct ParamBoundWithParams {
    node: ast::GenericParam,
    /// Generic parameter `node` introduces.
    ///
    /// ```text
    /// impl<T> S<T> {
    ///     fn f<U: Trait<T>>() {}
    ///          ^ this
    /// }
    /// ```
    ///
    /// `U` in this example.
    self_ty_param: hir::GenericParam,
    /// Generic parameters contained in the trait reference of this bound.
    ///
    /// ```text
    /// impl<T> S<T> {
    ///     fn f<U: Trait<T>>() {}
    ///             ^^^^^^^^ params in this part
    /// }
    /// ```
    ///
    /// `T` in this example.
    other_params: FxHashSet<hir::GenericParam>,
}

#[derive(Debug)]
struct WherePredWithParams {
    node: ast::WherePred,
    /// Generic parameters contained in the "self type" of this where predicate.
    ///
    /// ```text
    /// Struct<T, U>: Trait<T, Assoc = V>,
    /// ^^^^^^^^^^^^ params in this part
    /// ```
    ///
    /// `T` and `U` in this example.
    self_ty_params: FxHashSet<hir::GenericParam>,
    /// Generic parameters contained in the trait reference of this where predicate.
    ///
    /// ```text
    /// Struct<T, U>: Trait<T, Assoc = V>,
    ///               ^^^^^^^^^^^^^^^^^^^ params in this part
    /// ```
    ///
    /// `T` and `V` in this example.
    other_params: FxHashSet<hir::GenericParam>,
}

fn compute_contained_params_in_generic_param(
    ctx: &AssistContext<'_>,
    node: ast::GenericParam,
) -> Option<ParamBoundWithParams> {
    match &node {
        ast::GenericParam::TypeParam(ty) => {
            let self_ty_param = ctx.sema.to_def(ty)?.into();

            let other_params = ty
                .type_bound_list()
                .into_iter()
                .flat_map(|it| it.bounds())
                .flat_map(|bound| bound.syntax().descendants())
                .filter_map(|node| filter_generic_params(ctx, node))
                .collect();

            Some(ParamBoundWithParams { node, self_ty_param, other_params })
        }
        ast::GenericParam::ConstParam(ct) => {
            let self_ty_param = ctx.sema.to_def(ct)?.into();
            Some(ParamBoundWithParams { node, self_ty_param, other_params: FxHashSet::default() })
        }
        ast::GenericParam::LifetimeParam(_) => {
            // FIXME: It might be a good idea to handle lifetime parameters too.
            None
        }
    }
}

fn compute_contained_params_in_where_pred(
    ctx: &AssistContext<'_>,
    node: ast::WherePred,
) -> Option<WherePredWithParams> {
    let self_ty = node.ty()?;
    let bound_list = node.type_bound_list()?;

    let self_ty_params = self_ty
        .syntax()
        .descendants()
        .filter_map(|node| filter_generic_params(ctx, node))
        .collect();

    let other_params = bound_list
        .bounds()
        .flat_map(|bound| bound.syntax().descendants())
        .filter_map(|node| filter_generic_params(ctx, node))
        .collect();

    Some(WherePredWithParams { node, self_ty_params, other_params })
}

fn filter_generic_params(ctx: &AssistContext<'_>, node: SyntaxNode) -> Option<hir::GenericParam> {
    let path = ast::Path::cast(node)?;
    match ctx.sema.resolve_path(&path)? {
        PathResolution::TypeParam(it) => Some(it.into()),
        PathResolution::ConstParam(it) => Some(it.into()),
        _ => None,
    }
}

/// Filters out irrelevant bounds from `generic_params` and `where_preds`.
///
/// Say we have a trait bound `Struct<T>: Trait<U>`. Given `necessary_params`, when is it relevant
/// and when not? Some observations:
/// - When `necessary_params` contains `T`, it's likely that we want this bound, but now we have
/// an extra param to consider: `U`.
/// - On the other hand, when `necessary_params` contains `U` (but not `T`), then it's unlikely
/// that we want this bound because it doesn't really constrain `U`.
///
/// (FIXME?: The latter clause might be overstating. We may want to include the bound if the self
/// type does *not* include generic params at all - like `Option<i32>: From<U>`)
///
/// Can we make this a bit more formal? Let's define "dependency" between generic parameters and
/// trait bounds:
/// - A generic parameter `T` depends on a trait bound if `T` appears in the self type (i.e. left
/// part) of the bound.
/// - A trait bound depends on a generic parameter `T` if `T` appears in the bound.
///
/// Using the notion, what we want is all the bounds that params in `necessary_params`
/// *transitively* depend on!
///
/// Now it's not hard to solve: we build a dependency graph and compute all reachable nodes from
/// nodes that represent params in `necessary_params` by usual and boring DFS.
///
/// The time complexity is O(|generic_params| + |where_preds| + |necessary_params|).
fn filter_unnecessary_bounds(
    generic_params: &mut Vec<ParamBoundWithParams>,
    where_preds: &mut Vec<WherePredWithParams>,
    necessary_params: FxHashSet<hir::GenericParam>,
) {
    // All `self_ty_param` should be unique as they were collected from `ast::GenericParamList`s.
    let param_map: FxHashMap<hir::GenericParam, usize> =
        generic_params.iter().map(|it| it.self_ty_param).zip(0..).collect();
    let param_count = param_map.len();
    let generic_params_upper_bound = param_count + generic_params.len();
    let node_count = generic_params_upper_bound + where_preds.len();

    // | node index range                        | what the node represents |
    // |-----------------------------------------|--------------------------|
    // | 0..param_count                          | generic parameter        |
    // | param_count..generic_params_upper_bound | `ast::GenericParam`      |
    // | generic_params_upper_bound..node_count  | `ast::WherePred`         |
    let mut graph = Graph::new(node_count);
    for (pred, pred_idx) in generic_params.iter().zip(param_count..) {
        let param_idx = param_map[&pred.self_ty_param];
        graph.add_edge(param_idx, pred_idx);
        graph.add_edge(pred_idx, param_idx);

        for param in &pred.other_params {
            let param_idx = param_map[param];
            graph.add_edge(pred_idx, param_idx);
        }
    }
    for (pred, pred_idx) in where_preds.iter().zip(generic_params_upper_bound..) {
        for param in &pred.self_ty_params {
            let param_idx = param_map[param];
            graph.add_edge(param_idx, pred_idx);
            graph.add_edge(pred_idx, param_idx);
        }
        for param in &pred.other_params {
            let param_idx = param_map[param];
            graph.add_edge(pred_idx, param_idx);
        }
    }

    let starting_nodes = necessary_params.iter().map(|param| param_map[param]);
    let reachable = graph.compute_reachable_nodes(starting_nodes);

    // Not pretty, but effective. If only there were `Vec::retain_index()`...
    let mut idx = param_count;
    generic_params.retain(|_| {
        idx += 1;
        reachable[idx - 1]
    });
    stdx::always!(idx == generic_params_upper_bound, "inconsistent index");
    where_preds.retain(|_| {
        idx += 1;
        reachable[idx - 1]
    });
}

/// Filters out bounds from impl if we're generating the function into the same impl we're
/// generating from.
fn filter_bounds_in_scope(
    generic_params: &mut Vec<ParamBoundWithParams>,
    where_preds: &mut Vec<WherePredWithParams>,
    ctx: &AssistContext<'_>,
    target: &GeneratedFunctionTarget,
) -> Option<()> {
    let target_impl = target.parent().ancestors().find_map(ast::Impl::cast)?;
    let target_impl = ctx.sema.to_def(&target_impl)?;
    // It's sufficient to test only the first element of `generic_params` because of the order of
    // insertion (see `params_and_where_preds_in_scope()`).
    let def = generic_params.first()?.self_ty_param.parent();
    if def != hir::GenericDef::Impl(target_impl) {
        return None;
    }

    // Now we know every element that belongs to an impl would be in scope at `target`, we can
    // filter them out just by looking at their parent.
    generic_params.retain(|it| !matches!(it.self_ty_param.parent(), hir::GenericDef::Impl(_)));
    where_preds.retain(|it| {
        it.node.syntax().parent().and_then(|it| it.parent()).and_then(ast::Impl::cast).is_none()
    });

    Some(())
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
fn deduplicate_arg_names(arg_names: &mut [String]) {
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

fn fn_arg_type(
    ctx: &AssistContext<'_>,
    target_module: Module,
    fn_arg: &ast::Expr,
    generic_params: &mut FxHashSet<hir::GenericParam>,
) -> String {
    fn maybe_displayed_type(
        ctx: &AssistContext<'_>,
        target_module: Module,
        fn_arg: &ast::Expr,
        generic_params: &mut FxHashSet<hir::GenericParam>,
    ) -> Option<String> {
        let ty = ctx.sema.type_of_expr(fn_arg)?.adjusted();
        if ty.is_unknown() {
            return None;
        }

        generic_params.extend(ty.generic_params(ctx.db()));

        if ty.is_reference() || ty.is_mutable_reference() {
            let famous_defs = &FamousDefs(&ctx.sema, ctx.sema.scope(fn_arg.syntax())?.krate());
            convert_reference_type(ty.strip_references(), ctx.db(), famous_defs)
                .map(|conversion| conversion.convert_type(ctx.db()))
                .or_else(|| ty.display_source_code(ctx.db(), target_module.into(), true).ok())
        } else {
            ty.display_source_code(ctx.db(), target_module.into(), true).ok()
        }
    }

    maybe_displayed_type(ctx, target_module, fn_arg, generic_params)
        .unwrap_or_else(|| String::from("_"))
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
    db: &dyn hir::db::ExpandDatabase,
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
    let assoc_item_list = impl_.assoc_item_list()?;
    if let Some(last_item) = assoc_item_list.assoc_items().last() {
        Some(GeneratedFunctionTarget::BehindItem(last_item.syntax().clone()))
    } else {
        Some(GeneratedFunctionTarget::InEmptyItemList(assoc_item_list.syntax().clone()))
    }
}

#[derive(Clone, Copy)]
enum Visibility {
    None,
    Crate,
    Pub,
}

fn calculate_necessary_visibility(
    current_module: Module,
    target_module: Module,
    ctx: &AssistContext<'_>,
) -> Visibility {
    let db = ctx.db();
    let current_module = current_module.nearest_non_block_module(db);
    let target_module = target_module.nearest_non_block_module(db);

    if target_module.krate() != current_module.krate() {
        Visibility::Pub
    } else if current_module.path_to_root(db).contains(&target_module) {
        Visibility::None
    } else {
        Visibility::Crate
    }
}

// This is never intended to be used as a generic graph structure. If there's ever another need of
// graph algorithm, consider adding a library for that (and replace the following).
/// Minimally implemented directed graph structure represented by adjacency list.
struct Graph {
    edges: Vec<Vec<usize>>,
}

impl Graph {
    fn new(node_count: usize) -> Self {
        Self { edges: vec![Vec::new(); node_count] }
    }

    fn add_edge(&mut self, from: usize, to: usize) {
        self.edges[from].push(to);
    }

    fn edges_for(&self, node_idx: usize) -> &[usize] {
        &self.edges[node_idx]
    }

    fn len(&self) -> usize {
        self.edges.len()
    }

    fn compute_reachable_nodes(
        &self,
        starting_nodes: impl IntoIterator<Item = usize>,
    ) -> Vec<bool> {
        let mut visitor = Visitor::new(self);
        for idx in starting_nodes {
            visitor.mark_reachable(idx);
        }
        visitor.visited
    }
}

struct Visitor<'g> {
    graph: &'g Graph,
    visited: Vec<bool>,
    // Stack is held in this struct so we can reuse its buffer.
    stack: Vec<usize>,
}

impl<'g> Visitor<'g> {
    fn new(graph: &'g Graph) -> Self {
        let visited = vec![false; graph.len()];
        Self { graph, visited, stack: Vec::new() }
    }

    fn mark_reachable(&mut self, start_idx: usize) {
        // non-recursive DFS
        stdx::always!(self.stack.is_empty());

        self.stack.push(start_idx);
        while let Some(idx) = self.stack.pop() {
            if !self.visited[idx] {
                self.visited[idx] = true;
                for &neighbor in self.graph.edges_for(idx) {
                    if !self.visited[neighbor] {
                        self.stack.push(neighbor);
                    }
                }
            }
        }
    }
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
    fn generate_function_with_generic_param() {
        check_assist(
            generate_function,
            r"
fn foo<T, const N: usize>(t: [T; N]) { $0bar(t) }
",
            r"
fn foo<T, const N: usize>(t: [T; N]) { bar(t) }

fn bar<T, const N: usize>(t: [T; N]) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn generate_function_with_parent_generic_param() {
        check_assist(
            generate_function,
            r"
struct S<T>(T);
impl<T> S<T> {
    fn foo<U>(t: T, u: U) { $0bar(t, u) }
}
",
            r"
struct S<T>(T);
impl<T> S<T> {
    fn foo<U>(t: T, u: U) { bar(t, u) }
}

fn bar<T, U>(t: T, u: U) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn generic_param_in_receiver_type() {
        // FIXME: Generic parameter `T` should be part of impl, not method.
        check_assist(
            generate_function,
            r"
struct S<T>(T);
fn foo<T, U>(s: S<T>, u: U) { s.$0foo(u) }
",
            r"
struct S<T>(T);
impl S {
    fn foo<T, U>(&self, u: U) {
        ${0:todo!()}
    }
}
fn foo<T, U>(s: S<T>, u: U) { s.foo(u) }
",
        )
    }

    #[test]
    fn generic_param_in_return_type() {
        check_assist(
            generate_function,
            r"
fn foo<T, const N: usize>() -> [T; N] { $0bar() }
",
            r"
fn foo<T, const N: usize>() -> [T; N] { bar() }

fn bar<T, const N: usize>() -> [T; N] {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn generate_fn_with_bounds() {
        // FIXME: where predicates should be on next lines.
        check_assist(
            generate_function,
            r"
trait A<T> {}
struct S<T>(T);
impl<T: A<i32>> S<T>
where
    T: A<i64>,
{
    fn foo<U>(t: T, u: U)
    where
        T: A<()>,
        U: A<i32> + A<i64>,
    {
        $0bar(t, u)
    }
}
",
            r"
trait A<T> {}
struct S<T>(T);
impl<T: A<i32>> S<T>
where
    T: A<i64>,
{
    fn foo<U>(t: T, u: U)
    where
        T: A<()>,
        U: A<i32> + A<i64>,
    {
        bar(t, u)
    }
}

fn bar<T: A<i32>, U>(t: T, u: U) where T: A<i64>, T: A<()>, U: A<i32> + A<i64> {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn include_transitive_param_dependency() {
        // FIXME: where predicates should be on next lines.
        check_assist(
            generate_function,
            r"
trait A<T> { type Assoc; }
trait B { type Item; }
struct S<T>(T);
impl<T, U, V: B, W> S<(T, U, V, W)>
where
    T: A<U, Assoc = V>,
    S<V::Item>: A<U, Assoc = W>,
{
    fn foo<I>(t: T, u: U)
    where
        U: A<T, Assoc = I>,
    {
        $0bar(u)
    }
}
",
            r"
trait A<T> { type Assoc; }
trait B { type Item; }
struct S<T>(T);
impl<T, U, V: B, W> S<(T, U, V, W)>
where
    T: A<U, Assoc = V>,
    S<V::Item>: A<U, Assoc = W>,
{
    fn foo<I>(t: T, u: U)
    where
        U: A<T, Assoc = I>,
    {
        bar(u)
    }
}

fn bar<T, U, V: B, W, I>(u: U) where T: A<U, Assoc = V>, S<V::Item>: A<U, Assoc = W>, U: A<T, Assoc = I> {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn irrelevant_bounds_are_filtered_out() {
        check_assist(
            generate_function,
            r"
trait A<T> {}
struct S<T>(T);
impl<T, U, V, W> S<(T, U, V, W)>
where
    T: A<U>,
    V: A<W>,
{
    fn foo<I>(t: T, u: U)
    where
        U: A<T> + A<I>,
    {
        $0bar(u)
    }
}
",
            r"
trait A<T> {}
struct S<T>(T);
impl<T, U, V, W> S<(T, U, V, W)>
where
    T: A<U>,
    V: A<W>,
{
    fn foo<I>(t: T, u: U)
    where
        U: A<T> + A<I>,
    {
        bar(u)
    }
}

fn bar<T, U, I>(u: U) where T: A<U>, U: A<T> + A<I> {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn params_in_trait_arg_are_not_dependency() {
        // Even though `bar` depends on `U` and `I`, we don't have to copy these bounds:
        // `T: A<I>` and `T: A<U>`.
        check_assist(
            generate_function,
            r"
trait A<T> {}
struct S<T>(T);
impl<T, U> S<(T, U)>
where
    T: A<U>,
{
    fn foo<I>(t: T, u: U)
    where
        T: A<I>,
        U: A<I>,
    {
        $0bar(u)
    }
}
",
            r"
trait A<T> {}
struct S<T>(T);
impl<T, U> S<(T, U)>
where
    T: A<U>,
{
    fn foo<I>(t: T, u: U)
    where
        T: A<I>,
        U: A<I>,
    {
        bar(u)
    }
}

fn bar<U, I>(u: U) where U: A<I> {
    ${0:todo!()}
}
",
        )
    }

    #[test]
    fn dont_copy_bounds_already_in_scope() {
        check_assist(
            generate_function,
            r"
trait A<T> {}
struct S<T>(T);
impl<T: A<i32>> S<T>
where
    T: A<usize>,
{
    fn foo<U: A<()>>(t: T, u: U)
    where
        T: A<S<i32>>,
    {
        Self::$0bar(t, u);
    }
}
",
            r"
trait A<T> {}
struct S<T>(T);
impl<T: A<i32>> S<T>
where
    T: A<usize>,
{
    fn foo<U: A<()>>(t: T, u: U)
    where
        T: A<S<i32>>,
    {
        Self::bar(t, u);
    }

    fn bar<U: A<()>>(t: T, u: U) ${0:-> _} where T: A<S<i32>> {
        todo!()
    }
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

fn bar(closure: impl Fn(i64) -> i64) {
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
    fn qualified_path_in_generic_bounds_uses_correct_scope() {
        check_assist(
            generate_function,
            r"
mod a {
    pub trait A {};
}
pub mod b {
    pub struct S<T>(T);
}
struct S<T>(T);
impl<T> S<T>
where
    T: a::A,
{
    fn foo<U: a::A>(t: b::S<T>, u: S<U>) {
        a::$0bar(t, u);
    }
}
",
            r"
mod a {
    pub trait A {}

    pub(crate) fn bar<T, U: self::A>(t: crate::b::S<T>, u: crate::S<U>) ${0:-> _} where T: self::A {
        todo!()
    };
}
pub mod b {
    pub struct S<T>(T);
}
struct S<T>(T);
impl<T> S<T>
where
    T: a::A,
{
    fn foo<U: a::A>(t: b::S<T>, u: S<U>) {
        a::bar(t, u);
    }
}
",
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
    pub mod baz {}
}

fn foo() {
    bar::baz::my_fn$0()
}
",
            r"
mod bar {
    pub mod baz {
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
async fn foo() {
    $0bar(42).await;
}
",
            r"
async fn foo() {
    bar(42).await;
}

async fn bar(arg: i32) ${0:-> _} {
    todo!()
}
",
        )
    }

    #[test]
    fn return_type_for_async_fn() {
        check_assist(
            generate_function,
            r"
//- minicore: result
async fn foo() {
    if Err(()) = $0bar(42).await {}
}
",
            r"
async fn foo() {
    if Err(()) = bar(42).await {}
}

async fn bar(arg: i32) -> Result<_, ()> {
    ${0:todo!()}
}
",
        );
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
impl S {
    fn bar(&self) ${0:-> _} {
        todo!()
    }
}
fn foo() {S.bar();}
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
impl S {
    fn bar(&self) ${0:-> _} {
        todo!()
    }
}
mod s {
    fn foo() {
        super::S.bar();
    }
}

",
        )
    }

    #[test]
    fn create_method_with_cursor_anywhere_on_call_expression() {
        check_assist(
            generate_function,
            r"
struct S;
fn foo() {$0S.bar();}
",
            r"
struct S;
impl S {
    fn bar(&self) ${0:-> _} {
        todo!()
    }
}
fn foo() {S.bar();}
",
        )
    }

    #[test]
    fn create_async_method() {
        check_assist(
            generate_function,
            r"
//- minicore: result
struct S;
async fn foo() {
    if let Err(()) = S.$0bar(42).await {}
}
",
            r"
struct S;
impl S {
    async fn bar(&self, arg: i32) -> Result<_, ()> {
        ${0:todo!()}
    }
}
async fn foo() {
    if let Err(()) = S.bar(42).await {}
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
impl S {
    fn bar() ${0:-> _} {
        todo!()
    }
}
fn foo() {S::bar();}
",
        )
    }

    #[test]
    fn create_async_static_method() {
        check_assist(
            generate_function,
            r"
//- minicore: result
struct S;
async fn foo() {
    if let Err(()) = S::$0bar(42).await {}
}
",
            r"
struct S;
impl S {
    async fn bar(arg: i32) -> Result<_, ()> {
        ${0:todo!()}
    }
}
async fn foo() {
    if let Err(()) = S::bar(42).await {}
}
",
        )
    }

    #[test]
    fn create_generic_static_method() {
        check_assist(
            generate_function,
            r"
struct S;
fn foo<T, const N: usize>(t: [T; N]) { S::bar$0(t); }
",
            r"
struct S;
impl S {
    fn bar<T, const N: usize>(t: [T; N]) ${0:-> _} {
        todo!()
    }
}
fn foo<T, const N: usize>(t: [T; N]) { S::bar(t); }
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
    fn create_static_method_with_cursor_anywhere_on_call_expression() {
        check_assist(
            generate_function,
            r"
struct S;
fn foo() {$0S::bar();}
",
            r"
struct S;
impl S {
    fn bar() ${0:-> _} {
        todo!()
    }
}
fn foo() {S::bar();}
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
impl Foo {
    fn new() ${0:-> _} {
        todo!()
    }
}
fn main() {
    Foo::new();
}
",
        )
    }

    #[test]
    fn applicable_in_different_local_crate() {
        check_assist(
            generate_function,
            r"
//- /lib.rs crate:lib new_source_root:local
fn dummy() {}
//- /main.rs crate:main deps:lib new_source_root:local
fn main() {
    lib::foo$0();
}
",
            r"
fn dummy() {}

pub fn foo() ${0:-> _} {
    todo!()
}
",
        );
    }

    #[test]
    fn applicable_in_different_local_crate_method() {
        check_assist(
            generate_function,
            r"
//- /lib.rs crate:lib new_source_root:local
pub struct S;
//- /main.rs crate:main deps:lib new_source_root:local
fn main() {
    lib::S.foo$0();
}
",
            r"
pub struct S;
impl S {
    pub fn foo(&self) ${0:-> _} {
        todo!()
    }
}
",
        );
    }

    #[test]
    fn not_applicable_in_different_library_crate() {
        check_assist_not_applicable(
            generate_function,
            r"
//- /lib.rs crate:lib new_source_root:library
fn dummy() {}
//- /main.rs crate:main deps:lib new_source_root:local
fn main() {
    lib::foo$0();
}
",
        );
    }

    #[test]
    fn not_applicable_in_different_library_crate_method() {
        check_assist_not_applicable(
            generate_function,
            r"
//- /lib.rs crate:lib new_source_root:library
pub struct S;
//- /main.rs crate:main deps:lib new_source_root:local
fn main() {
    lib::S.foo$0();
}
",
        );
    }
}
