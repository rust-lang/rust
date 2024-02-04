//! Renderer for function calls.

use hir::{db::HirDatabase, AsAssocItem, HirDisplay};
use ide_db::{SnippetCap, SymbolKind};
use itertools::Itertools;
use stdx::{format_to, to_lower_snake_case};
use syntax::{format_smolstr, AstNode, SmolStr};

use crate::{
    context::{CompletionContext, DotAccess, DotAccessKind, PathCompletionCtx, PathKind},
    item::{Builder, CompletionItem, CompletionItemKind, CompletionRelevance},
    render::{compute_exact_name_match, compute_ref_match, compute_type_match, RenderContext},
    CallableSnippets,
};

#[derive(Debug)]
enum FuncKind<'ctx> {
    Function(&'ctx PathCompletionCtx),
    Method(&'ctx DotAccess, Option<hir::Name>),
}

pub(crate) fn render_fn(
    ctx: RenderContext<'_>,
    path_ctx: &PathCompletionCtx,
    local_name: Option<hir::Name>,
    func: hir::Function,
) -> Builder {
    let _p = tracing::span!(tracing::Level::INFO, "render_fn").entered();
    render(ctx, local_name, func, FuncKind::Function(path_ctx))
}

pub(crate) fn render_method(
    ctx: RenderContext<'_>,
    dot_access: &DotAccess,
    receiver: Option<hir::Name>,
    local_name: Option<hir::Name>,
    func: hir::Function,
) -> Builder {
    let _p = tracing::span!(tracing::Level::INFO, "render_method").entered();
    render(ctx, local_name, func, FuncKind::Method(dot_access, receiver))
}

fn render(
    ctx @ RenderContext { completion, .. }: RenderContext<'_>,
    local_name: Option<hir::Name>,
    func: hir::Function,
    func_kind: FuncKind<'_>,
) -> Builder {
    let db = completion.db;

    let name = local_name.unwrap_or_else(|| func.name(db));

    let (call, escaped_call) = match &func_kind {
        FuncKind::Method(_, Some(receiver)) => (
            format_smolstr!(
                "{}.{}",
                receiver.unescaped().display(ctx.db()),
                name.unescaped().display(ctx.db())
            ),
            format_smolstr!("{}.{}", receiver.display(ctx.db()), name.display(ctx.db())),
        ),
        _ => (name.unescaped().to_smol_str(), name.to_smol_str()),
    };

    let mut item = CompletionItem::new(
        if func.self_param(db).is_some() {
            CompletionItemKind::Method
        } else {
            CompletionItemKind::SymbolKind(SymbolKind::Function)
        },
        ctx.source_range(),
        call.clone(),
    );

    let ret_type = func.ret_type(db);
    let assoc_item = func.as_assoc_item(db);

    let trait_ = assoc_item.and_then(|trait_| trait_.containing_trait_or_trait_impl(db));
    let is_op_method = trait_.map_or(false, |trait_| completion.is_ops_trait(trait_));

    let is_item_from_notable_trait =
        trait_.map_or(false, |trait_| completion.is_doc_notable_trait(trait_));

    let (has_dot_receiver, has_call_parens, cap) = match func_kind {
        FuncKind::Function(&PathCompletionCtx {
            kind: PathKind::Expr { .. },
            has_call_parens,
            ..
        }) => (false, has_call_parens, ctx.completion.config.snippet_cap),
        FuncKind::Method(&DotAccess { kind: DotAccessKind::Method { has_parens }, .. }, _) => {
            (true, has_parens, ctx.completion.config.snippet_cap)
        }
        FuncKind::Method(DotAccess { kind: DotAccessKind::Field { .. }, .. }, _) => {
            (true, false, ctx.completion.config.snippet_cap)
        }
        _ => (false, false, None),
    };
    let complete_call_parens = cap
        .filter(|_| !has_call_parens)
        .and_then(|cap| Some((cap, params(ctx.completion, func, &func_kind, has_dot_receiver)?)));

    item.set_relevance(CompletionRelevance {
        type_match: if has_call_parens || complete_call_parens.is_some() {
            compute_type_match(completion, &ret_type)
        } else {
            compute_type_match(completion, &func.ty(db))
        },
        exact_name_match: compute_exact_name_match(completion, &call),
        is_op_method,
        is_item_from_notable_trait,
        ..ctx.completion_relevance()
    });

    match func_kind {
        FuncKind::Function(path_ctx) => {
            super::path_ref_match(completion, path_ctx, &ret_type, &mut item);
        }
        FuncKind::Method(DotAccess { receiver: Some(receiver), .. }, _) => {
            if let Some(original_expr) = completion.sema.original_ast_node(receiver.clone()) {
                if let Some(ref_match) = compute_ref_match(completion, &ret_type) {
                    item.ref_match(ref_match, original_expr.syntax().text_range().start());
                }
            }
        }
        _ => (),
    }

    let detail = if ctx.completion.config.full_function_signatures {
        detail_full(db, func)
    } else {
        detail(db, func)
    };
    item.set_documentation(ctx.docs(func))
        .set_deprecated(ctx.is_deprecated(func) || ctx.is_deprecated_assoc_item(func))
        .detail(detail)
        .lookup_by(name.unescaped().to_smol_str());

    if let Some((cap, (self_param, params))) = complete_call_parens {
        add_call_parens(&mut item, completion, cap, call, escaped_call, self_param, params);
    }

    match ctx.import_to_add {
        Some(import_to_add) => {
            item.add_import(import_to_add);
        }
        None => {
            if let Some(actm) = assoc_item {
                if let Some(trt) = actm.containing_trait_or_trait_impl(db) {
                    item.trait_name(trt.name(db).to_smol_str());
                }
            }
        }
    }

    item.doc_aliases(ctx.doc_aliases);
    item
}

pub(super) fn add_call_parens<'b>(
    builder: &'b mut Builder,
    ctx: &CompletionContext<'_>,
    cap: SnippetCap,
    name: SmolStr,
    escaped_name: SmolStr,
    self_param: Option<hir::SelfParam>,
    params: Vec<hir::Param>,
) -> &'b mut Builder {
    cov_mark::hit!(inserts_parens_for_function_calls);

    let (snippet, label_suffix) = if self_param.is_none() && params.is_empty() {
        (format!("{escaped_name}()$0"), "()")
    } else {
        builder.trigger_call_info();
        let snippet = if let Some(CallableSnippets::FillArguments) = ctx.config.callable {
            let offset = if self_param.is_some() { 2 } else { 1 };
            let function_params_snippet =
                params.iter().enumerate().format_with(", ", |(index, param), f| {
                    match param.name(ctx.db) {
                        Some(n) => {
                            let smol_str = n.to_smol_str();
                            let text = smol_str.as_str().trim_start_matches('_');
                            let ref_ = ref_of_param(ctx, text, param.ty());
                            f(&format_args!("${{{}:{ref_}{text}}}", index + offset))
                        }
                        None => {
                            let name = match param.ty().as_adt() {
                                None => "_".to_string(),
                                Some(adt) => adt
                                    .name(ctx.db)
                                    .as_text()
                                    .map(|s| to_lower_snake_case(s.as_str()))
                                    .unwrap_or_else(|| "_".to_string()),
                            };
                            f(&format_args!("${{{}:{name}}}", index + offset))
                        }
                    }
                });
            match self_param {
                Some(self_param) => {
                    format!(
                        "{}(${{1:{}}}{}{})$0",
                        escaped_name,
                        self_param.display(ctx.db),
                        if params.is_empty() { "" } else { ", " },
                        function_params_snippet
                    )
                }
                None => {
                    format!("{escaped_name}({function_params_snippet})$0")
                }
            }
        } else {
            cov_mark::hit!(suppress_arg_snippets);
            format!("{escaped_name}($0)")
        };

        (snippet, "(…)")
    };
    builder.label(SmolStr::from_iter([&name, label_suffix])).insert_snippet(cap, snippet)
}

fn ref_of_param(ctx: &CompletionContext<'_>, arg: &str, ty: &hir::Type) -> &'static str {
    if let Some(derefed_ty) = ty.remove_ref() {
        for (name, local) in ctx.locals.iter() {
            if name.as_text().as_deref() == Some(arg) {
                return if local.ty(ctx.db) == derefed_ty {
                    if ty.is_mutable_reference() {
                        "&mut "
                    } else {
                        "&"
                    }
                } else {
                    ""
                };
            }
        }
    }
    ""
}

fn detail(db: &dyn HirDatabase, func: hir::Function) -> String {
    let mut ret_ty = func.ret_type(db);
    let mut detail = String::new();

    if func.is_const(db) {
        format_to!(detail, "const ");
    }
    if func.is_async(db) {
        format_to!(detail, "async ");
        if let Some(async_ret) = func.async_ret_type(db) {
            ret_ty = async_ret;
        }
    }
    if func.is_unsafe_to_call(db) {
        format_to!(detail, "unsafe ");
    }

    format_to!(detail, "fn({})", params_display(db, func));
    if !ret_ty.is_unit() {
        format_to!(detail, " -> {}", ret_ty.display(db));
    }
    detail
}

fn detail_full(db: &dyn HirDatabase, func: hir::Function) -> String {
    let signature = format!("{}", func.display(db));
    let mut detail = String::with_capacity(signature.len());

    for segment in signature.split_whitespace() {
        if !detail.is_empty() {
            detail.push(' ');
        }

        detail.push_str(segment);
    }

    detail
}

fn params_display(db: &dyn HirDatabase, func: hir::Function) -> String {
    if let Some(self_param) = func.self_param(db) {
        let assoc_fn_params = func.assoc_fn_params(db);
        let params = assoc_fn_params
            .iter()
            .skip(1) // skip the self param because we are manually handling that
            .map(|p| p.ty().display(db));
        format!(
            "{}{}",
            self_param.display(db),
            params.format_with("", |display, f| {
                f(&", ")?;
                f(&display)
            })
        )
    } else {
        let assoc_fn_params = func.assoc_fn_params(db);
        assoc_fn_params.iter().map(|p| p.ty().display(db)).join(", ")
    }
}

fn params(
    ctx: &CompletionContext<'_>,
    func: hir::Function,
    func_kind: &FuncKind<'_>,
    has_dot_receiver: bool,
) -> Option<(Option<hir::SelfParam>, Vec<hir::Param>)> {
    ctx.config.callable.as_ref()?;

    // Don't add parentheses if the expected type is a function reference with the same signature.
    if let Some(expected) = ctx.expected_type.as_ref().filter(|e| e.is_fn()) {
        if let Some(expected) = expected.as_callable(ctx.db) {
            if let Some(completed) = func.ty(ctx.db).as_callable(ctx.db) {
                if expected.sig() == completed.sig() {
                    cov_mark::hit!(no_call_parens_if_fn_ptr_needed);
                    return None;
                }
            }
        }
    }

    let self_param = if has_dot_receiver || matches!(func_kind, FuncKind::Method(_, Some(_))) {
        None
    } else {
        func.self_param(ctx.db)
    };
    Some((self_param, func.params_without_self(ctx.db)))
}

#[cfg(test)]
mod tests {
    use crate::{
        tests::{check_edit, check_edit_with_config, TEST_CONFIG},
        CallableSnippets, CompletionConfig,
    };

    #[test]
    fn inserts_parens_for_function_calls() {
        cov_mark::check!(inserts_parens_for_function_calls);
        check_edit(
            "no_args",
            r#"
fn no_args() {}
fn main() { no_$0 }
"#,
            r#"
fn no_args() {}
fn main() { no_args()$0 }
"#,
        );

        check_edit(
            "with_args",
            r#"
fn with_args(x: i32, y: String) {}
fn main() { with_$0 }
"#,
            r#"
fn with_args(x: i32, y: String) {}
fn main() { with_args(${1:x}, ${2:y})$0 }
"#,
        );

        check_edit(
            "foo",
            r#"
struct S;
impl S {
    fn foo(&self) {}
}
fn bar(s: &S) { s.f$0 }
"#,
            r#"
struct S;
impl S {
    fn foo(&self) {}
}
fn bar(s: &S) { s.foo()$0 }
"#,
        );

        check_edit(
            "foo",
            r#"
struct S {}
impl S {
    fn foo(&self, x: i32) {}
}
fn bar(s: &S) {
    s.f$0
}
"#,
            r#"
struct S {}
impl S {
    fn foo(&self, x: i32) {}
}
fn bar(s: &S) {
    s.foo(${1:x})$0
}
"#,
        );

        check_edit(
            "foo",
            r#"
struct S {}
impl S {
    fn foo(&self, x: i32) {
        $0
    }
}
"#,
            r#"
struct S {}
impl S {
    fn foo(&self, x: i32) {
        self.foo(${1:x})$0
    }
}
"#,
        );
    }

    #[test]
    fn parens_for_method_call_as_assoc_fn() {
        check_edit(
            "foo",
            r#"
struct S;
impl S {
    fn foo(&self) {}
}
fn main() { S::f$0 }
"#,
            r#"
struct S;
impl S {
    fn foo(&self) {}
}
fn main() { S::foo(${1:&self})$0 }
"#,
        );
    }

    #[test]
    fn suppress_arg_snippets() {
        cov_mark::check!(suppress_arg_snippets);
        check_edit_with_config(
            CompletionConfig { callable: Some(CallableSnippets::AddParentheses), ..TEST_CONFIG },
            "with_args",
            r#"
fn with_args(x: i32, y: String) {}
fn main() { with_$0 }
"#,
            r#"
fn with_args(x: i32, y: String) {}
fn main() { with_args($0) }
"#,
        );
    }

    #[test]
    fn strips_underscores_from_args() {
        check_edit(
            "foo",
            r#"
fn foo(_foo: i32, ___bar: bool, ho_ge_: String) {}
fn main() { f$0 }
"#,
            r#"
fn foo(_foo: i32, ___bar: bool, ho_ge_: String) {}
fn main() { foo(${1:foo}, ${2:bar}, ${3:ho_ge_})$0 }
"#,
        );
    }

    #[test]
    fn insert_ref_when_matching_local_in_scope() {
        check_edit(
            "ref_arg",
            r#"
struct Foo {}
fn ref_arg(x: &Foo) {}
fn main() {
    let x = Foo {};
    ref_ar$0
}
"#,
            r#"
struct Foo {}
fn ref_arg(x: &Foo) {}
fn main() {
    let x = Foo {};
    ref_arg(${1:&x})$0
}
"#,
        );
    }

    #[test]
    fn insert_mut_ref_when_matching_local_in_scope() {
        check_edit(
            "ref_arg",
            r#"
struct Foo {}
fn ref_arg(x: &mut Foo) {}
fn main() {
    let x = Foo {};
    ref_ar$0
}
"#,
            r#"
struct Foo {}
fn ref_arg(x: &mut Foo) {}
fn main() {
    let x = Foo {};
    ref_arg(${1:&mut x})$0
}
"#,
        );
    }

    #[test]
    fn insert_ref_when_matching_local_in_scope_for_method() {
        check_edit(
            "apply_foo",
            r#"
struct Foo {}
struct Bar {}
impl Bar {
    fn apply_foo(&self, x: &Foo) {}
}

fn main() {
    let x = Foo {};
    let y = Bar {};
    y.$0
}
"#,
            r#"
struct Foo {}
struct Bar {}
impl Bar {
    fn apply_foo(&self, x: &Foo) {}
}

fn main() {
    let x = Foo {};
    let y = Bar {};
    y.apply_foo(${1:&x})$0
}
"#,
        );
    }

    #[test]
    fn trim_mut_keyword_in_func_completion() {
        check_edit(
            "take_mutably",
            r#"
fn take_mutably(mut x: &i32) {}

fn main() {
    take_m$0
}
"#,
            r#"
fn take_mutably(mut x: &i32) {}

fn main() {
    take_mutably(${1:x})$0
}
"#,
        );
    }

    #[test]
    fn complete_pattern_args_with_type_name_if_adt() {
        check_edit(
            "qux",
            r#"
struct Foo {
    bar: i32
}

fn qux(Foo { bar }: Foo) {
    println!("{}", bar);
}

fn main() {
  qu$0
}
"#,
            r#"
struct Foo {
    bar: i32
}

fn qux(Foo { bar }: Foo) {
    println!("{}", bar);
}

fn main() {
  qux(${1:foo})$0
}
"#,
        );
    }

    #[test]
    fn complete_fn_param() {
        // has mut kw
        check_edit(
            "mut bar: u32",
            r#"
fn f(foo: (), mut bar: u32) {}
fn g(foo: (), mut ba$0)
"#,
            r#"
fn f(foo: (), mut bar: u32) {}
fn g(foo: (), mut bar: u32)
"#,
        );

        // has type param
        check_edit(
            "mut bar: u32",
            r#"
fn g(foo: (), mut ba$0: u32)
fn f(foo: (), mut bar: u32) {}
"#,
            r#"
fn g(foo: (), mut bar: u32)
fn f(foo: (), mut bar: u32) {}
"#,
        );
    }

    #[test]
    fn complete_fn_mut_param_add_comma() {
        // add leading and trailing comma
        check_edit(
            ", mut bar: u32,",
            r#"
fn f(foo: (), mut bar: u32) {}
fn g(foo: ()mut ba$0 baz: ())
"#,
            r#"
fn f(foo: (), mut bar: u32) {}
fn g(foo: (), mut bar: u32, baz: ())
"#,
        );
    }

    #[test]
    fn complete_fn_mut_param_has_attribute() {
        check_edit(
            r#"#[baz = "qux"] mut bar: u32"#,
            r#"
fn f(foo: (), #[baz = "qux"] mut bar: u32) {}
fn g(foo: (), mut ba$0)
"#,
            r#"
fn f(foo: (), #[baz = "qux"] mut bar: u32) {}
fn g(foo: (), #[baz = "qux"] mut bar: u32)
"#,
        );

        check_edit(
            r#"#[baz = "qux"] mut bar: u32"#,
            r#"
fn f(foo: (), #[baz = "qux"] mut bar: u32) {}
fn g(foo: (), #[baz = "qux"] mut ba$0)
"#,
            r#"
fn f(foo: (), #[baz = "qux"] mut bar: u32) {}
fn g(foo: (), #[baz = "qux"] mut bar: u32)
"#,
        );

        check_edit(
            r#", #[baz = "qux"] mut bar: u32"#,
            r#"
fn f(foo: (), #[baz = "qux"] mut bar: u32) {}
fn g(foo: ()#[baz = "qux"] mut ba$0)
"#,
            r#"
fn f(foo: (), #[baz = "qux"] mut bar: u32) {}
fn g(foo: (), #[baz = "qux"] mut bar: u32)
"#,
        );
    }
}
