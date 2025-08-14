//! Renderer for function calls.

use hir::{AsAssocItem, HirDisplay, db::HirDatabase};
use ide_db::{SnippetCap, SymbolKind};
use itertools::Itertools;
use stdx::{format_to, to_lower_snake_case};
use syntax::{AstNode, SmolStr, ToSmolStr, format_smolstr};

use crate::{
    CallableSnippets,
    context::{
        CompleteSemicolon, CompletionContext, DotAccess, DotAccessKind, PathCompletionCtx, PathKind,
    },
    item::{
        Builder, CompletionItem, CompletionItemKind, CompletionRelevance, CompletionRelevanceFn,
        CompletionRelevanceReturnType, CompletionRelevanceTraitInfo,
    },
    render::{
        RenderContext, compute_exact_name_match, compute_ref_match, compute_type_match, match_types,
    },
};

#[derive(Debug)]
enum FuncKind<'ctx> {
    Function(&'ctx PathCompletionCtx<'ctx>),
    Method(&'ctx DotAccess<'ctx>, Option<SmolStr>),
}

pub(crate) fn render_fn(
    ctx: RenderContext<'_>,
    path_ctx: &PathCompletionCtx<'_>,
    local_name: Option<hir::Name>,
    func: hir::Function,
) -> Builder {
    let _p = tracing::info_span!("render_fn").entered();
    render(ctx, local_name, func, FuncKind::Function(path_ctx))
}

pub(crate) fn render_method(
    ctx: RenderContext<'_>,
    dot_access: &DotAccess<'_>,
    receiver: Option<SmolStr>,
    local_name: Option<hir::Name>,
    func: hir::Function,
) -> Builder {
    let _p = tracing::info_span!("render_method").entered();
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
            format_smolstr!("{}.{}", receiver, name.as_str()),
            format_smolstr!("{}.{}", receiver, name.display(ctx.db(), completion.edition)),
        ),
        _ => (name.as_str().to_smolstr(), name.display(db, completion.edition).to_smolstr()),
    };
    let has_self_param = func.self_param(db).is_some();
    let mut item = CompletionItem::new(
        CompletionItemKind::SymbolKind(if has_self_param {
            SymbolKind::Method
        } else {
            SymbolKind::Function
        }),
        ctx.source_range(),
        call.clone(),
        completion.edition,
    );

    let ret_type = func.ret_type(db);
    let assoc_item = func.as_assoc_item(db);

    let trait_info =
        assoc_item.and_then(|trait_| trait_.container_or_implemented_trait(db)).map(|trait_| {
            CompletionRelevanceTraitInfo {
                notable_trait: completion.is_doc_notable_trait(trait_),
                is_op_method: completion.is_ops_trait(trait_),
            }
        });

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

    let function = assoc_item
        .and_then(|assoc_item| assoc_item.implementing_ty(db))
        .map(|self_type| compute_return_type_match(db, &ctx, self_type, &ret_type))
        .map(|return_type| CompletionRelevanceFn {
            has_params: has_self_param || func.num_params(db) > 0,
            has_self_param,
            return_type,
        });

    item.set_relevance(CompletionRelevance {
        type_match: if has_call_parens || complete_call_parens.is_some() {
            compute_type_match(completion, &ret_type)
        } else {
            compute_type_match(completion, &func.ty(db))
        },
        exact_name_match: compute_exact_name_match(completion, &call),
        function,
        trait_: trait_info,
        is_skipping_completion: matches!(func_kind, FuncKind::Method(_, Some(_))),
        ..ctx.completion_relevance()
    });

    match func_kind {
        FuncKind::Function(path_ctx) => {
            super::path_ref_match(completion, path_ctx, &ret_type, &mut item);
        }
        FuncKind::Method(DotAccess { receiver: Some(receiver), .. }, _) => {
            if let Some(original_expr) = completion.sema.original_ast_node(receiver.clone())
                && let Some(ref_mode) = compute_ref_match(completion, &ret_type)
            {
                item.ref_match(ref_mode, original_expr.syntax().text_range().start());
            }
        }
        _ => (),
    }

    let detail = if ctx.completion.config.full_function_signatures {
        detail_full(ctx.completion, func)
    } else {
        detail(ctx.completion, func)
    };
    item.set_documentation(ctx.docs(func))
        .set_deprecated(ctx.is_deprecated(func) || ctx.is_deprecated_assoc_item(func))
        .detail(detail)
        .lookup_by(name.as_str().to_smolstr());

    if let Some((cap, (self_param, params))) = complete_call_parens {
        add_call_parens(
            &mut item,
            completion,
            cap,
            call,
            escaped_call,
            self_param,
            params,
            &ret_type,
        );
    }

    match ctx.import_to_add {
        Some(import_to_add) => {
            item.add_import(import_to_add);
        }
        None => {
            if let Some(actm) = assoc_item
                && let Some(trt) = actm.container_or_implemented_trait(db)
            {
                item.trait_name(trt.name(db).display_no_db(ctx.completion.edition).to_smolstr());
            }
        }
    }

    item.doc_aliases(ctx.doc_aliases);
    item
}

fn compute_return_type_match(
    db: &dyn HirDatabase,
    ctx: &RenderContext<'_>,
    self_type: hir::Type<'_>,
    ret_type: &hir::Type<'_>,
) -> CompletionRelevanceReturnType {
    if match_types(ctx.completion, &self_type, ret_type).is_some() {
        // fn([..]) -> Self
        CompletionRelevanceReturnType::DirectConstructor
    } else if ret_type
        .type_arguments()
        .any(|ret_type_arg| match_types(ctx.completion, &self_type, &ret_type_arg).is_some())
    {
        // fn([..]) -> Result<Self, E> OR Wrapped<Foo, Self>
        CompletionRelevanceReturnType::Constructor
    } else if ret_type
        .as_adt()
        .map(|adt| adt.name(db).as_str().ends_with("Builder"))
        .unwrap_or(false)
    {
        // fn([..]) -> [..]Builder
        CompletionRelevanceReturnType::Builder
    } else {
        CompletionRelevanceReturnType::Other
    }
}

pub(super) fn add_call_parens<'b>(
    builder: &'b mut Builder,
    ctx: &CompletionContext<'_>,
    cap: SnippetCap,
    name: SmolStr,
    escaped_name: SmolStr,
    self_param: Option<hir::SelfParam>,
    params: Vec<hir::Param<'_>>,
    ret_type: &hir::Type<'_>,
) -> &'b mut Builder {
    cov_mark::hit!(inserts_parens_for_function_calls);

    let (mut snippet, label_suffix) = if self_param.is_none() && params.is_empty() {
        (format!("{escaped_name}()$0"), "()")
    } else {
        builder.trigger_call_info();
        let snippet = if let Some(CallableSnippets::FillArguments) = ctx.config.callable {
            let offset = if self_param.is_some() { 2 } else { 1 };
            let function_params_snippet =
                params.iter().enumerate().format_with(", ", |(index, param), f| {
                    match param.name(ctx.db) {
                        Some(n) => {
                            let smol_str = n.display_no_db(ctx.edition).to_smolstr();
                            let text = smol_str.as_str().trim_start_matches('_');
                            let ref_ = ref_of_param(ctx, text, param.ty());
                            f(&format_args!("${{{}:{ref_}{text}}}", index + offset))
                        }
                        None => {
                            let name = match param.ty().as_adt() {
                                None => "_".to_owned(),
                                Some(adt) => to_lower_snake_case(adt.name(ctx.db).as_str()),
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
                        self_param.display(ctx.db, ctx.display_target),
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
    if ret_type.is_unit() {
        match ctx.complete_semicolon {
            CompleteSemicolon::DoNotComplete => {}
            CompleteSemicolon::CompleteSemi | CompleteSemicolon::CompleteComma => {
                cov_mark::hit!(complete_semicolon);
                let ch = if matches!(ctx.complete_semicolon, CompleteSemicolon::CompleteComma) {
                    ','
                } else {
                    ';'
                };
                if snippet.ends_with("$0") {
                    snippet.insert(snippet.len() - "$0".len(), ch);
                } else {
                    snippet.push(ch);
                }
            }
        }
    }
    builder.label(SmolStr::from_iter([&name, label_suffix])).insert_snippet(cap, snippet)
}

fn ref_of_param(ctx: &CompletionContext<'_>, arg: &str, ty: &hir::Type<'_>) -> &'static str {
    if let Some(derefed_ty) = ty.remove_ref() {
        for (name, local) in ctx.locals.iter().sorted_by_key(|&(k, _)| k.clone()) {
            if name.as_str() == arg {
                return if local.ty(ctx.db) == derefed_ty {
                    if ty.is_mutable_reference() { "&mut " } else { "&" }
                } else {
                    ""
                };
            }
        }
    }
    ""
}

fn detail(ctx: &CompletionContext<'_>, func: hir::Function) -> String {
    let mut ret_ty = func.ret_type(ctx.db);
    let mut detail = String::new();

    if func.is_const(ctx.db) {
        format_to!(detail, "const ");
    }
    if func.is_async(ctx.db) {
        format_to!(detail, "async ");
        if let Some(async_ret) = func.async_ret_type(ctx.db) {
            ret_ty = async_ret;
        }
    }
    if func.is_unsafe_to_call(ctx.db, ctx.containing_function, ctx.edition) {
        format_to!(detail, "unsafe ");
    }

    detail.push_str("fn(");
    params_display(ctx, &mut detail, func);
    detail.push(')');
    if !ret_ty.is_unit() {
        format_to!(detail, " -> {}", ret_ty.display(ctx.db, ctx.display_target));
    }
    detail
}

fn detail_full(ctx: &CompletionContext<'_>, func: hir::Function) -> String {
    let signature = format!("{}", func.display(ctx.db, ctx.display_target));
    let mut detail = String::with_capacity(signature.len());

    for segment in signature.split_whitespace() {
        if !detail.is_empty() {
            detail.push(' ');
        }

        detail.push_str(segment);
    }

    detail
}

fn params_display(ctx: &CompletionContext<'_>, detail: &mut String, func: hir::Function) {
    if let Some(self_param) = func.self_param(ctx.db) {
        format_to!(detail, "{}", self_param.display(ctx.db, ctx.display_target));
        let assoc_fn_params = func.assoc_fn_params(ctx.db);
        let params = assoc_fn_params
            .iter()
            .skip(1) // skip the self param because we are manually handling that
            .map(|p| p.ty().display(ctx.db, ctx.display_target));
        for param in params {
            format_to!(detail, ", {}", param);
        }
    } else {
        let assoc_fn_params = func.assoc_fn_params(ctx.db);
        format_to!(
            detail,
            "{}",
            assoc_fn_params.iter().map(|p| p.ty().display(ctx.db, ctx.display_target)).format(", ")
        );
    }

    if func.is_varargs(ctx.db) {
        detail.push_str(", ...");
    }
}

fn params<'db>(
    ctx: &CompletionContext<'db>,
    func: hir::Function,
    func_kind: &FuncKind<'_>,
    has_dot_receiver: bool,
) -> Option<(Option<hir::SelfParam>, Vec<hir::Param<'db>>)> {
    ctx.config.callable.as_ref()?;

    // Don't add parentheses if the expected type is a function reference with the same signature.
    if let Some(expected) = ctx.expected_type.as_ref().filter(|e| e.is_fn())
        && let Some(expected) = expected.as_callable(ctx.db)
        && let Some(completed) = func.ty(ctx.db).as_callable(ctx.db)
        && expected.sig() == completed.sig()
    {
        cov_mark::hit!(no_call_parens_if_fn_ptr_needed);
        return None;
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
        CallableSnippets, CompletionConfig,
        tests::{TEST_CONFIG, check_edit, check_edit_with_config},
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
fn main() { no_args();$0 }
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
fn main() { with_args(${1:x}, ${2:y});$0 }
"#,
        );

        check_edit(
            "foo",
            r#"
struct S;
impl S {
    fn foo(&self) -> i32 { 0 }
}
fn bar(s: &S) { s.f$0 }
"#,
            r#"
struct S;
impl S {
    fn foo(&self) -> i32 { 0 }
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
    s.foo(${1:x});$0
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
        self.foo(${1:x});$0
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
fn main() { S::foo(${1:&self});$0 }
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
fn main() { with_args($0); }
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
fn main() { foo(${1:foo}, ${2:bar}, ${3:ho_ge_});$0 }
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
    ref_arg(${1:&x});$0
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
    ref_arg(${1:&mut x});$0
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
    y.apply_foo(${1:&x});$0
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
    take_mutably(${1:x});$0
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
  qux(${1:foo});$0
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

    #[test]
    fn complete_semicolon_for_unit() {
        cov_mark::check!(complete_semicolon);
        check_edit(
            r#"foo"#,
            r#"
fn foo() {}
fn bar() {
    foo$0
}
"#,
            r#"
fn foo() {}
fn bar() {
    foo();$0
}
"#,
        );
        check_edit(
            r#"foo"#,
            r#"
fn foo(a: i32) {}
fn bar() {
    foo$0
}
"#,
            r#"
fn foo(a: i32) {}
fn bar() {
    foo(${1:a});$0
}
"#,
        );
        check_edit(
            r#"foo"#,
            r#"
fn foo(a: i32) {}
fn bar() {
    foo$0;
}
"#,
            r#"
fn foo(a: i32) {}
fn bar() {
    foo(${1:a})$0;
}
"#,
        );
        check_edit_with_config(
            CompletionConfig { add_semicolon_to_unit: false, ..TEST_CONFIG },
            r#"foo"#,
            r#"
fn foo(a: i32) {}
fn bar() {
    foo$0
}
"#,
            r#"
fn foo(a: i32) {}
fn bar() {
    foo(${1:a})$0
}
"#,
        );
    }

    #[test]
    fn complete_comma_for_unit_match_arm() {
        cov_mark::check!(complete_semicolon);
        check_edit(
            r#"foo"#,
            r#"
fn foo() {}
fn bar() {
    match Some(false) {
        v => fo$0
    }
}
"#,
            r#"
fn foo() {}
fn bar() {
    match Some(false) {
        v => foo(),$0
    }
}
"#,
        );
        check_edit(
            r#"foo"#,
            r#"
fn foo() {}
fn bar() {
    match Some(false) {
        v => fo$0,
    }
}
"#,
            r#"
fn foo() {}
fn bar() {
    match Some(false) {
        v => foo()$0,
    }
}
"#,
        );
    }

    #[test]
    fn no_semicolon_in_closure_ret() {
        check_edit(
            r#"foo"#,
            r#"
fn foo() {}
fn baz(_: impl FnOnce()) {}
fn bar() {
    baz(|| fo$0);
}
"#,
            r#"
fn foo() {}
fn baz(_: impl FnOnce()) {}
fn bar() {
    baz(|| foo()$0);
}
"#,
        );
    }
}
