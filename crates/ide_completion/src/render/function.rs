//! Renderer for function calls.

use hir::{db::HirDatabase, AsAssocItem, HirDisplay};
use ide_db::SymbolKind;
use itertools::Itertools;
use stdx::format_to;

use crate::{
    context::CompletionContext,
    item::{CompletionItem, CompletionItemKind, CompletionRelevance, ImportEdit},
    render::{
        builder_ext::Params, compute_exact_name_match, compute_ref_match, compute_type_match,
        RenderContext,
    },
};

enum FuncType {
    Function,
    Method(Option<hir::Name>),
}

pub(crate) fn render_fn(
    ctx: RenderContext<'_>,
    import_to_add: Option<ImportEdit>,
    local_name: Option<hir::Name>,
    func: hir::Function,
) -> CompletionItem {
    let _p = profile::span("render_fn");
    render(ctx, local_name, func, FuncType::Function, import_to_add)
}

pub(crate) fn render_method(
    ctx: RenderContext<'_>,
    import_to_add: Option<ImportEdit>,
    receiver: Option<hir::Name>,
    local_name: Option<hir::Name>,
    func: hir::Function,
) -> CompletionItem {
    let _p = profile::span("render_method");
    render(ctx, local_name, func, FuncType::Method(receiver), import_to_add)
}

fn render(
    ctx @ RenderContext { completion }: RenderContext<'_>,
    local_name: Option<hir::Name>,
    func: hir::Function,
    func_type: FuncType,
    import_to_add: Option<ImportEdit>,
) -> CompletionItem {
    let db = completion.db;

    let name = local_name.unwrap_or_else(|| func.name(db));
    let params = params(completion, func, &func_type);

    let call = match &func_type {
        FuncType::Method(Some(receiver)) => format!("{}.{}", receiver, &name).into(),
        _ => name.to_smol_str(),
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
    item.set_relevance(CompletionRelevance {
        type_match: compute_type_match(completion, &ret_type),
        exact_name_match: compute_exact_name_match(completion, &call),
        is_op_method: match func_type {
            FuncType::Method(_) => func
                .as_assoc_item(ctx.db())
                .and_then(|trait_| trait_.containing_trait_or_trait_impl(ctx.db()))
                .map_or(false, |trait_| completion.is_ops_trait(trait_)),
            _ => false,
        },
        ..CompletionRelevance::default()
    });

    if let Some(ref_match) = compute_ref_match(completion, &ret_type) {
        // FIXME
        // For now we don't properly calculate the edits for ref match
        // completions on methods, so we've disabled them. See #8058.
        if matches!(func_type, FuncType::Function) {
            item.ref_match(ref_match);
        }
    }

    item.set_documentation(ctx.docs(func))
        .set_deprecated(ctx.is_deprecated(func) || ctx.is_deprecated_assoc_item(func))
        .detail(detail(db, func))
        .add_call_parens(completion, call, params);

    if import_to_add.is_none() {
        if let Some(actm) = func.as_assoc_item(db) {
            if let Some(trt) = actm.containing_trait_or_trait_impl(db) {
                item.trait_name(trt.name(db).to_smol_str());
            }
        }
    }

    if let Some(import_to_add) = import_to_add {
        item.add_import(import_to_add);
    }
    item.lookup_by(name.to_smol_str());

    item.build()
}

fn detail(db: &dyn HirDatabase, func: hir::Function) -> String {
    let ret_ty = func.ret_type(db);
    let mut detail = format!("fn({})", params_display(db, func));
    if !ret_ty.is_unit() {
        format_to!(detail, " -> {}", ret_ty.display(db));
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

fn params(ctx: &CompletionContext<'_>, func: hir::Function, func_type: &FuncType) -> Params {
    let (params, self_param) =
        if ctx.has_dot_receiver() || matches!(func_type, FuncType::Method(Some(_))) {
            (func.method_params(ctx.db).unwrap_or_default(), None)
        } else {
            let self_param = func.self_param(ctx.db);

            let mut assoc_params = func.assoc_fn_params(ctx.db);
            if self_param.is_some() {
                assoc_params.remove(0);
            }
            (assoc_params, self_param)
        };

    Params::Named(self_param, params)
}

#[cfg(test)]
mod tests {
    use crate::{
        tests::{check_edit, check_edit_with_config, TEST_CONFIG},
        CompletionConfig,
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
            CompletionConfig { add_call_argument_snippets: false, ..TEST_CONFIG },
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
}
