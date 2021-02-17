//! Renderer for function calls.

use hir::{HasSource, HirDisplay, Type};
use ide_db::SymbolKind;
use syntax::ast::Fn;
use test_utils::mark;

use crate::{
    item::{CompletionItem, CompletionItemKind, CompletionKind, ImportEdit},
    render::{builder_ext::Params, RenderContext},
};

pub(crate) fn render_fn<'a>(
    ctx: RenderContext<'a>,
    import_to_add: Option<ImportEdit>,
    local_name: Option<String>,
    fn_: hir::Function,
) -> Option<CompletionItem> {
    let _p = profile::span("render_fn");
    Some(FunctionRender::new(ctx, local_name, fn_)?.render(import_to_add))
}

#[derive(Debug)]
struct FunctionRender<'a> {
    ctx: RenderContext<'a>,
    name: String,
    func: hir::Function,
    ast_node: Fn,
}

impl<'a> FunctionRender<'a> {
    fn new(
        ctx: RenderContext<'a>,
        local_name: Option<String>,
        fn_: hir::Function,
    ) -> Option<FunctionRender<'a>> {
        let name = local_name.unwrap_or_else(|| fn_.name(ctx.db()).to_string());
        let ast_node = fn_.source(ctx.db())?.value;

        Some(FunctionRender { ctx, name, func: fn_, ast_node })
    }

    fn render(self, import_to_add: Option<ImportEdit>) -> CompletionItem {
        let params = self.params();
        CompletionItem::new(CompletionKind::Reference, self.ctx.source_range(), self.name.clone())
            .kind(self.kind())
            .set_documentation(self.ctx.docs(self.func))
            .set_deprecated(
                self.ctx.is_deprecated(self.func) || self.ctx.is_deprecated_assoc_item(self.func),
            )
            .detail(self.detail())
            .add_call_parens(self.ctx.completion, self.name, params)
            .add_import(import_to_add)
            .build()
    }

    fn detail(&self) -> String {
        let ty = self.func.ret_type(self.ctx.db());
        format!("-> {}", ty.display(self.ctx.db()))
    }

    fn add_arg(&self, arg: &str, ty: &Type) -> String {
        if let Some(derefed_ty) = ty.remove_ref() {
            for (name, local) in self.ctx.completion.locals.iter() {
                if name == arg && local.ty(self.ctx.db()) == derefed_ty {
                    let mutability = if ty.is_mutable_reference() { "&mut " } else { "&" };
                    return format!("{}{}", mutability, arg);
                }
            }
        }
        arg.to_string()
    }

    fn params(&self) -> Params {
        let ast_params = match self.ast_node.param_list() {
            Some(it) => it,
            None => return Params::Named(Vec::new()),
        };

        let mut params_pats = Vec::new();
        let params_ty = if self.ctx.completion.dot_receiver.is_some() {
            self.func.method_params(self.ctx.db()).unwrap_or_default()
        } else {
            if let Some(s) = ast_params.self_param() {
                mark::hit!(parens_for_method_call_as_assoc_fn);
                params_pats.push(Some(s.to_string()));
            }
            self.func.assoc_fn_params(self.ctx.db())
        };
        params_pats
            .extend(ast_params.params().into_iter().map(|it| it.pat().map(|it| it.to_string())));

        let params = params_pats
            .into_iter()
            .zip(params_ty)
            .flat_map(|(pat, param_ty)| {
                let pat = pat?;
                let name = pat;
                let arg = name.trim_start_matches("mut ").trim_start_matches('_');
                Some(self.add_arg(arg, param_ty.ty()))
            })
            .collect();
        Params::Named(params)
    }

    fn kind(&self) -> CompletionItemKind {
        if self.func.self_param(self.ctx.db()).is_some() {
            CompletionItemKind::Method
        } else {
            SymbolKind::Function.into()
        }
    }
}

#[cfg(test)]
mod tests {
    use test_utils::mark;

    use crate::{
        test_utils::{check_edit, check_edit_with_config, TEST_CONFIG},
        CompletionConfig,
    };

    #[test]
    fn inserts_parens_for_function_calls() {
        mark::check!(inserts_parens_for_function_calls);
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
    }

    #[test]
    fn parens_for_method_call_as_assoc_fn() {
        mark::check!(parens_for_method_call_as_assoc_fn);
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
        mark::check!(suppress_arg_snippets);
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
