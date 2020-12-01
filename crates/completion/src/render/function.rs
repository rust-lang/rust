//! Renderer for function calls.

use hir::{HasSource, Type};
use syntax::{ast::Fn, display::function_declaration};

use crate::{
    item::{CompletionItem, CompletionItemKind, CompletionKind, ImportToAdd},
    render::{builder_ext::Params, RenderContext},
};

pub(crate) fn render_fn<'a>(
    ctx: RenderContext<'a>,
    import_to_add: Option<ImportToAdd>,
    local_name: Option<String>,
    fn_: hir::Function,
) -> CompletionItem {
    let _p = profile::span("render_fn");
    FunctionRender::new(ctx, local_name, fn_).render(import_to_add)
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
    ) -> FunctionRender<'a> {
        let name = local_name.unwrap_or_else(|| fn_.name(ctx.db()).to_string());
        let ast_node = fn_.source(ctx.db()).value;

        FunctionRender { ctx, name, func: fn_, ast_node }
    }

    fn render(self, import_to_add: Option<ImportToAdd>) -> CompletionItem {
        let params = self.params();
        CompletionItem::new(CompletionKind::Reference, self.ctx.source_range(), self.name.clone())
            .kind(self.kind())
            .set_documentation(self.ctx.docs(self.func))
            .set_deprecated(self.ctx.is_deprecated(self.func))
            .detail(self.detail())
            .add_call_parens(self.ctx.completion, self.name, params)
            .add_import(import_to_add)
            .build()
    }

    fn detail(&self) -> String {
        function_declaration(&self.ast_node)
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
        let params_ty = if self.ctx.completion.dot_receiver.is_some() {
            self.func.method_params(self.ctx.db()).unwrap_or_default()
        } else {
            self.func.assoc_fn_params(self.ctx.db())
        };
        let params = self
            .ast_node
            .param_list()
            .into_iter()
            .flat_map(|it| it.params())
            .zip(params_ty)
            .flat_map(|(it, param_ty)| {
                if let Some(pat) = it.pat() {
                    let name = pat.to_string();
                    let arg = name.trim_start_matches("mut ").trim_start_matches('_');
                    return Some(self.add_arg(arg, param_ty.ty()));
                }
                None
            })
            .collect();
        Params::Named(params)
    }

    fn kind(&self) -> CompletionItemKind {
        if self.func.self_param(self.ctx.db()).is_some() {
            CompletionItemKind::Method
        } else {
            CompletionItemKind::Function
        }
    }
}

#[cfg(test)]
mod tests {
    use test_utils::mark;

    use crate::{
        test_utils::{check_edit, check_edit_with_config},
        CompletionConfig,
    };

    #[test]
    fn inserts_parens_for_function_calls() {
        mark::check!(inserts_parens_for_function_calls);
        check_edit(
            "no_args",
            r#"
fn no_args() {}
fn main() { no_<|> }
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
fn main() { with_<|> }
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
fn bar(s: &S) { s.f<|> }
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
    s.f<|>
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
    fn suppress_arg_snippets() {
        mark::check!(suppress_arg_snippets);
        check_edit_with_config(
            CompletionConfig { add_call_argument_snippets: false, ..CompletionConfig::default() },
            "with_args",
            r#"
fn with_args(x: i32, y: String) {}
fn main() { with_<|> }
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
fn main() { f<|> }
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
    ref_ar<|>
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
    ref_ar<|>
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
    y.<|>
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
    take_m<|>
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
