use hir::{HasSource, Type};
use syntax::{ast::Fn, display::function_declaration};

use crate::{
    item::{CompletionItem, CompletionItemKind, CompletionKind},
    render::{builder_ext::Params, RenderContext},
};

#[derive(Debug)]
pub(crate) struct FunctionRender<'a> {
    ctx: RenderContext<'a>,
    name: String,
    fn_: hir::Function,
    ast_node: Fn,
}

impl<'a> FunctionRender<'a> {
    pub(crate) fn new(
        ctx: RenderContext<'a>,
        local_name: Option<String>,
        fn_: hir::Function,
    ) -> FunctionRender<'a> {
        let name = local_name.unwrap_or_else(|| fn_.name(ctx.db()).to_string());
        let ast_node = fn_.source(ctx.db()).value;

        FunctionRender { ctx, name, fn_, ast_node }
    }

    pub(crate) fn render(self) -> CompletionItem {
        let params = self.params();
        CompletionItem::new(CompletionKind::Reference, self.ctx.source_range(), self.name.clone())
            .kind(self.kind())
            .set_documentation(self.ctx.docs(self.fn_))
            .set_deprecated(self.ctx.is_deprecated(self.fn_))
            .detail(self.detail())
            .add_call_parens(self.ctx.completion, self.name, params)
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
        let params_ty = self.fn_.params(self.ctx.db());
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
        if self.fn_.self_param(self.ctx.db()).is_some() {
            CompletionItemKind::Method
        } else {
            CompletionItemKind::Function
        }
    }
}
