//! This modules takes care of rendering various defenitions as completion items.
use test_utils::tested_by;
use hir::Docs;

use crate::completion::{
    Completions, CompletionKind, CompletionItemKind, CompletionContext, CompletionItem,
    function_label,
};

impl Completions {
    pub(crate) fn add_field(
        &mut self,
        kind: CompletionKind,
        ctx: &CompletionContext,
        field: hir::StructField,
        substs: &hir::Substs,
    ) {
        CompletionItem::new(kind, ctx.source_range(), field.name(ctx.db).to_string())
            .kind(CompletionItemKind::Field)
            .detail(field.ty(ctx.db).subst(substs).to_string())
            .set_documentation(field.docs(ctx.db))
            .add_to(self);
    }

    pub(crate) fn add_pos_field(
        &mut self,
        kind: CompletionKind,
        ctx: &CompletionContext,
        field: usize,
        ty: &hir::Ty,
    ) {
        CompletionItem::new(kind, ctx.source_range(), field.to_string())
            .kind(CompletionItemKind::Field)
            .detail(ty.to_string())
            .add_to(self);
    }

    pub(crate) fn add_function(
        &mut self,
        kind: CompletionKind,
        ctx: &CompletionContext,
        func: hir::Function,
    ) {
        let sig = func.signature(ctx.db);

        let mut builder = CompletionItem::new(kind, ctx.source_range(), sig.name().to_string())
            .kind(if sig.has_self_param() {
                CompletionItemKind::Method
            } else {
                CompletionItemKind::Function
            })
            .set_documentation(func.docs(ctx.db))
            .set_detail(function_item_label(ctx, func));
        // If not an import, add parenthesis automatically.
        if ctx.use_item_syntax.is_none() && !ctx.is_call {
            tested_by!(inserts_parens_for_function_calls);
            let snippet =
                if sig.params().is_empty() || sig.has_self_param() && sig.params().len() == 1 {
                    format!("{}()$0", sig.name())
                } else {
                    format!("{}($0)", sig.name())
                };
            builder = builder.insert_snippet(snippet);
        }
        self.add(builder)
    }
}

fn function_item_label(ctx: &CompletionContext, function: hir::Function) -> Option<String> {
    let node = function.source(ctx.db).1;
    function_label(&node)
}
