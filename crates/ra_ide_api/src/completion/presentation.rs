//! This modules takes care of rendering various defenitions as completion items.
use hir::Docs;

use crate::completion::{Completions, CompletionKind, CompletionItemKind, CompletionContext, CompletionItem};

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
}
