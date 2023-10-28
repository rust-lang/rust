//! This module contains helpers for walking all types of
//! a signature, while preserving spans as much as possible

use std::ops::ControlFlow;

use rustc_hir::{def::DefKind, def_id::LocalDefId};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::Span;
use rustc_type_ir::visit::TypeVisitable;

pub trait SpannedTypeVisitor<'tcx> {
    type BreakTy = !;
    fn visit(
        &mut self,
        span: Span,
        value: impl TypeVisitable<TyCtxt<'tcx>>,
    ) -> ControlFlow<Self::BreakTy>;
}

pub fn walk_types<'tcx, V: SpannedTypeVisitor<'tcx>>(
    tcx: TyCtxt<'tcx>,
    item: LocalDefId,
    visitor: &mut V,
) -> ControlFlow<V::BreakTy> {
    let kind = tcx.def_kind(item);
    trace!(?kind);
    match kind {
        DefKind::Coroutine => {
            match tcx.type_of(item).instantiate_identity().kind() {
                ty::Coroutine(_, args, _) => visitor.visit(tcx.def_span(item), args.as_coroutine().sig())?,
                _ => bug!(),
            }
            for (pred, span) in tcx.predicates_of(item).instantiate_identity(tcx) {
                visitor.visit(span, pred)?;
            }
        }
        // Walk over the signature of the function-like
        DefKind::Closure | DefKind::AssocFn | DefKind::Fn => {
            let ty_sig = match kind {
                DefKind::Closure => match tcx.type_of(item).instantiate_identity().kind() {
                    ty::Closure(_, args) => args.as_closure().sig(),
                    _ => bug!(),
                },
                _ => tcx.fn_sig(item).instantiate_identity(),
            };
            let hir_sig = tcx.hir().get_by_def_id(item).fn_decl().unwrap();
            // Walk over the inputs and outputs manually in order to get good spans for them.
            visitor.visit(hir_sig.output.span(), ty_sig.output());
            for (hir, ty) in hir_sig.inputs.iter().zip(ty_sig.inputs().iter()) {
                visitor.visit(hir.span, ty.map_bound(|x| *x))?;
            }
            for (pred, span) in tcx.predicates_of(item).instantiate_identity(tcx) {
                visitor.visit(span, pred)?;
            }
        }
        // Walk over the type behind the alias
        DefKind::TyAlias {..} | DefKind::AssocTy |
        // Walk over the type of the item
        DefKind::Static(_) | DefKind::Const | DefKind::AssocConst | DefKind::AnonConst => {
            let span = match tcx.hir().get_by_def_id(item).ty() {
                Some(ty) => ty.span,
                _ => tcx.def_span(item),
            };
            visitor.visit(span,  tcx.type_of(item).instantiate_identity());
            for (pred, span) in tcx.predicates_of(item).instantiate_identity(tcx) {
                visitor.visit(span, pred)?;
            }
        }
        DefKind::OpaqueTy => {
            for (pred, span) in tcx.explicit_item_bounds(item).instantiate_identity_iter_copied() {
                visitor.visit(span, pred)?;
            }
        }
        // Look at field types
        DefKind::Struct | DefKind::Union | DefKind::Enum => {
            let span = tcx.def_ident_span(item).unwrap();
            visitor.visit(span,  tcx.type_of(item).instantiate_identity());
            for (pred, span) in tcx.predicates_of(item).instantiate_identity(tcx) {
                visitor.visit(span, pred)?;
            }
        }
        // Does not have a syntactical signature
        DefKind::InlineConst => {}
        DefKind::Impl { of_trait } => {
            if of_trait {
                let span = tcx.hir().get_by_def_id(item).expect_item().expect_impl().of_trait.unwrap().path.span;
                let args = &tcx.impl_trait_ref(item).unwrap().instantiate_identity().args[1..];
                visitor.visit(span, args)?;
            }
            let span = match tcx.hir().get_by_def_id(item).ty() {
                Some(ty) => ty.span,
                _ => tcx.def_span(item),
            };
            visitor.visit(span, tcx.type_of(item).instantiate_identity());
            for (pred, span) in tcx.predicates_of(item).instantiate_identity(tcx) {
                visitor.visit(span, pred)?;
            }}
        DefKind::Trait => {
            for (pred, span) in tcx.predicates_of(item).instantiate_identity(tcx) {
                visitor.visit(span, pred)?;
            }
        }
        DefKind::TraitAlias => {
            for (pred, span) in tcx.predicates_of(item).instantiate_identity(tcx) {
                visitor.visit(span, pred)?;
            }
        }
        | DefKind::Variant
        | DefKind::ForeignTy
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Ctor(_, _)
        | DefKind::Field
        | DefKind::LifetimeParam => {
            span_bug!(
                tcx.def_span(item),
                "{kind:?} has not seen any uses of `walk_types` yet, ping oli-obk if you'd like any help"
            )
        }
        // These don't have any types.
        | DefKind::ExternCrate
        | DefKind::ForeignMod
        | DefKind::Macro(_)
        | DefKind::GlobalAsm
        | DefKind::Mod
        | DefKind::Use => {}
    }
    ControlFlow::Continue(())
}
