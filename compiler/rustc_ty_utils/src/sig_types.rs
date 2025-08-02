//! This module contains helpers for walking all types of
//! a signature, while preserving spans as much as possible

use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, TyCtxt, TypeVisitable, VisitorResult, try_visit};
use rustc_span::Span;
use tracing::{instrument, trace};

pub trait SpannedTypeVisitor<'tcx> {
    type Result: VisitorResult = ();
    fn visit(&mut self, span: Span, value: impl TypeVisitable<TyCtxt<'tcx>>) -> Self::Result;
}

#[instrument(level = "trace", skip(tcx, visitor))]
pub fn walk_types<'tcx, V: SpannedTypeVisitor<'tcx>>(
    tcx: TyCtxt<'tcx>,
    item: LocalDefId,
    visitor: &mut V,
) -> V::Result {
    let kind = tcx.def_kind(item);
    trace!(?kind);
    match kind {
        // Walk over the signature of the function
        DefKind::AssocFn | DefKind::Fn => {
            let hir_sig = tcx.hir_node_by_def_id(item).fn_decl().unwrap();
            // If the type of the item uses `_`, we're gonna error out anyway, but
            // typeck (which type_of invokes below), will call back into opaque_types_defined_by
            // causing a cycle. So we just bail out in this case.
            if hir_sig.output.is_suggestable_infer_ty().is_some() {
                return V::Result::output();
            }
            let ty_sig = tcx.fn_sig(item).instantiate_identity();
            // Walk over the inputs and outputs manually in order to get good spans for them.
            try_visit!(visitor.visit(hir_sig.output.span(), ty_sig.output()));
            for (hir, ty) in hir_sig.inputs.iter().zip(ty_sig.inputs().iter()) {
                try_visit!(visitor.visit(hir.span, ty.map_bound(|x| *x)));
            }
            for (pred, span) in tcx.explicit_predicates_of(item).instantiate_identity(tcx) {
                try_visit!(visitor.visit(span, pred));
            }
        }
        // Walk over the type behind the alias
        DefKind::TyAlias { .. } | DefKind::AssocTy |
        // Walk over the type of the item
        DefKind::Static { .. } | DefKind::Const | DefKind::AssocConst | DefKind::AnonConst => {
            if let Some(ty) = tcx.hir_node_by_def_id(item).ty() {
                // If the type of the item uses `_`, we're gonna error out anyway, but
                // typeck (which type_of invokes below), will call back into opaque_types_defined_by
                // causing a cycle. So we just bail out in this case.
                if ty.is_suggestable_infer_ty() {
                    return V::Result::output();
                }
                // Associated types in traits don't necessarily have a type that we can visit
                try_visit!(visitor.visit(ty.span, tcx.type_of(item).instantiate_identity()));
            }
            for (pred, span) in tcx.explicit_predicates_of(item).instantiate_identity(tcx) {
                try_visit!(visitor.visit(span, pred));
            }
        }
        DefKind::OpaqueTy => {
            for (pred, span) in tcx.explicit_item_bounds(item).iter_identity_copied() {
                try_visit!(visitor.visit(span, pred));
            }
        }
        // Look at field types
        DefKind::Struct | DefKind::Union | DefKind::Enum => {
            let span = tcx.def_ident_span(item).unwrap();
            let ty = tcx.type_of(item).instantiate_identity();
            try_visit!(visitor.visit(span, ty));
            let ty::Adt(def, args) = ty.kind() else {
                span_bug!(span, "invalid type for {kind:?}: {:#?}", ty.kind())
            };
            for field in def.all_fields() {
                let span = tcx.def_ident_span(field.did).unwrap();
                let ty = field.ty(tcx, args);
                try_visit!(visitor.visit(span, ty));
            }
            for (pred, span) in tcx.explicit_predicates_of(item).instantiate_identity(tcx) {
                try_visit!(visitor.visit(span, pred));
            }
        }
        // These are not part of a public API, they can only appear as hidden types, and there
        // the interesting parts are solely in the signature of the containing item's opaque type
        // or dyn type.
        DefKind::InlineConst | DefKind::Closure | DefKind::SyntheticCoroutineBody => {}
        DefKind::Impl { of_trait } => {
            if of_trait {
                let span = tcx.hir_node_by_def_id(item).expect_item().expect_impl().of_trait.unwrap().trait_ref.path.span;
                let args = &tcx.impl_trait_ref(item).unwrap().instantiate_identity().args[1..];
                try_visit!(visitor.visit(span, args));
            }
            let span = match tcx.hir_node_by_def_id(item).ty() {
                Some(ty) => ty.span,
                _ => tcx.def_span(item),
            };
            try_visit!(visitor.visit(span, tcx.type_of(item).instantiate_identity()));
            for (pred, span) in tcx.explicit_predicates_of(item).instantiate_identity(tcx) {
                try_visit!(visitor.visit(span, pred));
            }
        }
        DefKind::TraitAlias | DefKind::Trait => {
            for (pred, span) in tcx.explicit_predicates_of(item).instantiate_identity(tcx) {
                try_visit!(visitor.visit(span, pred));
            }
        }
        | DefKind::Variant
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
        // These don't have any types, but are visited during privacy checking.
        | DefKind::ExternCrate
        | DefKind::ForeignMod
        | DefKind::ForeignTy
        | DefKind::Macro(_)
        | DefKind::GlobalAsm
        | DefKind::Mod
        | DefKind::Use => {}
    }
    V::Result::output()
}
