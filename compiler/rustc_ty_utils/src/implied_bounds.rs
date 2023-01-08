use crate::rustc_middle::ty::DefIdTree;
use rustc_hir::{self as hir, def::DefKind, def_id::DefId};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::{Span, DUMMY_SP};

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers { assumed_wf_types, ..*providers };
}

fn assumed_wf_types(tcx: TyCtxt<'_>, def_id: DefId) -> &[(Ty<'_>, Span)] {
    match tcx.def_kind(def_id) {
        DefKind::Fn => {
            let sig = tcx.fn_sig(def_id);
            let liberated_sig = tcx.liberate_late_bound_regions(def_id, sig);
            if let Some(node) = tcx.hir().get_if_local(def_id)
                && let Some(decl) = node.fn_decl()
            {
                assert_eq!(decl.inputs.len(), liberated_sig.inputs().len());
                tcx.arena.alloc_from_iter(std::iter::zip(
                    liberated_sig.inputs_and_output,
                    decl.inputs.iter().map(|ty| ty.span).chain([decl.output.span()]),
                ))
            } else {
                tcx.arena.alloc_from_iter(
                    liberated_sig.inputs_and_output.iter().map(|ty| (ty, DUMMY_SP)),
                )
            }
        }
        DefKind::AssocFn => {
            let sig = tcx.fn_sig(def_id);
            let liberated_sig = tcx.liberate_late_bound_regions(def_id, sig);
            let assumed_wf_types = tcx.assumed_wf_types(tcx.parent(def_id));
            if let Some(node) = tcx.hir().get_if_local(def_id)
                && let Some(decl) = node.fn_decl()
            {
                assert_eq!(decl.inputs.len(), liberated_sig.inputs().len());
                tcx.arena.alloc_from_iter(assumed_wf_types.iter().copied().chain(std::iter::zip(
                    liberated_sig.inputs_and_output,
                    decl.inputs.iter().map(|ty| ty.span).chain([decl.output.span()]),
                )))
            } else {
                tcx.arena.alloc_from_iter(assumed_wf_types.iter().copied().chain(
                    liberated_sig.inputs_and_output.iter().map(|ty| (ty, DUMMY_SP)),
                ))
            }
        }
        DefKind::Impl => match tcx.impl_trait_ref(def_id) {
            Some(trait_ref) => {
                let types: Vec<_> = trait_ref.substs.types().collect();
                let self_span = if let Some(hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Impl(impl_),
                    ..
                })) = tcx.hir().get_if_local(def_id)
                {
                    impl_.self_ty.span
                } else {
                    DUMMY_SP
                };
                tcx.arena.alloc_from_iter(std::iter::zip(
                    types,
                    // FIXME: reliable way of getting trait ref substs...
                    [self_span].into_iter().chain(std::iter::repeat(DUMMY_SP)),
                ))
            }
            // Only the impl self type
            None => {
                let span = if let Some(hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Impl(impl_),
                    ..
                })) = tcx.hir().get_if_local(def_id)
                {
                    impl_.self_ty.span
                } else {
                    DUMMY_SP
                };
                tcx.arena.alloc_from_iter([(tcx.type_of(def_id), span)])
            }
        },
        DefKind::AssocConst | DefKind::AssocTy => tcx.assumed_wf_types(tcx.parent(def_id)),
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::TyParam
        | DefKind::Const
        | DefKind::ConstParam
        | DefKind::Static(_)
        | DefKind::Ctor(_, _)
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::OpaqueTy
        | DefKind::ImplTraitPlaceholder
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::Closure
        | DefKind::Generator => &[],
    }
}
