use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;
use std::iter;

pub fn provide(providers: &mut Providers) {
    *providers = Providers { assumed_wf_types, ..*providers };
}

fn assumed_wf_types<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &'tcx [(Ty<'tcx>, Span)] {
    match tcx.def_kind(def_id) {
        DefKind::Fn => {
            let sig = tcx.fn_sig(def_id).instantiate_identity();
            let liberated_sig = tcx.liberate_late_bound_regions(def_id.to_def_id(), sig);
            tcx.arena.alloc_from_iter(itertools::zip_eq(
                liberated_sig.inputs_and_output,
                fn_sig_spans(tcx, def_id),
            ))
        }
        DefKind::AssocFn => {
            let sig = tcx.fn_sig(def_id).instantiate_identity();
            let liberated_sig = tcx.liberate_late_bound_regions(def_id.to_def_id(), sig);
            let mut assumed_wf_types: Vec<_> =
                tcx.assumed_wf_types(tcx.local_parent(def_id)).into();
            assumed_wf_types.extend(itertools::zip_eq(
                liberated_sig.inputs_and_output,
                fn_sig_spans(tcx, def_id),
            ));
            tcx.arena.alloc_slice(&assumed_wf_types)
        }
        DefKind::Impl { .. } => {
            // Trait arguments and the self type for trait impls or only the self type for
            // inherent impls.
            let tys = match tcx.impl_trait_ref(def_id) {
                Some(trait_ref) => trait_ref.skip_binder().args.types().collect(),
                None => vec![tcx.type_of(def_id).instantiate_identity()],
            };

            let mut impl_spans = impl_spans(tcx, def_id);
            tcx.arena.alloc_from_iter(tys.into_iter().map(|ty| (ty, impl_spans.next().unwrap())))
        }
        DefKind::AssocConst | DefKind::AssocTy => tcx.assumed_wf_types(tcx.local_parent(def_id)),
        DefKind::OpaqueTy => match tcx.def_kind(tcx.local_parent(def_id)) {
            DefKind::TyAlias => ty::List::empty(),
            DefKind::AssocTy => tcx.assumed_wf_types(tcx.local_parent(def_id)),
            // Nested opaque types only occur in associated types:
            // ` type Opaque<T> = impl Trait<&'static T, AssocTy = impl Nested>; `
            // assumed_wf_types should include those of `Opaque<T>`, `Opaque<T>` itself
            // and `&'static T`.
            DefKind::OpaqueTy => bug!("unimplemented implied bounds for nested opaque types"),
            def_kind => {
                bug!("unimplemented implied bounds for opaque types with parent {def_kind:?}")
            }
        },
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
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::Closure
        | DefKind::Generator => ty::List::empty(),
    }
}

fn fn_sig_spans(tcx: TyCtxt<'_>, def_id: LocalDefId) -> impl Iterator<Item = Span> + '_ {
    let node = tcx.hir().get(tcx.local_def_id_to_hir_id(def_id));
    if let Some(decl) = node.fn_decl() {
        decl.inputs.iter().map(|ty| ty.span).chain(iter::once(decl.output.span()))
    } else {
        bug!("unexpected item for fn {def_id:?}: {node:?}")
    }
}

fn impl_spans(tcx: TyCtxt<'_>, def_id: LocalDefId) -> impl Iterator<Item = Span> + '_ {
    let item = tcx.hir().expect_item(def_id);
    if let hir::ItemKind::Impl(impl_) = item.kind {
        let trait_args = impl_
            .of_trait
            .into_iter()
            .flat_map(|trait_ref| trait_ref.path.segments.last().unwrap().args().args)
            .map(|arg| arg.span());
        let dummy_spans_for_default_args =
            impl_.of_trait.into_iter().flat_map(|trait_ref| iter::repeat(trait_ref.path.span));
        iter::once(impl_.self_ty.span).chain(trait_args).chain(dummy_spans_for_default_args)
    } else {
        bug!("unexpected item for impl {def_id:?}: {item:?}")
    }
}
