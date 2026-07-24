//! Collects trait impls for each item in the crate. For example, if a crate
//! defines a struct that implements a trait, this pass will note that the
//! struct implements that trait.

use rustc_data_structures::fx::FxHashSet;
use rustc_errors::FatalError;
use rustc_hir::attrs::{AttributeKind, DocAttribute};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::{Attribute, find_attr};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::kw;

use super::Pass;
use crate::clean::*;
use crate::core::DocContext;
use crate::formats::cache::Cache;
use crate::visit::DocVisitor;

pub(crate) const COLLECT_TRAIT_IMPLS: Pass = Pass {
    name: "collect-trait-impls",
    run: Some(collect_trait_impls),
    description: "retrieves trait impls for items in the crate",
};

pub(crate) fn collect_trait_impls(mut krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    let tcx = cx.tcx;
    // We need to check if there are errors before running this pass because it would crash when
    // we try to get auto and blanket implementations.
    if tcx.dcx().has_errors().is_some() {
        return krate;
    }

    let synth_impls = cx.sess().time("collect_synthetic_impls", || {
        let mut synth = SyntheticImplCollector { cx, impls: Vec::new() };
        synth.visit_crate(&krate);
        synth.impls
    });

    let crate_items = {
        let mut coll = ItemAndAliasCollector::new(&cx.cache);
        cx.sess().time("collect_items_for_trait_impls", || coll.visit_crate(&krate));
        coll.items
    };

    let mut new_items_external = Vec::new();
    let mut new_items_local = Vec::new();

    // External trait impls.
    {
        let _prof_timer = tcx.sess.prof.generic_activity("build_extern_trait_impls");
        for &cnum in tcx.crates(()) {
            for &impl_def_id in tcx.trait_impls_in_crate(cnum) {
                let opt_trait_ref = tcx.impl_opt_trait_ref(impl_def_id);
                if opt_trait_ref.is_some_and(|trait_ref| {
                    crate_items.contains(&ItemId::DefId(trait_ref.def_id()))
                        || Some(trait_ref.def_id()) == tcx.lang_items().deref_trait()
                        || tcx.is_doc_notable_trait(trait_ref.def_id())
                }) {
                    cx.with_param_env(impl_def_id, |cx| {
                        inline::build_impl(cx, impl_def_id, None, &mut new_items_external);
                    });
                } else {
                    let self_ty = tcx.type_of(impl_def_id).instantiate_identity().skip_norm_wip();
                    let self_ty_head =
                        SelfTyHead::of(ty::Binder::dummy(self_ty), tcx, Some(impl_def_id));
                    let keep_impl = match self_ty_head {
                        SelfTyHead::Generic => true,
                        SelfTyHead::Item(def_id) => crate_items.contains(&ItemId::DefId(def_id)),
                        SelfTyHead::Primitive | SelfTyHead::Other => false,
                    };
                    if keep_impl {
                        cx.with_param_env(impl_def_id, |cx| {
                            inline::build_impl(cx, impl_def_id, None, &mut new_items_external);
                        });
                    }
                }
            }
        }
    }

    // Local trait impls.
    {
        let _prof_timer = tcx.sess.prof.generic_activity("build_local_trait_impls");
        let mut attr_buf = Vec::new();
        for &impl_def_id in tcx.trait_impls_in_crate(LOCAL_CRATE) {
            let mut parent = Some(tcx.parent(impl_def_id));
            while let Some(did) = parent {
                attr_buf.extend(find_attr!(tcx, did, Doc(d) if !d.cfg.is_empty() => {
                    let mut new_attr = DocAttribute::default();
                    new_attr.cfg = d.cfg.clone();
                    Attribute::Parsed(AttributeKind::Doc(Box::new(new_attr)))
                }));
                parent = tcx.opt_parent(did);
            }
            cx.with_param_env(impl_def_id, |cx| {
                inline::build_impl(cx, impl_def_id, Some((&attr_buf, None)), &mut new_items_local);
            });
            attr_buf.clear();
        }
    }

    tcx.sess.prof.generic_activity("build_primitive_trait_impls").run(|| {
        for (prim, did) in PrimitiveType::primitive_locations(tcx) {
            // Do not calculate blanket impl list for docs that are not going to be rendered.
            // While the `impl` blocks themselves are only in `libcore`, the module with `doc`
            // attached is directly included in `libstd` as well.
            if did.is_local() {
                for impl_def_id in prim.impls(tcx) {
                    // Try to inline primitive impls from other crates.
                    if !impl_def_id.is_local() {
                        cx.with_param_env(impl_def_id, |cx| {
                            inline::build_impl(cx, impl_def_id, None, &mut new_items_external);
                        });
                    }
                }

                // HACK: this is all one massive hack that is very hard to get rid of (see comment below)
                for def_id in prim.impls(tcx).filter(|&def_id| {
                    // Avoid including impl blocks with filled-in generics.
                    // https://github.com/rust-lang/rust/issues/94937
                    //
                    // FIXME(notriddle): https://github.com/rust-lang/rust/issues/97129
                    //
                    // This tactic of using inherent impl blocks for getting
                    // auto traits and blanket impls is a hack. What we really
                    // want is to check if `[T]` impls `Send`, which has
                    // nothing to do with the inherent impl.
                    //
                    // Rustdoc currently uses these `impl` block as a source of
                    // the `Ty`, as well as the `ParamEnv`, `GenericArgsRef`, and
                    // `Generics`. To avoid relying on the `impl` block, these
                    // things would need to be created from wholecloth, in a
                    // form that is valid for use in type inference.
                    let ty = tcx.type_of(def_id).instantiate_identity().skip_norm_wip();
                    match ty.kind() {
                        ty::Slice(ty) | ty::Ref(_, ty, _) | ty::RawPtr(ty, _) => {
                            matches!(ty.kind(), ty::Param(..))
                        }
                        ty::Tuple(tys) => tys.iter().all(|ty| matches!(ty.kind(), ty::Param(..))),
                        _ => true,
                    }
                }) {
                    let impls = synthesize_auto_trait_and_blanket_impls(cx, def_id);
                    new_items_external.extend(impls.filter(|i| cx.inlined.insert(i.item_id)));
                }
            }
        }
    });

    if let ModuleItem(Module { items, .. }) = &mut krate.module.inner.kind {
        items.extend(synth_impls);
        items.extend(new_items_external);
        items.extend(new_items_local);
    } else {
        panic!("collect-trait-impls can't run");
    };

    krate.external_traits.extend(cx.external_traits.drain(..));

    krate
}

enum SelfTyHead {
    Generic,
    Primitive,
    Item(DefId),
    Other,
}

impl SelfTyHead {
    fn of<'tcx>(
        bound_ty: ty::Binder<'tcx, Ty<'tcx>>,
        tcx: TyCtxt<'tcx>,
        parent: Option<DefId>,
    ) -> Self {
        match *bound_ty.skip_binder().kind() {
            ty::Never => Self::Primitive,
            ty::Bool => Self::Primitive,
            ty::Char => Self::Primitive,
            ty::Int(..) => Self::Primitive,
            ty::Uint(..) => Self::Primitive,
            ty::Float(..) => Self::Primitive,
            ty::Str => Self::Primitive,
            ty::Slice(..) => Self::Primitive,
            ty::Pat(ty, _) => Self::of(bound_ty.rebind(ty), tcx, parent),
            ty::Array(..) => Self::Primitive,
            ty::RawPtr(..) => Self::Primitive,
            ty::Ref(_, ty, _) => match Self::of(bound_ty.rebind(ty), tcx, parent) {
                Self::Generic => Self::Primitive,
                head => head,
            },
            ty::FnDef(..) | ty::FnPtr(..) => Self::Primitive,
            ty::UnsafeBinder(_) => Self::Other,
            ty::Adt(def, _) => Self::Item(def.did()),
            ty::Foreign(did) => Self::Item(did),
            ty::Dynamic(obj, _) => {
                // HACK: pick the first `did` as the `did` of the trait object. Someone
                // might want to implement "native" support for marker-trait-only
                // trait objects.
                let mut dids = obj.auto_traits();
                let did = obj
                    .principal_def_id()
                    .or_else(|| dids.next())
                    .unwrap_or_else(|| panic!("found trait object `{obj:?}` with no traits?"));
                Self::Item(did)
            }
            ty::Tuple(_) => Self::Primitive,

            ty::Alias(_, alias_ty @ ty::AliasTy { kind: ty::Projection { def_id }, .. }) => {
                if tcx.is_impl_trait_in_trait(def_id) {
                    Self::Other
                } else {
                    Self::of(bound_ty.rebind(alias_ty.self_ty()), tcx, parent)
                }
            }

            ty::Alias(_, alias_ty @ ty::AliasTy { kind: ty::Inherent { .. }, .. }) => {
                let alias_ty = bound_ty.rebind(alias_ty);
                Self::of(alias_ty.map_bound(|ty| ty.self_ty()), tcx, parent)
            }

            ty::Alias(_, ty::AliasTy { kind: ty::Free { def_id }, args, .. }) => {
                if tcx.features().checked_type_aliases() {
                    // Free type alias `data` represents the `type X` in `type X = Y`. If we need `Y`,
                    // we need to use `type_of`.
                    Self::Item(def_id)
                } else {
                    let ty = tcx.type_of(def_id).instantiate(tcx, args).skip_norm_wip();
                    Self::of(bound_ty.rebind(ty), tcx, parent)
                }
            }

            ty::Param(ref p) => {
                // FIXME: there's a slight behavior difference from clean_middle_ty here
                // since here we represent impl traits as Generic not ImplTrait.
                // probably doesn't matter for collect trait impls since impl trait
                // can't be a self ty
                if p.name == kw::SelfUpper { Self::Other } else { Self::Generic }
            }

            ty::Bound(_, ref ty) => match ty.kind {
                ty::BoundTyKind::Param(_) => Self::Generic,
                ty::BoundTyKind::Anon => panic!("unexpected anonymous bound type variable"),
            },

            ty::Alias(_, ty::AliasTy { kind: ty::Opaque { .. }, .. }) => {
                panic!("should not appear as impl self ty")
            }

            ty::Closure(..) => panic!("Closure"),
            ty::CoroutineClosure(..) => panic!("CoroutineClosure"),
            ty::Coroutine(..) => panic!("Coroutine"),
            ty::Placeholder(..) => panic!("Placeholder"),
            ty::CoroutineWitness(..) => panic!("CoroutineWitness"),
            ty::Infer(..) => panic!("Infer"),

            ty::Error(_) => FatalError.raise(),
        }
    }
}

struct SyntheticImplCollector<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
    impls: Vec<Item>,
}

impl DocVisitor<'_> for SyntheticImplCollector<'_, '_> {
    fn visit_item(&mut self, i: &Item) {
        if i.is_struct() || i.is_enum() || i.is_union() {
            let item_def_id = i.item_id.expect_def_id();
            // FIXME(eddyb) is this `doc(hidden)` check needed?
            // FIXME(camelid) should we skip the `doc(hidden)` check if --document-hidden-items is passed?
            if (self.cx.document_private()
                || self.cx.cache.effective_visibilities.is_reachable(self.cx.tcx, item_def_id))
                && !self.cx.tcx.is_doc_hidden(item_def_id)
            {
                self.impls.extend(synthesize_auto_trait_and_blanket_impls(self.cx, item_def_id));
            }
        }

        self.visit_item_recur(i)
    }
}

struct ItemAndAliasCollector<'cache> {
    items: FxHashSet<ItemId>,
    cache: &'cache Cache,
}

impl<'cache> ItemAndAliasCollector<'cache> {
    fn new(cache: &'cache Cache) -> Self {
        ItemAndAliasCollector { items: FxHashSet::default(), cache }
    }
}

impl DocVisitor<'_> for ItemAndAliasCollector<'_> {
    fn visit_item(&mut self, i: &Item) {
        self.items.insert(i.item_id);

        if let TypeAliasItem(alias) = &i.inner.kind
            && let Some(did) = alias.type_.def_id(self.cache)
        {
            self.items.insert(ItemId::DefId(did));
        }

        self.visit_item_recur(i)
    }
}
