//! Call graph metadata

use crate::hir::def_id::{DefId, LOCAL_CRATE};
use crate::ty::subst::SubstsRef;
use crate::ty::{
    ExistentialTraitRef, Instance, InstanceDef, ParamEnv, PolyTraitRef, TraitRef, Ty, TyCtxt,
};
use crate::mir::interpret::{AllocKind, Allocation};

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_macros::HashStable;

use std::sync::Arc;

/// Call graph metadata loaded from an external crate
#[derive(HashStable)]
pub struct ExternCallGraphMetadata<'tcx> {
    /// list of functions casted / coerced into function pointers
    pub function_pointers: Vec<(DefId, SubstsRef<'tcx>)>,

    /// list of types casted / coerced into trait objects
    pub trait_objects: Vec<TraitRef<'tcx>>,

    /// list of types whose drop glue may be invoked by trait object drop glue
    pub dynamic_drop_glue: Vec<(Ty<'tcx>, Vec<ExistentialTraitRef<'tcx>>)>,
}

impl<'tcx> ExternCallGraphMetadata<'tcx> {
    pub fn empty() -> Self {
        ExternCallGraphMetadata {
            function_pointers: vec![],
            trait_objects: vec![],
            dynamic_drop_glue: vec![],
        }
    }
}

/// Local call graph metadata
#[derive(Clone, Default, HashStable)]
pub struct LocalCallGraphMetadata<'tcx> {
    /// list of functions casted / coerced into function pointers
    pub function_pointers: FxHashSet<(DefId, SubstsRef<'tcx>)>,

    /// list of types casted / coerced into trait objects
    pub trait_objects: FxHashSet<TraitRef<'tcx>>,

    /// list of types whose drop glue may be invoked by trait object drop glue
    pub dynamic_drop_glue: FxHashMap<Ty<'tcx>, FxHashSet<ExistentialTraitRef<'tcx>>>,
}

impl<'tcx> LocalCallGraphMetadata<'tcx> {
    /// Registers `alloc` as a const-evaluated trait object
    ///
    /// Returns `false` if `alloc` was *not* a trait object
    pub fn register_const_trait_object(&mut self,
                                       tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       alloc: &'tcx Allocation) -> bool {
        let has_drop_glue = alloc.relocations.values().any(|(_, inner)| {
            let kind = tcx.alloc_map.lock().get(*inner);

            if let Some(AllocKind::Function(instance)) = kind {
                if let InstanceDef::DropGlue(..) = instance.def {
                    return true;
                }
            }

            false
        });

        if has_drop_glue {
            for (_, inner) in alloc.relocations.values() {
                let kind = tcx.alloc_map.lock().get(*inner);

                if let Some(AllocKind::Function(method)) = kind {
                    self.register_dynamic_dispatch(tcx, method);
                } else {
                    bug!("unexpected `AllocKind`: {:?}", kind);
                }
            }
        }

        has_drop_glue
    }

    /// Registers that `method` is dynamically dispatched
    pub fn register_dynamic_dispatch(&mut self,
                                     tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     method: Instance<'tcx>) {
        if let Some((trait_ref, _)) = method.trait_ref_and_method(tcx) {
            self.trait_objects.insert(trait_ref);
        }
    }

    /// Registers that `function` is casted / coerced into a function pointer
    pub fn register_function_pointer(&mut self, function: Instance<'tcx>) {
        self.function_pointers.insert((function.def_id(), function.substs));
    }

    /// Registers that `poly_trait_ref` is used as a trait object
    pub fn register_trait_object(&mut self,
                                 tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 poly_trait_ref: PolyTraitRef<'tcx>) {
        let trait_ref = tcx.normalize_erasing_late_bound_regions(
            ParamEnv::reveal_all(),
            &poly_trait_ref,
        );

        self.trait_objects.insert(trait_ref);

        let self_ty = trait_ref.self_ty();
        let existential_trait_ref = ExistentialTraitRef::erase_self_ty(tcx, trait_ref);

        self.dynamic_drop_glue.entry(self_ty).or_default().insert(existential_trait_ref);
    }

    pub fn extend(&mut self, other: Self) {
        self.function_pointers.extend(other.function_pointers);
        self.trait_objects.extend(other.trait_objects);

        for (ty, traits) in other.dynamic_drop_glue {
            self.dynamic_drop_glue.entry(ty).or_default().extend(traits);
        }
    }
}

/// Local and external call graph metadata
#[derive(HashStable)]
pub struct AllCallGraphMetadata<'tcx> {
    /// list of functions casted / coerced into function pointers
    pub function_pointers: FxHashSet<(DefId, SubstsRef<'tcx>)>,

    /// list of types casted / coerced into trait objects
    pub trait_objects: FxHashSet<TraitRef<'tcx>>,

    /// list of types whose drop glue may be invoked by trait object drop glue
    pub dynamic_drop_glue: FxHashMap<Ty<'tcx>, Arc<FxHashSet<ExistentialTraitRef<'tcx>>>>,
}

pub fn collect<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> AllCallGraphMetadata<'tcx> {
    let mut cg = (*tcx.local_call_graph_metadata(LOCAL_CRATE)).clone();

    for &cnum in tcx.crates().iter() {
        let ext = tcx.extern_call_graph_metadata(cnum);

        cg.function_pointers.extend(ext.function_pointers.iter().cloned());
        cg.trait_objects.extend(ext.trait_objects.iter().cloned());

        for (ty, traits) in &ext.dynamic_drop_glue {
            cg.dynamic_drop_glue.entry(*ty).or_default().extend(traits.iter().cloned());
        }
    }

    AllCallGraphMetadata {
        function_pointers: cg.function_pointers,
        trait_objects: cg.trait_objects,
        dynamic_drop_glue: cg.dynamic_drop_glue.into_iter().map(|(ty, traits)| {
            (ty, Arc::new(traits))
        }).collect(),
    }
}
