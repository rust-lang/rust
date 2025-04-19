use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::unord::UnordSet;
use rustc_hir::def_id::DefId;
use rustc_span::Span;
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

use crate::ty::{self, Ty, TyCtxt};

/// An additional filter to exclude well-known types from the ecosystem
/// because their drops are trivial.
/// This returns additional types to check if the drops are delegated to those.
/// A typical example is `hashbrown::HashMap<K, V>`, whose drop is delegated to `K` and `V`.
fn true_significant_drop_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<SmallVec<[Ty<'tcx>; 2]>> {
    if let ty::Adt(def, args) = ty.kind() {
        let mut did = def.did();
        let mut name_rev = vec![];
        loop {
            let key = tcx.def_key(did);

            match key.disambiguated_data.data {
                rustc_hir::definitions::DefPathData::CrateRoot => {
                    name_rev.push(tcx.crate_name(did.krate));
                }
                rustc_hir::definitions::DefPathData::TypeNs(symbol) => {
                    name_rev.push(symbol);
                }
                _ => return None,
            }
            if let Some(parent) = key.parent {
                did = DefId { krate: did.krate, index: parent };
            } else {
                break;
            }
        }
        let name_str: Vec<_> = name_rev.iter().rev().map(|x| x.as_str()).collect();
        debug!(?name_str);
        match name_str[..] {
            // These are the types from Rust core ecosystem
            ["syn" | "proc_macro2", ..]
            | ["core" | "std", "task", "LocalWaker" | "Waker"]
            | ["core" | "std", "task", "wake", "LocalWaker" | "Waker"] => Some(smallvec![]),
            // These are important types from Rust ecosystem
            ["tracing", "instrument", "Instrumented"] | ["bytes", "Bytes"] => Some(smallvec![]),
            ["hashbrown", "raw", "RawTable" | "RawIntoIter"] => {
                if let [ty, ..] = &***args
                    && let Some(ty) = ty.as_type()
                {
                    Some(smallvec![ty])
                } else {
                    None
                }
            }
            ["hashbrown", "raw", "RawDrain"] => {
                if let [_, ty, ..] = &***args
                    && let Some(ty) = ty.as_type()
                {
                    Some(smallvec![ty])
                } else {
                    None
                }
            }
            _ => None,
        }
    } else {
        None
    }
}

/// Returns the list of types with a "potentially sigificant" that may be dropped
/// by dropping a value of type `ty`.
#[instrument(level = "trace", skip(tcx, typing_env))]
pub fn extract_component_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
    ty_seen: &mut UnordSet<Ty<'tcx>>,
) -> SmallVec<[Ty<'tcx>; 4]> {
    // Droppiness does not depend on regions, so let us erase them.
    let ty = tcx.try_normalize_erasing_regions(typing_env, ty).unwrap_or(ty);

    let tys = tcx.list_significant_drop_tys(typing_env.as_query_input(ty));
    debug!(?ty, "components");
    let mut out_tys = smallvec![];
    for ty in tys {
        if let Some(tys) = true_significant_drop_ty(tcx, ty) {
            // Some types can be further opened up because the drop is simply delegated
            for ty in tys {
                if ty_seen.insert(ty) {
                    out_tys.extend(extract_component_raw(tcx, typing_env, ty, ty_seen));
                }
            }
        } else {
            if ty_seen.insert(ty) {
                out_tys.push(ty);
            }
        }
    }
    out_tys
}

#[instrument(level = "trace", skip(tcx, typing_env))]
pub fn extract_component_with_significant_dtor<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> SmallVec<[Ty<'tcx>; 4]> {
    let mut tys = extract_component_raw(tcx, typing_env, ty, &mut Default::default());
    let mut deduplicate = FxHashSet::default();
    tys.retain(|oty| deduplicate.insert(*oty));
    tys.into_iter().collect()
}

/// Extract the span of the custom destructor of a type
/// especially the span of the `impl Drop` header or its entire block
/// when we are working with current local crate.
#[instrument(level = "trace", skip(tcx))]
pub fn ty_dtor_span<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<Span> {
    match ty.kind() {
        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Error(_)
        | ty::Str
        | ty::Never
        | ty::RawPtr(_, _)
        | ty::Ref(_, _, _)
        | ty::FnPtr(_, _)
        | ty::Tuple(_)
        | ty::Dynamic(_, _, _)
        | ty::Alias(_, _)
        | ty::Bound(_, _)
        | ty::Pat(_, _)
        | ty::Placeholder(_)
        | ty::Infer(_)
        | ty::Slice(_)
        | ty::Array(_, _)
        | ty::UnsafeBinder(_) => None,

        ty::Adt(adt_def, _) => {
            if let Some(dtor) = tcx.adt_destructor(adt_def.did()) {
                Some(tcx.def_span(tcx.parent(dtor.did)))
            } else {
                Some(tcx.def_span(adt_def.did()))
            }
        }
        ty::Coroutine(did, _)
        | ty::CoroutineWitness(did, _)
        | ty::CoroutineClosure(did, _)
        | ty::Closure(did, _)
        | ty::FnDef(did, _)
        | ty::Foreign(did) => Some(tcx.def_span(did)),
        ty::Param(_) => None,
    }
}
