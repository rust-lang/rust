//! Check whether a type is representable.
use rustc_data_structures::stable_map::FxHashMap;
use rustc_hir as hir;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;
use std::cmp;

/// Describes whether a type is representable. For types that are not
/// representable, 'SelfRecursive' and 'ContainsRecursive' are used to
/// distinguish between types that are recursive with themselves and types that
/// contain a different recursive type. These cases can therefore be treated
/// differently when reporting errors.
///
/// The ordering of the cases is significant. They are sorted so that cmp::max
/// will keep the "more erroneous" of two values.
#[derive(Clone, PartialOrd, Ord, Eq, PartialEq, Debug)]
pub enum Representability {
    Representable,
    ContainsRecursive,
    SelfRecursive(Vec<Span>),
}

/// Check whether a type is representable. This means it cannot contain unboxed
/// structural recursion. This check is needed for structs and enums.
pub fn ty_is_representable<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, sp: Span) -> Representability {
    debug!("is_type_representable: {:?}", ty);
    // To avoid a stack overflow when checking an enum variant or struct that
    // contains a different, structurally recursive type, maintain a stack
    // of seen types and check recursion for each of them (issues #3008, #3779).
    let mut seen: Vec<Ty<'_>> = Vec::new();
    let mut representable_cache = FxHashMap::default();
    let r = is_type_structurally_recursive(tcx, sp, &mut seen, &mut representable_cache, ty);
    debug!("is_type_representable: {:?} is {:?}", ty, r);
    r
}

// Iterate until something non-representable is found
fn fold_repr<It: Iterator<Item = Representability>>(iter: It) -> Representability {
    iter.fold(Representability::Representable, |r1, r2| match (r1, r2) {
        (Representability::SelfRecursive(v1), Representability::SelfRecursive(v2)) => {
            Representability::SelfRecursive(v1.into_iter().chain(v2).collect())
        }
        (r1, r2) => cmp::max(r1, r2),
    })
}

fn are_inner_types_recursive<'tcx>(
    tcx: TyCtxt<'tcx>,
    sp: Span,
    seen: &mut Vec<Ty<'tcx>>,
    representable_cache: &mut FxHashMap<Ty<'tcx>, Representability>,
    ty: Ty<'tcx>,
) -> Representability {
    match ty.kind() {
        ty::Tuple(..) => {
            // Find non representable
            fold_repr(
                ty.tuple_fields().map(|ty| {
                    is_type_structurally_recursive(tcx, sp, seen, representable_cache, ty)
                }),
            )
        }
        // Fixed-length vectors.
        // FIXME(#11924) Behavior undecided for zero-length vectors.
        ty::Array(ty, _) => is_type_structurally_recursive(tcx, sp, seen, representable_cache, ty),
        ty::Adt(def, substs) => {
            // Find non representable fields with their spans
            fold_repr(def.all_fields().map(|field| {
                let ty = field.ty(tcx, substs);
                let span = match field
                    .did
                    .as_local()
                    .map(|id| tcx.hir().local_def_id_to_hir_id(id))
                    .and_then(|id| tcx.hir().find(id))
                {
                    Some(hir::Node::Field(field)) => field.ty.span,
                    _ => sp,
                };
                match is_type_structurally_recursive(tcx, span, seen, representable_cache, ty) {
                    Representability::SelfRecursive(_) => {
                        Representability::SelfRecursive(vec![span])
                    }
                    x => x,
                }
            }))
        }
        ty::Closure(..) => {
            // this check is run on type definitions, so we don't expect
            // to see closure types
            bug!("requires check invoked on inapplicable type: {:?}", ty)
        }
        _ => Representability::Representable,
    }
}

fn same_adt<'tcx>(ty: Ty<'tcx>, def: &'tcx ty::AdtDef) -> bool {
    match *ty.kind() {
        ty::Adt(ty_def, _) => ty_def == def,
        _ => false,
    }
}

// Does the type `ty` directly (without indirection through a pointer)
// contain any types on stack `seen`?
fn is_type_structurally_recursive<'tcx>(
    tcx: TyCtxt<'tcx>,
    sp: Span,
    seen: &mut Vec<Ty<'tcx>>,
    representable_cache: &mut FxHashMap<Ty<'tcx>, Representability>,
    ty: Ty<'tcx>,
) -> Representability {
    debug!("is_type_structurally_recursive: {:?} {:?}", ty, sp);
    if let Some(representability) = representable_cache.get(ty) {
        debug!(
            "is_type_structurally_recursive: {:?} {:?} - (cached) {:?}",
            ty, sp, representability
        );
        return representability.clone();
    }

    let representability =
        is_type_structurally_recursive_inner(tcx, sp, seen, representable_cache, ty);

    representable_cache.insert(ty, representability.clone());
    representability
}

fn is_type_structurally_recursive_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    sp: Span,
    seen: &mut Vec<Ty<'tcx>>,
    representable_cache: &mut FxHashMap<Ty<'tcx>, Representability>,
    ty: Ty<'tcx>,
) -> Representability {
    match ty.kind() {
        ty::Adt(def, _) => {
            {
                // Iterate through stack of previously seen types.
                let mut iter = seen.iter();

                // The first item in `seen` is the type we are actually curious about.
                // We want to return SelfRecursive if this type contains itself.
                // It is important that we DON'T take generic parameters into account
                // for this check, so that Bar<T> in this example counts as SelfRecursive:
                //
                // struct Foo;
                // struct Bar<T> { x: Bar<Foo> }

                if let Some(&seen_adt) = iter.next() {
                    if same_adt(seen_adt, *def) {
                        debug!("SelfRecursive: {:?} contains {:?}", seen_adt, ty);
                        return Representability::SelfRecursive(vec![sp]);
                    }
                }

                // We also need to know whether the first item contains other types
                // that are structurally recursive. If we don't catch this case, we
                // will recurse infinitely for some inputs.
                //
                // It is important that we DO take generic parameters into account
                // here, so that code like this is considered SelfRecursive, not
                // ContainsRecursive:
                //
                // struct Foo { Option<Option<Foo>> }

                for &seen_adt in iter {
                    if ty::TyS::same_type(ty, seen_adt) {
                        debug!("ContainsRecursive: {:?} contains {:?}", seen_adt, ty);
                        return Representability::ContainsRecursive;
                    }
                }
            }

            // For structs and enums, track all previously seen types by pushing them
            // onto the 'seen' stack.
            seen.push(ty);
            let out = are_inner_types_recursive(tcx, sp, seen, representable_cache, ty);
            seen.pop();
            out
        }
        _ => {
            // No need to push in other cases.
            are_inner_types_recursive(tcx, sp, seen, representable_cache, ty)
        }
    }
}
