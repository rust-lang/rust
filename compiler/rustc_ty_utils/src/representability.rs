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
    // contains a different, structurally recursive type, maintain a stack of
    // seen types and check recursion for each of them (issues #3008, #3779,
    // #74224, #84611). `shadow_seen` contains the full stack and `seen` only
    // the one for the current type (e.g. if we have structs A and B, B contains
    // a field of type A, and we're currently looking at B, then `seen` will be
    // cleared when recursing to check A, but `shadow_seen` won't, so that we
    // can catch cases of mutual recursion where A also contains B).
    let mut seen: Vec<Ty<'_>> = Vec::new();
    let mut shadow_seen: Vec<&'tcx ty::AdtDef> = Vec::new();
    let mut representable_cache = FxHashMap::default();
    let mut force_result = false;
    let r = is_type_structurally_recursive(
        tcx,
        sp,
        &mut seen,
        &mut shadow_seen,
        &mut representable_cache,
        ty,
        &mut force_result,
    );
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
    shadow_seen: &mut Vec<&'tcx ty::AdtDef>,
    representable_cache: &mut FxHashMap<Ty<'tcx>, Representability>,
    ty: Ty<'tcx>,
    force_result: &mut bool,
) -> Representability {
    debug!("are_inner_types_recursive({:?}, {:?}, {:?})", ty, seen, shadow_seen);
    match ty.kind() {
        ty::Tuple(..) => {
            // Find non representable
            fold_repr(ty.tuple_fields().map(|ty| {
                is_type_structurally_recursive(
                    tcx,
                    sp,
                    seen,
                    shadow_seen,
                    representable_cache,
                    ty,
                    force_result,
                )
            }))
        }
        // Fixed-length vectors.
        // FIXME(#11924) Behavior undecided for zero-length vectors.
        ty::Array(ty, _) => is_type_structurally_recursive(
            tcx,
            sp,
            seen,
            shadow_seen,
            representable_cache,
            *ty,
            force_result,
        ),
        ty::Adt(def, substs) => {
            // Find non representable fields with their spans
            fold_repr(def.all_fields().map(|field| {
                let ty = field.ty(tcx, substs);
                let span = match field.did.as_local().and_then(|id| tcx.hir().find_by_def_id(id)) {
                    Some(hir::Node::Field(field)) => field.ty.span,
                    _ => sp,
                };

                let mut result = None;

                // First, we check whether the field type per se is representable.
                // This catches cases as in #74224 and #84611. There is a special
                // case related to mutual recursion, though; consider this example:
                //
                //   struct A<T> {
                //       z: T,
                //       x: B<T>,
                //   }
                //
                //   struct B<T> {
                //       y: A<T>
                //   }
                //
                // Here, without the following special case, both A and B are
                // ContainsRecursive, which is a problem because we only report
                // errors for SelfRecursive. We fix this by detecting this special
                // case (shadow_seen.first() is the type we are originally
                // interested in, and if we ever encounter the same AdtDef again,
                // we know that it must be SelfRecursive) and "forcibly" returning
                // SelfRecursive (by setting force_result, which tells the calling
                // invocations of are_inner_types_representable to forward the
                // result without adjusting).
                if shadow_seen.len() > seen.len() && shadow_seen.first() == Some(def) {
                    *force_result = true;
                    result = Some(Representability::SelfRecursive(vec![span]));
                }

                if result == None {
                    result = Some(Representability::Representable);

                    // Now, we check whether the field types per se are representable, e.g.
                    // for struct Foo { x: Option<Foo> }, we first check whether Option<_>
                    // by itself is representable (which it is), and the nesting of Foo
                    // will be detected later. This is necessary for #74224 and #84611.

                    // If we have encountered an ADT definition that we have not seen
                    // before (no need to check them twice), recurse to see whether that
                    // definition is SelfRecursive. If so, we must be ContainsRecursive.
                    if shadow_seen.len() > 1
                        && !shadow_seen
                            .iter()
                            .take(shadow_seen.len() - 1)
                            .any(|seen_def| seen_def == def)
                    {
                        let adt_def_id = def.did;
                        let raw_adt_ty = tcx.type_of(adt_def_id);
                        debug!("are_inner_types_recursive: checking nested type: {:?}", raw_adt_ty);

                        // Check independently whether the ADT is SelfRecursive. If so,
                        // we must be ContainsRecursive (except for the special case
                        // mentioned above).
                        let mut nested_seen: Vec<Ty<'_>> = vec![];
                        result = Some(
                            match is_type_structurally_recursive(
                                tcx,
                                span,
                                &mut nested_seen,
                                shadow_seen,
                                representable_cache,
                                raw_adt_ty,
                                force_result,
                            ) {
                                Representability::SelfRecursive(_) => {
                                    if *force_result {
                                        Representability::SelfRecursive(vec![span])
                                    } else {
                                        Representability::ContainsRecursive
                                    }
                                }
                                x => x,
                            },
                        );
                    }

                    // We only enter the following block if the type looks representable
                    // so far. This is necessary for cases such as this one (#74224):
                    //
                    //   struct A<T> {
                    //       x: T,
                    //       y: A<A<T>>,
                    //   }
                    //
                    //   struct B {
                    //       z: A<usize>
                    //   }
                    //
                    // When checking B, we recurse into A and check field y of type
                    // A<A<usize>>. We haven't seen this exact type before, so we recurse
                    // into A<A<usize>>, which contains, A<A<A<usize>>>, and so forth,
                    // ad infinitum. We can prevent this from happening by first checking
                    // A separately (the code above) and only checking for nested Bs if
                    // A actually looks representable (which it wouldn't in this example).
                    if result == Some(Representability::Representable) {
                        // Now, even if the type is representable (e.g. Option<_>),
                        // it might still contribute to a recursive type, e.g.:
                        //   struct Foo { x: Option<Option<Foo>> }
                        // These cases are handled by passing the full `seen`
                        // stack to is_type_structurally_recursive (instead of the
                        // empty `nested_seen` above):
                        result = Some(
                            match is_type_structurally_recursive(
                                tcx,
                                span,
                                seen,
                                shadow_seen,
                                representable_cache,
                                ty,
                                force_result,
                            ) {
                                Representability::SelfRecursive(_) => {
                                    Representability::SelfRecursive(vec![span])
                                }
                                x => x,
                            },
                        );
                    }
                }

                result.unwrap()
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
    shadow_seen: &mut Vec<&'tcx ty::AdtDef>,
    representable_cache: &mut FxHashMap<Ty<'tcx>, Representability>,
    ty: Ty<'tcx>,
    force_result: &mut bool,
) -> Representability {
    debug!("is_type_structurally_recursive: {:?} {:?}", ty, sp);
    if let Some(representability) = representable_cache.get(&ty) {
        debug!(
            "is_type_structurally_recursive: {:?} {:?} - (cached) {:?}",
            ty, sp, representability
        );
        return representability.clone();
    }

    let representability = is_type_structurally_recursive_inner(
        tcx,
        sp,
        seen,
        shadow_seen,
        representable_cache,
        ty,
        force_result,
    );

    representable_cache.insert(ty, representability.clone());
    representability
}

fn is_type_structurally_recursive_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    sp: Span,
    seen: &mut Vec<Ty<'tcx>>,
    shadow_seen: &mut Vec<&'tcx ty::AdtDef>,
    representable_cache: &mut FxHashMap<Ty<'tcx>, Representability>,
    ty: Ty<'tcx>,
    force_result: &mut bool,
) -> Representability {
    match ty.kind() {
        ty::Adt(def, _) => {
            {
                debug!("is_type_structurally_recursive_inner: adt: {:?}, seen: {:?}", ty, seen);

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
                // here, because nesting e.g. Options is allowed (as long as the
                // definition of Option doesn't itself include an Option field, which
                // would be a case of SelfRecursive above). The following, too, counts
                // as SelfRecursive:
                //
                // struct Foo { Option<Option<Foo>> }

                for &seen_adt in iter {
                    if ty == seen_adt {
                        debug!("ContainsRecursive: {:?} contains {:?}", seen_adt, ty);
                        return Representability::ContainsRecursive;
                    }
                }
            }

            // For structs and enums, track all previously seen types by pushing them
            // onto the 'seen' stack.
            seen.push(ty);
            shadow_seen.push(def);
            let out = are_inner_types_recursive(
                tcx,
                sp,
                seen,
                shadow_seen,
                representable_cache,
                ty,
                force_result,
            );
            shadow_seen.pop();
            seen.pop();
            out
        }
        _ => {
            // No need to push in other cases.
            are_inner_types_recursive(
                tcx,
                sp,
                seen,
                shadow_seen,
                representable_cache,
                ty,
                force_result,
            )
        }
    }
}
