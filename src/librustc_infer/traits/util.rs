use smallvec::smallvec;

use rustc::ty::outlives::Component;
use rustc::ty::{self, ToPolyTraitRef, TyCtxt};
use rustc_data_structures::fx::FxHashSet;

fn anonymize_predicate<'tcx>(tcx: TyCtxt<'tcx>, pred: &ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
    match *pred {
        ty::Predicate::Trait(ref data, constness) => {
            ty::Predicate::Trait(tcx.anonymize_late_bound_regions(data), constness)
        }

        ty::Predicate::RegionOutlives(ref data) => {
            ty::Predicate::RegionOutlives(tcx.anonymize_late_bound_regions(data))
        }

        ty::Predicate::TypeOutlives(ref data) => {
            ty::Predicate::TypeOutlives(tcx.anonymize_late_bound_regions(data))
        }

        ty::Predicate::Projection(ref data) => {
            ty::Predicate::Projection(tcx.anonymize_late_bound_regions(data))
        }

        ty::Predicate::WellFormed(data) => ty::Predicate::WellFormed(data),

        ty::Predicate::ObjectSafe(data) => ty::Predicate::ObjectSafe(data),

        ty::Predicate::ClosureKind(closure_def_id, closure_substs, kind) => {
            ty::Predicate::ClosureKind(closure_def_id, closure_substs, kind)
        }

        ty::Predicate::Subtype(ref data) => {
            ty::Predicate::Subtype(tcx.anonymize_late_bound_regions(data))
        }

        ty::Predicate::ConstEvaluatable(def_id, substs) => {
            ty::Predicate::ConstEvaluatable(def_id, substs)
        }
    }
}

struct PredicateSet<'tcx> {
    tcx: TyCtxt<'tcx>,
    set: FxHashSet<ty::Predicate<'tcx>>,
}

impl PredicateSet<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx: tcx, set: Default::default() }
    }

    fn insert(&mut self, pred: &ty::Predicate<'tcx>) -> bool {
        // We have to be careful here because we want
        //
        //    for<'a> Foo<&'a int>
        //
        // and
        //
        //    for<'b> Foo<&'b int>
        //
        // to be considered equivalent. So normalize all late-bound
        // regions before we throw things into the underlying set.
        self.set.insert(anonymize_predicate(self.tcx, pred))
    }
}

impl<T: AsRef<ty::Predicate<'tcx>>> Extend<T> for PredicateSet<'tcx> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for pred in iter {
            self.insert(pred.as_ref());
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// `Elaboration` iterator
///////////////////////////////////////////////////////////////////////////

/// "Elaboration" is the process of identifying all the predicates that
/// are implied by a source predicate. Currently, this basically means
/// walking the "supertraits" and other similar assumptions. For example,
/// if we know that `T: Ord`, the elaborator would deduce that `T: PartialOrd`
/// holds as well. Similarly, if we have `trait Foo: 'static`, and we know that
/// `T: Foo`, then we know that `T: 'static`.
pub struct Elaborator<'tcx> {
    stack: Vec<ty::Predicate<'tcx>>,
    visited: PredicateSet<'tcx>,
}

pub fn elaborate_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut predicates: Vec<ty::Predicate<'tcx>>,
) -> Elaborator<'tcx> {
    let mut visited = PredicateSet::new(tcx);
    predicates.retain(|pred| visited.insert(pred));
    Elaborator { stack: predicates, visited }
}

impl Elaborator<'tcx> {
    fn elaborate(&mut self, predicate: &ty::Predicate<'tcx>) {
        let tcx = self.visited.tcx;
        match *predicate {
            ty::Predicate::Trait(ref data, _) => {
                // Get predicates declared on the trait.
                let predicates = tcx.super_predicates_of(data.def_id());

                let predicates = predicates
                    .predicates
                    .iter()
                    .map(|(pred, _)| pred.subst_supertrait(tcx, &data.to_poly_trait_ref()));
                debug!("super_predicates: data={:?} predicates={:?}", data, predicates.clone());

                // Only keep those bounds that we haven't already seen.
                // This is necessary to prevent infinite recursion in some
                // cases. One common case is when people define
                // `trait Sized: Sized { }` rather than `trait Sized { }`.
                let visited = &mut self.visited;
                let predicates = predicates.filter(|pred| visited.insert(pred));

                self.stack.extend(predicates);
            }
            ty::Predicate::WellFormed(..) => {
                // Currently, we do not elaborate WF predicates,
                // although we easily could.
            }
            ty::Predicate::ObjectSafe(..) => {
                // Currently, we do not elaborate object-safe
                // predicates.
            }
            ty::Predicate::Subtype(..) => {
                // Currently, we do not "elaborate" predicates like `X <: Y`,
                // though conceivably we might.
            }
            ty::Predicate::Projection(..) => {
                // Nothing to elaborate in a projection predicate.
            }
            ty::Predicate::ClosureKind(..) => {
                // Nothing to elaborate when waiting for a closure's kind to be inferred.
            }
            ty::Predicate::ConstEvaluatable(..) => {
                // Currently, we do not elaborate const-evaluatable
                // predicates.
            }
            ty::Predicate::RegionOutlives(..) => {
                // Nothing to elaborate from `'a: 'b`.
            }
            ty::Predicate::TypeOutlives(ref data) => {
                // We know that `T: 'a` for some type `T`. We can
                // often elaborate this. For example, if we know that
                // `[U]: 'a`, that implies that `U: 'a`. Similarly, if
                // we know `&'a U: 'b`, then we know that `'a: 'b` and
                // `U: 'b`.
                //
                // We can basically ignore bound regions here. So for
                // example `for<'c> Foo<'a,'c>: 'b` can be elaborated to
                // `'a: 'b`.

                // Ignore `for<'a> T: 'a` -- we might in the future
                // consider this as evidence that `T: 'static`, but
                // I'm a bit wary of such constructions and so for now
                // I want to be conservative. --nmatsakis
                let ty_max = data.skip_binder().0;
                let r_min = data.skip_binder().1;
                if r_min.is_late_bound() {
                    return;
                }

                let visited = &mut self.visited;
                let mut components = smallvec![];
                tcx.push_outlives_components(ty_max, &mut components);
                self.stack.extend(
                    components
                        .into_iter()
                        .filter_map(|component| match component {
                            Component::Region(r) => {
                                if r.is_late_bound() {
                                    None
                                } else {
                                    Some(ty::Predicate::RegionOutlives(ty::Binder::dummy(
                                        ty::OutlivesPredicate(r, r_min),
                                    )))
                                }
                            }

                            Component::Param(p) => {
                                let ty = tcx.mk_ty_param(p.index, p.name);
                                Some(ty::Predicate::TypeOutlives(ty::Binder::dummy(
                                    ty::OutlivesPredicate(ty, r_min),
                                )))
                            }

                            Component::UnresolvedInferenceVariable(_) => None,

                            Component::Projection(_) | Component::EscapingProjection(_) => {
                                // We can probably do more here. This
                                // corresponds to a case like `<T as
                                // Foo<'a>>::U: 'b`.
                                None
                            }
                        })
                        .filter(|p| visited.insert(p)),
                );
            }
        }
    }
}

impl Iterator for Elaborator<'tcx> {
    type Item = ty::Predicate<'tcx>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stack.len(), None)
    }

    fn next(&mut self) -> Option<ty::Predicate<'tcx>> {
        // Extract next item from top-most stack frame, if any.
        if let Some(pred) = self.stack.pop() {
            self.elaborate(&pred);
            Some(pred)
        } else {
            None
        }
    }
}
