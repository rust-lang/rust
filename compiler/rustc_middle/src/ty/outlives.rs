// The outlines relation `T: 'a` or `'a: 'b`. This code frequently
// refers to rules defined in RFC 1214 (`OutlivesFooBar`), so see that
// RFC for reference.

use crate::ty::subst::{GenericArg, GenericArgKind};
use crate::ty::{self, Ty, TyCtxt, TypeFoldable};
use smallvec::SmallVec;

#[derive(Debug)]
pub enum Component<'tcx> {
    Region(ty::Region<'tcx>),
    Param(ty::ParamTy),
    UnresolvedInferenceVariable(ty::InferTy),

    // Projections like `T::Foo` are tricky because a constraint like
    // `T::Foo: 'a` can be satisfied in so many ways. There may be a
    // where-clause that says `T::Foo: 'a`, or the defining trait may
    // include a bound like `type Foo: 'static`, or -- in the most
    // conservative way -- we can prove that `T: 'a` (more generally,
    // that all components in the projection outlive `'a`). This code
    // is not in a position to judge which is the best technique, so
    // we just product the projection as a component and leave it to
    // the consumer to decide (but see `EscapingProjection` below).
    Projection(ty::ProjectionTy<'tcx>),

    // In the case where a projection has escaping regions -- meaning
    // regions bound within the type itself -- we always use
    // the most conservative rule, which requires that all components
    // outlive the bound. So for example if we had a type like this:
    //
    //     for<'a> Trait1<  <T as Trait2<'a,'b>>::Foo  >
    //                      ~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    // then the inner projection (underlined) has an escaping region
    // `'a`. We consider that outer trait `'c` to meet a bound if `'b`
    // outlives `'b: 'c`, and we don't consider whether the trait
    // declares that `Foo: 'static` etc. Therefore, we just return the
    // free components of such a projection (in this case, `'b`).
    //
    // However, in the future, we may want to get smarter, and
    // actually return a "higher-ranked projection" here. Therefore,
    // we mark that these components are part of an escaping
    // projection, so that implied bounds code can avoid relying on
    // them. This gives us room to improve the regionck reasoning in
    // the future without breaking backwards compat.
    EscapingProjection(Vec<Component<'tcx>>),
}

impl<'tcx> TyCtxt<'tcx> {
    /// Push onto `out` all the things that must outlive `'a` for the condition
    /// `ty0: 'a` to hold. Note that `ty0` must be a **fully resolved type**.
    pub fn push_outlives_components(self, ty0: Ty<'tcx>, out: &mut SmallVec<[Component<'tcx>; 4]>) {
        compute_components(self, ty0, out);
        debug!("components({:?}) = {:?}", ty0, out);
    }
}

fn compute_components(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, out: &mut SmallVec<[Component<'tcx>; 4]>) {
    // Descend through the types, looking for the various "base"
    // components and collecting them into `out`. This is not written
    // with `collect()` because of the need to sometimes skip subtrees
    // in the `subtys` iterator (e.g., when encountering a
    // projection).
    match *ty.kind() {
            ty::FnDef(_, substs) => {
                // HACK(eddyb) ignore lifetimes found shallowly in `substs`.
                // This is inconsistent with `ty::Adt` (including all substs)
                // and with `ty::Closure` (ignoring all substs other than
                // upvars, of which a `ty::FnDef` doesn't have any), but
                // consistent with previous (accidental) behavior.
                // See https://github.com/rust-lang/rust/issues/70917
                // for further background and discussion.
                for child in substs {
                    match child.unpack() {
                        GenericArgKind::Type(ty) => {
                            compute_components(tcx, ty, out);
                        }
                        GenericArgKind::Lifetime(_) => {}
                        GenericArgKind::Const(_) => {
                            compute_components_recursive(tcx, child, out);
                        }
                    }
                }
            }

            ty::Array(element, _) => {
                // Don't look into the len const as it doesn't affect regions
                compute_components(tcx, element, out);
            }

            ty::Closure(_, ref substs) => {
                for upvar_ty in substs.as_closure().upvar_tys() {
                    compute_components(tcx, upvar_ty, out);
                }
            }

            ty::Generator(_, ref substs, _) => {
                // Same as the closure case
                for upvar_ty in substs.as_generator().upvar_tys() {
                    compute_components(tcx, upvar_ty, out);
                }

                // We ignore regions in the generator interior as we don't
                // want these to affect region inference
            }

            // All regions are bound inside a witness
            ty::GeneratorWitness(..) => (),

            // OutlivesTypeParameterEnv -- the actual checking that `X:'a`
            // is implied by the environment is done in regionck.
            ty::Param(p) => {
                out.push(Component::Param(p));
            }

            // For projections, we prefer to generate an obligation like
            // `<P0 as Trait<P1...Pn>>::Foo: 'a`, because this gives the
            // regionck more ways to prove that it holds. However,
            // regionck is not (at least currently) prepared to deal with
            // higher-ranked regions that may appear in the
            // trait-ref. Therefore, if we see any higher-ranke regions,
            // we simply fallback to the most restrictive rule, which
            // requires that `Pi: 'a` for all `i`.
            ty::Projection(ref data) => {
                if !data.has_escaping_bound_vars() {
                    // best case: no escaping regions, so push the
                    // projection and skip the subtree (thus generating no
                    // constraints for Pi). This defers the choice between
                    // the rules OutlivesProjectionEnv,
                    // OutlivesProjectionTraitDef, and
                    // OutlivesProjectionComponents to regionck.
                    out.push(Component::Projection(*data));
                } else {
                    // fallback case: hard code
                    // OutlivesProjectionComponents.  Continue walking
                    // through and constrain Pi.
                    let mut subcomponents = smallvec![];
                    compute_components_recursive(tcx, ty.into(), &mut subcomponents);
                    out.push(Component::EscapingProjection(subcomponents.into_iter().collect()));
                }
            }

            // We assume that inference variables are fully resolved.
            // So, if we encounter an inference variable, just record
            // the unresolved variable as a component.
            ty::Infer(infer_ty) => {
                out.push(Component::UnresolvedInferenceVariable(infer_ty));
            }

            // Most types do not introduce any region binders, nor
            // involve any other subtle cases, and so the WF relation
            // simply constraints any regions referenced directly by
            // the type and then visits the types that are lexically
            // contained within. (The comments refer to relevant rules
            // from RFC1214.)
            ty::Bool |            // OutlivesScalar
            ty::Char |            // OutlivesScalar
            ty::Int(..) |         // OutlivesScalar
            ty::Uint(..) |        // OutlivesScalar
            ty::Float(..) |       // OutlivesScalar
            ty::Never |           // ...
            ty::Adt(..) |         // OutlivesNominalType
            ty::Opaque(..) |      // OutlivesNominalType (ish)
            ty::Foreign(..) |     // OutlivesNominalType
            ty::Str |             // OutlivesScalar (ish)
            ty::Slice(..) |       // ...
            ty::RawPtr(..) |      // ...
            ty::Ref(..) |         // OutlivesReference
            ty::Tuple(..) |       // ...
            ty::FnPtr(_) |        // OutlivesFunction (*)
            ty::Dynamic(..) |     // OutlivesObject, OutlivesFragment (*)
            ty::Placeholder(..) |
            ty::Bound(..) |
            ty::Error(_) => {
                // (*) Function pointers and trait objects are both binders.
                // In the RFC, this means we would add the bound regions to
                // the "bound regions list".  In our representation, no such
                // list is maintained explicitly, because bound regions
                // themselves can be readily identified.
                compute_components_recursive(tcx, ty.into(), out);
            }
        }
}

fn compute_components_recursive(
    tcx: TyCtxt<'tcx>,
    parent: GenericArg<'tcx>,
    out: &mut SmallVec<[Component<'tcx>; 4]>,
) {
    for child in parent.walk_shallow() {
        match child.unpack() {
            GenericArgKind::Type(ty) => {
                compute_components(tcx, ty, out);
            }
            GenericArgKind::Lifetime(lt) => {
                // Ignore late-bound regions.
                if !lt.is_late_bound() {
                    out.push(Component::Region(lt));
                }
            }
            GenericArgKind::Const(_) => {
                compute_components_recursive(tcx, child, out);
            }
        }
    }
}
