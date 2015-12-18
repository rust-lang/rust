// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The outlines relation `T: 'a` or `'a: 'b`. This code frequently
// refers to rules defined in RFC 1214 (`OutlivesFooBar`), so see that
// RFC for reference.

use middle::infer::InferCtxt;
use middle::ty::{self, RegionEscape, Ty};

#[derive(Debug)]
pub enum Component<'tcx> {
    Region(ty::Region),
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

/// Returns all the things that must outlive `'a` for the condition
/// `ty0: 'a` to hold.
pub fn components<'a,'tcx>(infcx: &InferCtxt<'a,'tcx>,
                           ty0: Ty<'tcx>)
                           -> Vec<Component<'tcx>> {
    let mut components = vec![];
    compute_components(infcx, ty0, &mut components);
    debug!("components({:?}) = {:?}", ty0, components);
    components
}

fn compute_components<'a,'tcx>(infcx: &InferCtxt<'a,'tcx>,
                               ty: Ty<'tcx>,
                               out: &mut Vec<Component<'tcx>>) {
    // Descend through the types, looking for the various "base"
    // components and collecting them into `out`. This is not written
    // with `collect()` because of the need to sometimes skip subtrees
    // in the `subtys` iterator (e.g., when encountering a
    // projection).
    match ty.sty {
        ty::TyClosure(_, ref substs) => {
            // FIXME(#27086). We do not accumulate from substs, since they
            // don't represent reachable data. This means that, in
            // practice, some of the lifetime parameters might not
            // be in scope when the body runs, so long as there is
            // no reachable data with that lifetime. For better or
            // worse, this is consistent with fn types, however,
            // which can also encapsulate data in this fashion
            // (though it's somewhat harder, and typically
            // requires virtual dispatch).
            //
            // Note that changing this (in a naive way, at least)
            // causes regressions for what appears to be perfectly
            // reasonable code like this:
            //
            // ```
            // fn foo<'a>(p: &Data<'a>) {
            //    bar(|q: &mut Parser| q.read_addr())
            // }
            // fn bar(p: Box<FnMut(&mut Parser)+'static>) {
            // }
            // ```
            //
            // Note that `p` (and `'a`) are not used in the
            // closure at all, but to meet the requirement that
            // the closure type `C: 'static` (so it can be coerced
            // to the object type), we get the requirement that
            // `'a: 'static` since `'a` appears in the closure
            // type `C`.
            //
            // A smarter fix might "prune" unused `func_substs` --
            // this would avoid breaking simple examples like
            // this, but would still break others (which might
            // indeed be invalid, depending on your POV). Pruning
            // would be a subtle process, since we have to see
            // what func/type parameters are used and unused,
            // taking into consideration UFCS and so forth.

            for &upvar_ty in &substs.upvar_tys {
                compute_components(infcx, upvar_ty, out);
            }
        }

        // OutlivesTypeParameterEnv -- the actual checking that `X:'a`
        // is implied by the environment is done in regionck.
        ty::TyParam(p) => {
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
        ty::TyProjection(ref data) => {
            if !data.has_escaping_regions() {
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
                let subcomponents = capture_components(infcx, ty);
                out.push(Component::EscapingProjection(subcomponents));
            }
        }

        // If we encounter an inference variable, try to resolve it
        // and proceed with resolved version. If we cannot resolve it,
        // then record the unresolved variable as a component.
        ty::TyInfer(_) => {
            let ty = infcx.resolve_type_vars_if_possible(&ty);
            if let ty::TyInfer(infer_ty) = ty.sty {
                out.push(Component::UnresolvedInferenceVariable(infer_ty));
            } else {
                compute_components(infcx, ty, out);
            }
        }

        // Most types do not introduce any region binders, nor
        // involve any other subtle cases, and so the WF relation
        // simply constraints any regions referenced directly by
        // the type and then visits the types that are lexically
        // contained within. (The comments refer to relevant rules
        // from RFC1214.)
        ty::TyBool |            // OutlivesScalar
        ty::TyChar |            // OutlivesScalar
        ty::TyInt(..) |         // OutlivesScalar
        ty::TyUint(..) |        // OutlivesScalar
        ty::TyFloat(..) |       // OutlivesScalar
        ty::TyEnum(..) |        // OutlivesNominalType
        ty::TyStruct(..) |      // OutlivesNominalType
        ty::TyBox(..) |         // OutlivesNominalType (ish)
        ty::TyStr |             // OutlivesScalar (ish)
        ty::TyArray(..) |       // ...
        ty::TySlice(..) |       // ...
        ty::TyRawPtr(..) |      // ...
        ty::TyRef(..) |         // OutlivesReference
        ty::TyTuple(..) |       // ...
        ty::TyBareFn(..) |      // OutlivesFunction (*)
        ty::TyTrait(..) |       // OutlivesObject, OutlivesFragment (*)
        ty::TyError => {
            // (*) Bare functions and traits are both binders. In the
            // RFC, this means we would add the bound regions to the
            // "bound regions list".  In our representation, no such
            // list is maintained explicitly, because bound regions
            // themselves can be readily identified.

            push_region_constraints(out, ty.regions());
            for subty in ty.walk_shallow() {
                compute_components(infcx, subty, out);
            }
        }
    }
}

fn capture_components<'a,'tcx>(infcx: &InferCtxt<'a,'tcx>,
                               ty: Ty<'tcx>)
                               -> Vec<Component<'tcx>> {
    let mut temp = vec![];
    push_region_constraints(&mut temp, ty.regions());
    for subty in ty.walk_shallow() {
        compute_components(infcx, subty, &mut temp);
    }
    temp
}

fn push_region_constraints<'tcx>(out: &mut Vec<Component<'tcx>>, regions: Vec<ty::Region>) {
    for r in regions {
        if !r.is_bound() {
            out.push(Component::Region(r));
        }
    }
}
