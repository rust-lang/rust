// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::InferCtxt;
use syntax::ast;
use syntax::codemap::Span;
use traits::FulfillmentContext;
use ty::{self, Ty, TypeFoldable};
use ty::outlives::Component;
use ty::wf;

/// Outlives bounds are relationships between generic parameters,
/// whether they both be regions (`'a: 'b`) or whether types are
/// involved (`T: 'a`).  These relationships can be extracted from the
/// full set of predicates we understand or also from types (in which
/// case they are called implied bounds). They are fed to the
/// `OutlivesEnv` which in turn is supplied to the region checker and
/// other parts of the inference system.
#[derive(Debug)]
pub enum OutlivesBound<'tcx> {
    RegionSubRegion(ty::Region<'tcx>, ty::Region<'tcx>),
    RegionSubParam(ty::Region<'tcx>, ty::ParamTy),
    RegionSubProjection(ty::Region<'tcx>, ty::ProjectionTy<'tcx>),
}

impl<'cx, 'gcx, 'tcx> InferCtxt<'cx, 'gcx, 'tcx> {
    /// Implied bounds are region relationships that we deduce
    /// automatically.  The idea is that (e.g.) a caller must check that a
    /// function's argument types are well-formed immediately before
    /// calling that fn, and hence the *callee* can assume that its
    /// argument types are well-formed. This may imply certain relationships
    /// between generic parameters. For example:
    ///
    ///     fn foo<'a,T>(x: &'a T)
    ///
    /// can only be called with a `'a` and `T` such that `&'a T` is WF.
    /// For `&'a T` to be WF, `T: 'a` must hold. So we can assume `T: 'a`.
    ///
    /// # Parameters
    ///
    /// - `param_env`, the where-clauses in scope
    /// - `body_id`, the body-id to use when normalizing assoc types.
    ///   Note that this may cause outlives obligations to be injected
    ///   into the inference context with this body-id.
    /// - `ty`, the type that we are supposed to assume is WF.
    /// - `span`, a span to use when normalizing, hopefully not important,
    ///   might be useful if a `bug!` occurs.
    pub fn implied_outlives_bounds(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        body_id: ast::NodeId,
        ty: Ty<'tcx>,
        span: Span,
    ) -> Vec<OutlivesBound<'tcx>> {
        let tcx = self.tcx;

        // Sometimes when we ask what it takes for T: WF, we get back that
        // U: WF is required; in that case, we push U onto this stack and
        // process it next. Currently (at least) these resulting
        // predicates are always guaranteed to be a subset of the original
        // type, so we need not fear non-termination.
        let mut wf_types = vec![ty];

        let mut implied_bounds = vec![];

        let mut fulfill_cx = FulfillmentContext::new();

        while let Some(ty) = wf_types.pop() {
            // Compute the obligations for `ty` to be well-formed. If `ty` is
            // an unresolved inference variable, just substituted an empty set
            // -- because the return type here is going to be things we *add*
            // to the environment, it's always ok for this set to be smaller
            // than the ultimate set. (Note: normally there won't be
            // unresolved inference variables here anyway, but there might be
            // during typeck under some circumstances.)
            let obligations = wf::obligations(self, param_env, body_id, ty, span).unwrap_or(vec![]);

            // NB: All of these predicates *ought* to be easily proven
            // true. In fact, their correctness is (mostly) implied by
            // other parts of the program. However, in #42552, we had
            // an annoying scenario where:
            //
            // - Some `T::Foo` gets normalized, resulting in a
            //   variable `_1` and a `T: Trait<Foo=_1>` constraint
            //   (not sure why it couldn't immediately get
            //   solved). This result of `_1` got cached.
            // - These obligations were dropped on the floor here,
            //   rather than being registered.
            // - Then later we would get a request to normalize
            //   `T::Foo` which would result in `_1` being used from
            //   the cache, but hence without the `T: Trait<Foo=_1>`
            //   constraint. As a result, `_1` never gets resolved,
            //   and we get an ICE (in dropck).
            //
            // Therefore, we register any predicates involving
            // inference variables. We restrict ourselves to those
            // involving inference variables both for efficiency and
            // to avoids duplicate errors that otherwise show up.
            fulfill_cx.register_predicate_obligations(
                self,
                obligations
                    .iter()
                    .filter(|o| o.predicate.has_infer_types())
                    .cloned(),
            );

            // From the full set of obligations, just filter down to the
            // region relationships.
            implied_bounds.extend(obligations.into_iter().flat_map(|obligation| {
                assert!(!obligation.has_escaping_regions());
                match obligation.predicate {
                    ty::Predicate::Trait(..) |
                    ty::Predicate::Equate(..) |
                    ty::Predicate::Subtype(..) |
                    ty::Predicate::Projection(..) |
                    ty::Predicate::ClosureKind(..) |
                    ty::Predicate::ObjectSafe(..) |
                    ty::Predicate::ConstEvaluatable(..) => vec![],

                    ty::Predicate::WellFormed(subty) => {
                        wf_types.push(subty);
                        vec![]
                    }

                    ty::Predicate::RegionOutlives(ref data) => match data.no_late_bound_regions() {
                        None => vec![],
                        Some(ty::OutlivesPredicate(r_a, r_b)) => {
                            vec![OutlivesBound::RegionSubRegion(r_b, r_a)]
                        }
                    },

                    ty::Predicate::TypeOutlives(ref data) => match data.no_late_bound_regions() {
                        None => vec![],
                        Some(ty::OutlivesPredicate(ty_a, r_b)) => {
                            let ty_a = self.resolve_type_vars_if_possible(&ty_a);
                            let components = tcx.outlives_components(ty_a);
                            Self::implied_bounds_from_components(r_b, components)
                        }
                    },
                }
            }));
        }

        // Ensure that those obligations that we had to solve
        // get solved *here*.
        match fulfill_cx.select_all_or_error(self) {
            Ok(()) => (),
            Err(errors) => self.report_fulfillment_errors(&errors, None),
        }

        implied_bounds
    }

    /// When we have an implied bound that `T: 'a`, we can further break
    /// this down to determine what relationships would have to hold for
    /// `T: 'a` to hold. We get to assume that the caller has validated
    /// those relationships.
    fn implied_bounds_from_components(
        sub_region: ty::Region<'tcx>,
        sup_components: Vec<Component<'tcx>>,
    ) -> Vec<OutlivesBound<'tcx>> {
        sup_components
            .into_iter()
            .flat_map(|component| {
                match component {
                    Component::Region(r) =>
                        vec![OutlivesBound::RegionSubRegion(sub_region, r)],
                    Component::Param(p) =>
                        vec![OutlivesBound::RegionSubParam(sub_region, p)],
                    Component::Projection(p) =>
                        vec![OutlivesBound::RegionSubProjection(sub_region, p)],
                    Component::EscapingProjection(_) =>
                    // If the projection has escaping regions, don't
                    // try to infer any implied bounds even for its
                    // free components. This is conservative, because
                    // the caller will still have to prove that those
                    // free components outlive `sub_region`. But the
                    // idea is that the WAY that the caller proves
                    // that may change in the future and we want to
                    // give ourselves room to get smarter here.
                        vec![],
                    Component::UnresolvedInferenceVariable(..) =>
                        vec![],
                }
            })
            .collect()
    }
}

pub fn explicit_outlives_bounds<'tcx>(
    param_env: ty::ParamEnv<'tcx>,
) -> impl Iterator<Item = OutlivesBound<'tcx>> + 'tcx {
    debug!("explicit_outlives_bounds()");
    param_env
        .caller_bounds
        .into_iter()
        .filter_map(move |predicate| match predicate {
            ty::Predicate::Projection(..) |
            ty::Predicate::Trait(..) |
            ty::Predicate::Equate(..) |
            ty::Predicate::Subtype(..) |
            ty::Predicate::WellFormed(..) |
            ty::Predicate::ObjectSafe(..) |
            ty::Predicate::ClosureKind(..) |
            ty::Predicate::TypeOutlives(..) |
            ty::Predicate::ConstEvaluatable(..) => None,
            ty::Predicate::RegionOutlives(ref data) => data.no_late_bound_regions().map(
                |ty::OutlivesPredicate(r_a, r_b)| OutlivesBound::RegionSubRegion(r_b, r_a),
            ),
        })
}
