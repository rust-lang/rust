use rustc_data_structures::fx::FxHashSet;
use rustc_middle::bug;
use rustc_middle::ty::visit::{TypeSuperVisitable, TypeVisitor};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;
use rustc_type_ir::fold::TypeFoldable;
use tracing::debug;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Parameter(pub u32);

impl From<ty::ParamTy> for Parameter {
    fn from(param: ty::ParamTy) -> Self {
        Parameter(param.index)
    }
}

impl From<ty::EarlyParamRegion> for Parameter {
    fn from(param: ty::EarlyParamRegion) -> Self {
        Parameter(param.index)
    }
}

impl From<ty::ParamConst> for Parameter {
    fn from(param: ty::ParamConst) -> Self {
        Parameter(param.index)
    }
}

/// Returns the set of parameters constrained by the impl header.
pub(crate) fn parameters_for_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_self_ty: Ty<'tcx>,
    impl_trait_ref: Option<ty::TraitRef<'tcx>>,
) -> FxHashSet<Parameter> {
    let vec = match impl_trait_ref {
        Some(tr) => parameters_for(tcx, tr, false),
        None => parameters_for(tcx, impl_self_ty, false),
    };
    vec.into_iter().collect()
}

/// If `include_nonconstraining` is false, returns the list of parameters that are
/// constrained by `value` - i.e., the value of each parameter in the list is
/// uniquely determined by `value` (see RFC 447). If it is true, return the list
/// of parameters whose values are needed in order to constrain `value` - these
/// differ, with the latter being a superset, in the presence of projections.
pub(crate) fn parameters_for<'tcx>(
    tcx: TyCtxt<'tcx>,
    value: impl TypeFoldable<TyCtxt<'tcx>>,
    include_nonconstraining: bool,
) -> Vec<Parameter> {
    let mut collector = ParameterCollector { parameters: vec![], include_nonconstraining };
    let value = if !include_nonconstraining { tcx.expand_weak_alias_tys(value) } else { value };
    value.visit_with(&mut collector);
    collector.parameters
}

struct ParameterCollector {
    parameters: Vec<Parameter>,
    include_nonconstraining: bool,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ParameterCollector {
    fn visit_ty(&mut self, t: Ty<'tcx>) {
        match *t.kind() {
            // Projections are not injective in general.
            ty::Alias(ty::Projection | ty::Inherent | ty::Opaque, _)
                if !self.include_nonconstraining =>
            {
                return;
            }
            // All weak alias types should've been expanded beforehand.
            ty::Alias(ty::Weak, _) if !self.include_nonconstraining => {
                bug!("unexpected weak alias type")
            }
            ty::Param(param) => self.parameters.push(Parameter::from(param)),
            _ => {}
        }

        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        if let ty::ReEarlyParam(data) = *r {
            self.parameters.push(Parameter::from(data));
        }
    }

    fn visit_const(&mut self, c: ty::Const<'tcx>) {
        match c.kind() {
            ty::ConstKind::Unevaluated(..) if !self.include_nonconstraining => {
                // Constant expressions are not injective in general.
                return;
            }
            ty::ConstKind::Param(data) => {
                self.parameters.push(Parameter::from(data));
            }
            _ => {}
        }

        c.super_visit_with(self)
    }
}

pub(crate) fn identify_constrained_generic_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    predicates: ty::GenericPredicates<'tcx>,
    impl_trait_ref: Option<ty::TraitRef<'tcx>>,
    input_parameters: &mut FxHashSet<Parameter>,
) {
    let mut predicates = predicates.predicates.to_vec();
    setup_constraining_predicates(tcx, &mut predicates, impl_trait_ref, input_parameters);
}

/// Order the predicates in `predicates` such that each parameter is
/// constrained before it is used, if that is possible, and add the
/// parameters so constrained to `input_parameters`. For example,
/// imagine the following impl:
/// ```ignore (illustrative)
/// impl<T: Debug, U: Iterator<Item = T>> Trait for U
/// ```
/// The impl's predicates are collected from left to right. Ignoring
/// the implicit `Sized` bounds, these are
///   * `T: Debug`
///   * `U: Iterator`
///   * `<U as Iterator>::Item = T` -- a desugared ProjectionPredicate
///
/// When we, for example, try to go over the trait-reference
/// `IntoIter<u32> as Trait`, we instantiate the impl parameters with fresh
/// variables and match them with the impl trait-ref, so we know that
/// `$U = IntoIter<u32>`.
///
/// However, in order to process the `$T: Debug` predicate, we must first
/// know the value of `$T` - which is only given by processing the
/// projection. As we occasionally want to process predicates in a single
/// pass, we want the projection to come first. In fact, as projections
/// can (acyclically) depend on one another - see RFC447 for details - we
/// need to topologically sort them.
///
/// We *do* have to be somewhat careful when projection targets contain
/// projections themselves, for example in
///
/// ```ignore (illustrative)
///     impl<S,U,V,W> Trait for U where
/// /* 0 */   S: Iterator<Item = U>,
/// /* - */   U: Iterator,
/// /* 1 */   <U as Iterator>::Item: ToOwned<Owned=(W,<V as Iterator>::Item)>
/// /* 2 */   W: Iterator<Item = V>
/// /* 3 */   V: Debug
/// ```
///
/// we have to evaluate the projections in the order I wrote them:
/// `V: Debug` requires `V` to be evaluated. The only projection that
/// *determines* `V` is 2 (1 contains it, but *does not determine it*,
/// as it is only contained within a projection), but that requires `W`
/// which is determined by 1, which requires `U`, that is determined
/// by 0. I should probably pick a less tangled example, but I can't
/// think of any.
pub(crate) fn setup_constraining_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    predicates: &mut [(ty::Clause<'tcx>, Span)],
    impl_trait_ref: Option<ty::TraitRef<'tcx>>,
    input_parameters: &mut FxHashSet<Parameter>,
) {
    // The canonical way of doing the needed topological sort
    // would be a DFS, but getting the graph and its ownership
    // right is annoying, so I am using an in-place fixed-point iteration,
    // which is `O(nt)` where `t` is the depth of type-parameter constraints,
    // remembering that `t` should be less than 7 in practice.
    //
    // Basically, I iterate over all projections and swap every
    // "ready" projection to the start of the list, such that
    // all of the projections before `i` are topologically sorted
    // and constrain all the parameters in `input_parameters`.
    //
    // In the example, `input_parameters` starts by containing `U` - which
    // is constrained by the trait-ref - and so on the first pass we
    // observe that `<U as Iterator>::Item = T` is a "ready" projection that
    // constrains `T` and swap it to front. As it is the sole projection,
    // no more swaps can take place afterwards, with the result being
    //   * <U as Iterator>::Item = T
    //   * T: Debug
    //   * U: Iterator
    debug!(
        "setup_constraining_predicates: predicates={:?} \
            impl_trait_ref={:?} input_parameters={:?}",
        predicates, impl_trait_ref, input_parameters
    );
    let mut i = 0;
    let mut changed = true;
    while changed {
        changed = false;

        for j in i..predicates.len() {
            // Note that we don't have to care about binders here,
            // as the impl trait ref never contains any late-bound regions.
            if let ty::ClauseKind::Projection(projection) = predicates[j].0.kind().skip_binder() {
                // Special case: watch out for some kind of sneaky attempt
                // to project out an associated type defined by this very
                // trait.
                let unbound_trait_ref = projection.projection_term.trait_ref(tcx);
                if Some(unbound_trait_ref) == impl_trait_ref {
                    continue;
                }

                // A projection depends on its input types and determines its output
                // type. For example, if we have
                //     `<<T as Bar>::Baz as Iterator>::Output = <U as Iterator>::Output`
                // Then the projection only applies if `T` is known, but it still
                // does not determine `U`.
                let inputs = parameters_for(tcx, projection.projection_term, true);
                let relies_only_on_inputs = inputs.iter().all(|p| input_parameters.contains(p));
                if !relies_only_on_inputs {
                    continue;
                }
                input_parameters.extend(parameters_for(tcx, projection.term, false));
            } else {
                continue;
            }
            // fancy control flow to bypass borrow checker
            predicates.swap(i, j);
            i += 1;
            changed = true;
        }
        debug!(
            "setup_constraining_predicates: predicates={:?} \
                i={} impl_trait_ref={:?} input_parameters={:?}",
            predicates, i, impl_trait_ref, input_parameters
        );
    }
}
