//! Canonicalization is used to separate some goal from its context,
//! throwing away unnecessary information in the process.
//!
//! This is necessary to cache goals containing inference variables
//! and placeholders without restricting them to the current `InferCtxt`.
//!
//! Canonicalization is fairly involved, for more details see the relevant
//! section of the [rustc-dev-guide][c].
//!
//! [c]: https://rustc-dev-guide.rust-lang.org/solve/canonicalization.html

use std::iter;

use canonicalizer::Canonicalizer;
use rustc_index::IndexVec;
use rustc_type_ir::inherent::*;
use rustc_type_ir::relate::solver_relating::RelateExt;
use rustc_type_ir::{
    self as ty, Canonical, CanonicalVarKind, CanonicalVarValues, InferCtxtLike, Interner,
    TypeFoldable,
};
use tracing::instrument;

use crate::delegate::SolverDelegate;
use crate::resolve::eager_resolve_vars;
use crate::solve::{
    CanonicalInput, CanonicalResponse, Certainty, ExternalConstraintsData, Goal,
    NestedNormalizationGoals, QueryInput, Response, inspect,
};

pub mod canonicalizer;

trait ResponseT<I: Interner> {
    fn var_values(&self) -> CanonicalVarValues<I>;
}

impl<I: Interner> ResponseT<I> for Response<I> {
    fn var_values(&self) -> CanonicalVarValues<I> {
        self.var_values
    }
}

impl<I: Interner, T> ResponseT<I> for inspect::State<I, T> {
    fn var_values(&self) -> CanonicalVarValues<I> {
        self.var_values
    }
}

/// Canonicalizes the goal remembering the original values
/// for each bound variable.
///
/// This expects `goal` and `opaque_types` to be eager resolved.
pub(super) fn canonicalize_goal<D, I>(
    delegate: &D,
    goal: Goal<I, I::Predicate>,
    opaque_types: &[(ty::OpaqueTypeKey<I>, I::Ty)],
) -> (Vec<I::GenericArg>, CanonicalInput<I, I::Predicate>)
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    let mut orig_values = Default::default();
    let canonical = Canonicalizer::canonicalize_input(
        delegate,
        &mut orig_values,
        QueryInput {
            goal,
            predefined_opaques_in_body: delegate.cx().mk_predefined_opaques_in_body(opaque_types),
        },
    );
    let query_input = ty::CanonicalQueryInput { canonical, typing_mode: delegate.typing_mode() };
    (orig_values, query_input)
}

pub(super) fn canonicalize_response<D, I, T>(
    delegate: &D,
    max_input_universe: ty::UniverseIndex,
    value: T,
) -> ty::Canonical<I, T>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
    T: TypeFoldable<I>,
{
    let mut orig_values = Default::default();
    let canonical =
        Canonicalizer::canonicalize_response(delegate, max_input_universe, &mut orig_values, value);
    canonical
}

/// After calling a canonical query, we apply the constraints returned
/// by the query using this function.
///
/// This happens in three steps:
/// - we instantiate the bound variables of the query response
/// - we unify the `var_values` of the response with the `original_values`
/// - we apply the `external_constraints` returned by the query, returning
///   the `normalization_nested_goals`
pub(super) fn instantiate_and_apply_query_response<D, I>(
    delegate: &D,
    param_env: I::ParamEnv,
    original_values: &[I::GenericArg],
    response: CanonicalResponse<I>,
    span: I::Span,
) -> (NestedNormalizationGoals<I>, Certainty)
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    let instantiation =
        compute_query_response_instantiation_values(delegate, &original_values, &response, span);

    let Response { var_values, external_constraints, certainty } =
        delegate.instantiate_canonical(response, instantiation);

    unify_query_var_values(delegate, param_env, &original_values, var_values, span);

    let ExternalConstraintsData { region_constraints, opaque_types, normalization_nested_goals } =
        &*external_constraints;

    register_region_constraints(delegate, region_constraints, span);
    register_new_opaque_types(delegate, opaque_types, span);

    (normalization_nested_goals.clone(), certainty)
}

/// This returns the canonical variable values to instantiate the bound variables of
/// the canonical response. This depends on the `original_values` for the
/// bound variables.
fn compute_query_response_instantiation_values<D, I, T>(
    delegate: &D,
    original_values: &[I::GenericArg],
    response: &Canonical<I, T>,
    span: I::Span,
) -> CanonicalVarValues<I>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
    T: ResponseT<I>,
{
    // FIXME: Longterm canonical queries should deal with all placeholders
    // created inside of the query directly instead of returning them to the
    // caller.
    let prev_universe = delegate.universe();
    let universes_created_in_query = response.max_universe.index();
    for _ in 0..universes_created_in_query {
        delegate.create_next_universe();
    }

    let var_values = response.value.var_values();
    assert_eq!(original_values.len(), var_values.len());

    // If the query did not make progress with constraining inference variables,
    // we would normally create a new inference variables for bound existential variables
    // only then unify this new inference variable with the inference variable from
    // the input.
    //
    // We therefore instantiate the existential variable in the canonical response with the
    // inference variable of the input right away, which is more performant.
    let mut opt_values = IndexVec::from_elem_n(None, response.variables.len());
    for (original_value, result_value) in iter::zip(original_values, var_values.var_values.iter()) {
        match result_value.kind() {
            ty::GenericArgKind::Type(t) => {
                // We disable the instantiation guess for inference variables
                // and only use it for placeholders. We need to handle the
                // `sub_root` of type inference variables which would make this
                // more involved. They are also a lot rarer than region variables.
                if let ty::Bound(index_kind, b) = t.kind()
                    && !matches!(
                        response.variables.get(b.var().as_usize()).unwrap(),
                        CanonicalVarKind::Ty { .. }
                    )
                {
                    assert!(matches!(index_kind, ty::BoundVarIndexKind::Canonical));
                    opt_values[b.var()] = Some(*original_value);
                }
            }
            ty::GenericArgKind::Lifetime(r) => {
                if let ty::ReBound(index_kind, br) = r.kind() {
                    assert!(matches!(index_kind, ty::BoundVarIndexKind::Canonical));
                    opt_values[br.var()] = Some(*original_value);
                }
            }
            ty::GenericArgKind::Const(c) => {
                if let ty::ConstKind::Bound(index_kind, bv) = c.kind() {
                    assert!(matches!(index_kind, ty::BoundVarIndexKind::Canonical));
                    opt_values[bv.var()] = Some(*original_value);
                }
            }
        }
    }
    CanonicalVarValues::instantiate(delegate.cx(), response.variables, |var_values, kind| {
        if kind.universe() != ty::UniverseIndex::ROOT {
            // A variable from inside a binder of the query. While ideally these shouldn't
            // exist at all (see the FIXME at the start of this method), we have to deal with
            // them for now.
            delegate.instantiate_canonical_var(kind, span, &var_values, |idx| {
                prev_universe + idx.index()
            })
        } else if kind.is_existential() {
            // As an optimization we sometimes avoid creating a new inference variable here.
            //
            // All new inference variables we create start out in the current universe of the caller.
            // This is conceptually wrong as these inference variables would be able to name
            // more placeholders then they should be able to. However the inference variables have
            // to "come from somewhere", so by equating them with the original values of the caller
            // later on, we pull them down into their correct universe again.
            if let Some(v) = opt_values[ty::BoundVar::from_usize(var_values.len())] {
                v
            } else {
                delegate.instantiate_canonical_var(kind, span, &var_values, |_| prev_universe)
            }
        } else {
            // For placeholders which were already part of the input, we simply map this
            // universal bound variable back the placeholder of the input.
            original_values[kind.expect_placeholder_index()]
        }
    })
}

/// Unify the `original_values` with the `var_values` returned by the canonical query..
///
/// This assumes that this unification will always succeed. This is the case when
/// applying a query response right away. However, calling a canonical query, doing any
/// other kind of trait solving, and only then instantiating the result of the query
/// can cause the instantiation to fail. This is not supported and we ICE in this case.
///
/// We always structurally instantiate aliases. Relating aliases needs to be different
/// depending on whether the alias is *rigid* or not. We're only really able to tell
/// whether an alias is rigid by using the trait solver. When instantiating a response
/// from the solver we assume that the solver correctly handled aliases and therefore
/// always relate them structurally here.
#[instrument(level = "trace", skip(delegate))]
fn unify_query_var_values<D, I>(
    delegate: &D,
    param_env: I::ParamEnv,
    original_values: &[I::GenericArg],
    var_values: CanonicalVarValues<I>,
    span: I::Span,
) where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    assert_eq!(original_values.len(), var_values.len());

    for (&orig, response) in iter::zip(original_values, var_values.var_values.iter()) {
        let goals =
            delegate.eq_structurally_relating_aliases(param_env, orig, response, span).unwrap();
        assert!(goals.is_empty());
    }
}

fn register_region_constraints<D, I>(
    delegate: &D,
    outlives: &[ty::OutlivesPredicate<I, I::GenericArg>],
    span: I::Span,
) where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    for &ty::OutlivesPredicate(lhs, rhs) in outlives {
        match lhs.kind() {
            ty::GenericArgKind::Lifetime(lhs) => delegate.sub_regions(rhs, lhs, span),
            ty::GenericArgKind::Type(lhs) => delegate.register_ty_outlives(lhs, rhs, span),
            ty::GenericArgKind::Const(_) => panic!("const outlives: {lhs:?}: {rhs:?}"),
        }
    }
}

fn register_new_opaque_types<D, I>(
    delegate: &D,
    opaque_types: &[(ty::OpaqueTypeKey<I>, I::Ty)],
    span: I::Span,
) where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    for &(key, ty) in opaque_types {
        let prev = delegate.register_hidden_type_in_storage(key, ty, span);
        // We eagerly resolve inference variables when computing the query response.
        // This can cause previously distinct opaque type keys to now be structurally equal.
        //
        // To handle this, we store any duplicate entries in a separate list to check them
        // at the end of typeck/borrowck. We could alternatively eagerly equate the hidden
        // types here. However, doing so is difficult as it may result in nested goals and
        // any errors may make it harder to track the control flow for diagnostics.
        if let Some(prev) = prev {
            delegate.add_duplicate_opaque_type(key, prev, span);
        }
    }
}

/// Used by proof trees to be able to recompute intermediate actions while
/// evaluating a goal. The `var_values` not only include the bound variables
/// of the query input, but also contain all unconstrained inference vars
/// created while evaluating this goal.
pub fn make_canonical_state<D, I, T>(
    delegate: &D,
    var_values: &[I::GenericArg],
    max_input_universe: ty::UniverseIndex,
    data: T,
) -> inspect::CanonicalState<I, T>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
    T: TypeFoldable<I>,
{
    let var_values = CanonicalVarValues { var_values: delegate.cx().mk_args(var_values) };
    let state = inspect::State { var_values, data };
    let state = eager_resolve_vars(delegate, state);
    Canonicalizer::canonicalize_response(delegate, max_input_universe, &mut vec![], state)
}

// FIXME: needs to be pub to be accessed by downstream
// `rustc_trait_selection::solve::inspect::analyse`.
pub fn instantiate_canonical_state<D, I, T>(
    delegate: &D,
    span: I::Span,
    param_env: I::ParamEnv,
    orig_values: &mut Vec<I::GenericArg>,
    state: inspect::CanonicalState<I, T>,
) -> T
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
    T: TypeFoldable<I>,
{
    // In case any fresh inference variables have been created between `state`
    // and the previous instantiation, extend `orig_values` for it.
    orig_values.extend(
        state.value.var_values.var_values.as_slice()[orig_values.len()..]
            .iter()
            .map(|&arg| delegate.fresh_var_for_kind_with_span(arg, span)),
    );

    let instantiation =
        compute_query_response_instantiation_values(delegate, orig_values, &state, span);

    let inspect::State { var_values, data } = delegate.instantiate_canonical(state, instantiation);

    unify_query_var_values(delegate, param_env, orig_values, var_values, span);
    data
}

pub fn response_no_constraints_raw<I: Interner>(
    cx: I,
    max_universe: ty::UniverseIndex,
    variables: I::CanonicalVarKinds,
    certainty: Certainty,
) -> CanonicalResponse<I> {
    ty::Canonical {
        max_universe,
        variables,
        value: Response {
            var_values: ty::CanonicalVarValues::make_identity(cx, variables),
            // FIXME: maybe we should store the "no response" version in cx, like
            // we do for cx.types and stuff.
            external_constraints: cx.mk_external_constraints(ExternalConstraintsData::default()),
            certainty,
        },
    }
}
