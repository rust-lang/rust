use std::marker::PhantomData;

use rustc_type_ir::search_graph::CandidateHeadUsages;
use rustc_type_ir::{InferCtxtLike, Interner};
use tracing::instrument;

use crate::delegate::SolverDelegate;
use crate::solve::assembly::Candidate;
use crate::solve::{
    BuiltinImplSource, CandidateSource, EvalCtxt, NoSolution, QueryResult, inspect,
};

pub(in crate::solve) struct ProbeCtxt<'me, 'a, D, I, F, T>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    ecx: &'me mut EvalCtxt<'a, D, I>,
    probe_kind: F,
    _result: PhantomData<T>,
}

impl<D, I, F, T> ProbeCtxt<'_, '_, D, I, F, T>
where
    F: FnOnce(&T) -> inspect::ProbeKind<I>,
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    pub(in crate::solve) fn enter_single_candidate(
        self,
        f: impl FnOnce(&mut EvalCtxt<'_, D>) -> T,
    ) -> (T, CandidateHeadUsages) {
        self.ecx.search_graph.enter_single_candidate();
        let mut candidate_usages = CandidateHeadUsages::default();
        let result = self.enter(|ecx| {
            let result = f(ecx);
            candidate_usages = ecx.search_graph.finish_single_candidate();
            result
        });
        (result, candidate_usages)
    }

    pub(in crate::solve) fn enter(self, f: impl FnOnce(&mut EvalCtxt<'_, D>) -> T) -> T {
        let ProbeCtxt { ecx: outer, probe_kind, _result } = self;

        let delegate = outer.delegate;
        let max_input_universe = outer.max_input_universe;
        let mut nested = EvalCtxt {
            delegate,
            variables: outer.variables,
            var_values: outer.var_values,
            current_goal_kind: outer.current_goal_kind,
            max_input_universe,
            initial_opaque_types_storage_num_entries: outer
                .initial_opaque_types_storage_num_entries,
            search_graph: outer.search_graph,
            nested_goals: outer.nested_goals.clone(),
            origin_span: outer.origin_span,
            tainted: outer.tainted,
            inspect: outer.inspect.take_and_enter_probe(),
        };
        let r = nested.delegate.probe(|| {
            let r = f(&mut nested);
            nested.inspect.probe_final_state(delegate, max_input_universe);
            r
        });
        if !nested.inspect.is_noop() {
            let probe_kind = probe_kind(&r);
            nested.inspect.probe_kind(probe_kind);
            outer.inspect = nested.inspect.finish_probe();
        }
        r
    }
}

pub(in crate::solve) struct TraitProbeCtxt<'me, 'a, D, I, F>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    cx: ProbeCtxt<'me, 'a, D, I, F, QueryResult<I>>,
    source: CandidateSource<I>,
}

impl<D, I, F> TraitProbeCtxt<'_, '_, D, I, F>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
    F: FnOnce(&QueryResult<I>) -> inspect::ProbeKind<I>,
{
    #[instrument(level = "debug", skip_all, fields(source = ?self.source))]
    pub(in crate::solve) fn enter(
        self,
        f: impl FnOnce(&mut EvalCtxt<'_, D>) -> QueryResult<I>,
    ) -> Result<Candidate<I>, NoSolution> {
        let (result, head_usages) = self.cx.enter_single_candidate(f);
        result.map(|result| Candidate { source: self.source, result, head_usages })
    }
}

impl<'a, D, I> EvalCtxt<'a, D, I>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    /// `probe_kind` is only called when proof tree building is enabled so it can be
    /// as expensive as necessary to output the desired information.
    pub(in crate::solve) fn probe<F, T>(&mut self, probe_kind: F) -> ProbeCtxt<'_, 'a, D, I, F, T>
    where
        F: FnOnce(&T) -> inspect::ProbeKind<I>,
    {
        ProbeCtxt { ecx: self, probe_kind, _result: PhantomData }
    }

    pub(in crate::solve) fn probe_builtin_trait_candidate(
        &mut self,
        source: BuiltinImplSource,
    ) -> TraitProbeCtxt<'_, 'a, D, I, impl FnOnce(&QueryResult<I>) -> inspect::ProbeKind<I>> {
        self.probe_trait_candidate(CandidateSource::BuiltinImpl(source))
    }

    pub(in crate::solve) fn probe_trait_candidate(
        &mut self,
        source: CandidateSource<I>,
    ) -> TraitProbeCtxt<'_, 'a, D, I, impl FnOnce(&QueryResult<I>) -> inspect::ProbeKind<I>> {
        TraitProbeCtxt {
            cx: ProbeCtxt {
                ecx: self,
                probe_kind: move |result: &QueryResult<I>| inspect::ProbeKind::TraitCandidate {
                    source,
                    result: *result,
                },
                _result: PhantomData,
            },
            source,
        }
    }
}
