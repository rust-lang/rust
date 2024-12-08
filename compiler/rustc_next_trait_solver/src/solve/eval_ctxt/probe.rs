use std::marker::PhantomData;

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
    pub(in crate::solve) fn enter(self, f: impl FnOnce(&mut EvalCtxt<'_, D>) -> T) -> T {
        let ProbeCtxt { ecx: outer_ecx, probe_kind, _result } = self;

        let delegate = outer_ecx.delegate;
        let max_input_universe = outer_ecx.max_input_universe;
        let mut nested_ecx = EvalCtxt {
            delegate,
            variables: outer_ecx.variables,
            var_values: outer_ecx.var_values,
            is_normalizes_to_goal: outer_ecx.is_normalizes_to_goal,
            predefined_opaques_in_body: outer_ecx.predefined_opaques_in_body,
            max_input_universe,
            search_graph: outer_ecx.search_graph,
            nested_goals: outer_ecx.nested_goals.clone(),
            tainted: outer_ecx.tainted,
            inspect: outer_ecx.inspect.take_and_enter_probe(),
        };
        let r = nested_ecx.delegate.probe(|| {
            let r = f(&mut nested_ecx);
            nested_ecx.inspect.probe_final_state(delegate, max_input_universe);
            r
        });
        if !nested_ecx.inspect.is_noop() {
            let probe_kind = probe_kind(&r);
            nested_ecx.inspect.probe_kind(probe_kind);
            outer_ecx.inspect = nested_ecx.inspect.finish_probe();
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
        self.cx.enter(|ecx| f(ecx)).map(|result| Candidate { source: self.source, result })
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
