use crate::solve::assembly::Candidate;

use super::EvalCtxt;
use rustc_next_trait_solver::solve::{
    inspect, BuiltinImplSource, CandidateSource, NoSolution, QueryResult,
};
use rustc_type_ir::{InferCtxtLike, Interner};
use std::marker::PhantomData;

pub(in crate::solve) struct ProbeCtxt<'me, 'a, Infcx, I, F, T>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    ecx: &'me mut EvalCtxt<'a, Infcx, I>,
    probe_kind: F,
    _result: PhantomData<T>,
}

impl<Infcx, I, F, T> ProbeCtxt<'_, '_, Infcx, I, F, T>
where
    F: FnOnce(&T) -> inspect::ProbeKind<I>,
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    pub(in crate::solve) fn enter(self, f: impl FnOnce(&mut EvalCtxt<'_, Infcx>) -> T) -> T {
        let ProbeCtxt { ecx: outer_ecx, probe_kind, _result } = self;

        let infcx = outer_ecx.infcx;
        let max_input_universe = outer_ecx.max_input_universe;
        let mut nested_ecx = EvalCtxt {
            infcx,
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
        let r = nested_ecx.infcx.probe(|| {
            let r = f(&mut nested_ecx);
            nested_ecx.inspect.probe_final_state(infcx, max_input_universe);
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

pub(in crate::solve) struct TraitProbeCtxt<'me, 'a, Infcx, I, F>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    cx: ProbeCtxt<'me, 'a, Infcx, I, F, QueryResult<I>>,
    source: CandidateSource<I>,
}

impl<Infcx, I, F> TraitProbeCtxt<'_, '_, Infcx, I, F>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    F: FnOnce(&QueryResult<I>) -> inspect::ProbeKind<I>,
{
    #[instrument(level = "debug", skip_all, fields(source = ?self.source))]
    pub(in crate::solve) fn enter(
        self,
        f: impl FnOnce(&mut EvalCtxt<'_, Infcx>) -> QueryResult<I>,
    ) -> Result<Candidate<I>, NoSolution> {
        self.cx.enter(|ecx| f(ecx)).map(|result| Candidate { source: self.source, result })
    }
}

impl<'a, Infcx, I> EvalCtxt<'a, Infcx, I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    /// `probe_kind` is only called when proof tree building is enabled so it can be
    /// as expensive as necessary to output the desired information.
    pub(in crate::solve) fn probe<F, T>(
        &mut self,
        probe_kind: F,
    ) -> ProbeCtxt<'_, 'a, Infcx, I, F, T>
    where
        F: FnOnce(&T) -> inspect::ProbeKind<I>,
    {
        ProbeCtxt { ecx: self, probe_kind, _result: PhantomData }
    }

    pub(in crate::solve) fn probe_builtin_trait_candidate(
        &mut self,
        source: BuiltinImplSource,
    ) -> TraitProbeCtxt<'_, 'a, Infcx, I, impl FnOnce(&QueryResult<I>) -> inspect::ProbeKind<I>>
    {
        self.probe_trait_candidate(CandidateSource::BuiltinImpl(source))
    }

    pub(in crate::solve) fn probe_trait_candidate(
        &mut self,
        source: CandidateSource<I>,
    ) -> TraitProbeCtxt<'_, 'a, Infcx, I, impl FnOnce(&QueryResult<I>) -> inspect::ProbeKind<I>>
    {
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
