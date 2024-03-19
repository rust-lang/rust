use crate::solve::assembly::Candidate;

use super::EvalCtxt;
use rustc_middle::traits::{
    query::NoSolution,
    solve::{inspect, CandidateSource, QueryResult},
};
use std::marker::PhantomData;

pub(in crate::solve) struct ProbeCtxt<'me, 'a, 'tcx, F, T> {
    ecx: &'me mut EvalCtxt<'a, 'tcx>,
    probe_kind: F,
    _result: PhantomData<T>,
}

impl<'tcx, F, T> ProbeCtxt<'_, '_, 'tcx, F, T>
where
    F: FnOnce(&T) -> inspect::ProbeKind<'tcx>,
{
    pub(in crate::solve) fn enter(self, f: impl FnOnce(&mut EvalCtxt<'_, 'tcx>) -> T) -> T {
        let ProbeCtxt { ecx: outer_ecx, probe_kind, _result } = self;

        let mut nested_ecx = EvalCtxt {
            infcx: outer_ecx.infcx,
            variables: outer_ecx.variables,
            var_values: outer_ecx.var_values,
            is_normalizes_to_goal: outer_ecx.is_normalizes_to_goal,
            predefined_opaques_in_body: outer_ecx.predefined_opaques_in_body,
            max_input_universe: outer_ecx.max_input_universe,
            search_graph: outer_ecx.search_graph,
            nested_goals: outer_ecx.nested_goals.clone(),
            tainted: outer_ecx.tainted,
            inspect: outer_ecx.inspect.new_probe(),
        };
        let r = nested_ecx.infcx.probe(|_| f(&mut nested_ecx));
        if !outer_ecx.inspect.is_noop() {
            let probe_kind = probe_kind(&r);
            nested_ecx.inspect.probe_kind(probe_kind);
            outer_ecx.inspect.finish_probe(nested_ecx.inspect);
        }
        r
    }
}

pub(in crate::solve) struct TraitProbeCtxt<'me, 'a, 'tcx, F> {
    cx: ProbeCtxt<'me, 'a, 'tcx, F, QueryResult<'tcx>>,
    source: CandidateSource,
}

impl<'tcx, F> TraitProbeCtxt<'_, '_, 'tcx, F>
where
    F: FnOnce(&QueryResult<'tcx>) -> inspect::ProbeKind<'tcx>,
{
    pub(in crate::solve) fn enter(
        self,
        f: impl FnOnce(&mut EvalCtxt<'_, 'tcx>) -> QueryResult<'tcx>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        self.cx.enter(|ecx| f(ecx)).map(|result| Candidate { source: self.source, result })
    }
}

impl<'a, 'tcx> EvalCtxt<'a, 'tcx> {
    /// `probe_kind` is only called when proof tree building is enabled so it can be
    /// as expensive as necessary to output the desired information.
    pub(in crate::solve) fn probe<F, T>(&mut self, probe_kind: F) -> ProbeCtxt<'_, 'a, 'tcx, F, T>
    where
        F: FnOnce(&T) -> inspect::ProbeKind<'tcx>,
    {
        ProbeCtxt { ecx: self, probe_kind, _result: PhantomData }
    }

    pub(in crate::solve) fn probe_misc_candidate(
        &mut self,
        name: &'static str,
    ) -> ProbeCtxt<
        '_,
        'a,
        'tcx,
        impl FnOnce(&QueryResult<'tcx>) -> inspect::ProbeKind<'tcx>,
        QueryResult<'tcx>,
    > {
        ProbeCtxt {
            ecx: self,
            probe_kind: move |result: &QueryResult<'tcx>| inspect::ProbeKind::MiscCandidate {
                name,
                result: *result,
            },
            _result: PhantomData,
        }
    }

    pub(in crate::solve) fn probe_trait_candidate(
        &mut self,
        source: CandidateSource,
    ) -> TraitProbeCtxt<'_, 'a, 'tcx, impl FnOnce(&QueryResult<'tcx>) -> inspect::ProbeKind<'tcx>>
    {
        TraitProbeCtxt {
            cx: ProbeCtxt {
                ecx: self,
                probe_kind: move |result: &QueryResult<'tcx>| inspect::ProbeKind::TraitCandidate {
                    source,
                    result: *result,
                },
                _result: PhantomData,
            },
            source,
        }
    }
}
