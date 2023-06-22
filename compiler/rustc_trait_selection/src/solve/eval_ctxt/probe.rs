use super::EvalCtxt;
use rustc_middle::traits::solve::inspect;
use std::marker::PhantomData;

pub(in crate::solve) struct ProbeCtxt<'me, 'a, 'tcx, F, T> {
    ecx: &'me mut EvalCtxt<'a, 'tcx>,
    probe_kind: F,
    _result: PhantomData<T>,
}

impl<'tcx, F, T> ProbeCtxt<'_, '_, 'tcx, F, T>
where
    F: FnOnce(&T) -> inspect::CandidateKind<'tcx>,
{
    pub(in crate::solve) fn enter(self, f: impl FnOnce(&mut EvalCtxt<'_, 'tcx>) -> T) -> T {
        let ProbeCtxt { ecx: outer_ecx, probe_kind, _result } = self;

        let mut nested_ecx = EvalCtxt {
            infcx: outer_ecx.infcx,
            var_values: outer_ecx.var_values,
            predefined_opaques_in_body: outer_ecx.predefined_opaques_in_body,
            max_input_universe: outer_ecx.max_input_universe,
            search_graph: outer_ecx.search_graph,
            nested_goals: outer_ecx.nested_goals.clone(),
            tainted: outer_ecx.tainted,
            inspect: outer_ecx.inspect.new_goal_candidate(),
        };
        let r = nested_ecx.infcx.probe(|_| f(&mut nested_ecx));
        if !outer_ecx.inspect.is_noop() {
            let cand_kind = probe_kind(&r);
            nested_ecx.inspect.candidate_kind(cand_kind);
            outer_ecx.inspect.goal_candidate(nested_ecx.inspect);
        }
        r
    }
}

impl<'a, 'tcx> EvalCtxt<'a, 'tcx> {
    /// `probe_kind` is only called when proof tree building is enabled so it can be
    /// as expensive as necessary to output the desired information.
    pub(in crate::solve) fn probe<F, T>(&mut self, probe_kind: F) -> ProbeCtxt<'_, 'a, 'tcx, F, T>
    where
        F: FnOnce(&T) -> inspect::CandidateKind<'tcx>,
    {
        ProbeCtxt { ecx: self, probe_kind, _result: PhantomData }
    }
}
