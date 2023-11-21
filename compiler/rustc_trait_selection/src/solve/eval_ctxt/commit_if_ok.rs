use super::EvalCtxt;
use crate::solve::inspect;
use rustc_middle::traits::query::NoSolution;

impl<'a, 'tcx> EvalCtxt<'a, 'tcx> {
    pub(in crate::solve) fn commit_if_ok<T>(
        &mut self,
        f: impl FnOnce(&mut EvalCtxt<'_, 'tcx>) -> Result<T, NoSolution>,
    ) -> Result<T, NoSolution> {
        let mut nested_ecx = EvalCtxt {
            infcx: self.infcx,
            variables: self.variables,
            var_values: self.var_values,
            predefined_opaques_in_body: self.predefined_opaques_in_body,
            max_input_universe: self.max_input_universe,
            search_graph: self.search_graph,
            nested_goals: self.nested_goals.clone(),
            tainted: self.tainted,
            inspect: self.inspect.new_probe(),
        };

        let result = nested_ecx.infcx.commit_if_ok(|_| f(&mut nested_ecx));
        if result.is_ok() {
            let EvalCtxt {
                infcx: _,
                variables: _,
                var_values: _,
                predefined_opaques_in_body: _,
                max_input_universe: _,
                search_graph: _,
                nested_goals,
                tainted,
                inspect,
            } = nested_ecx;
            self.nested_goals = nested_goals;
            self.tainted = tainted;
            self.inspect.integrate_snapshot(inspect);
        } else {
            nested_ecx.inspect.probe_kind(inspect::ProbeKind::CommitIfOk);
            self.inspect.finish_probe(nested_ecx.inspect);
        }

        result
    }
}
