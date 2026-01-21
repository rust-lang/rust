use rustc_type_ir::{solve::GoalSource, solve::inspect::GoalEvaluation};
use serde_derive::{Deserialize, Serialize};

use crate::next_solver::inspect::{InspectCandidate, InspectGoal};
use crate::next_solver::{AnyImplId, infer::InferCtxt};
use crate::next_solver::{DbInterner, Span};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofTreeData {
    pub goal: String,
    pub result: String,
    pub depth: usize,
    pub candidates: Vec<CandidateData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateData {
    pub kind: String,
    pub result: String,
    pub impl_header: Option<String>,
    pub nested_goals: Vec<ProofTreeData>,
}

pub fn dump_proof_tree_structured<'db>(
    proof_tree: GoalEvaluation<DbInterner<'db>>,
    _span: Span,
    infcx: &InferCtxt<'db>,
) -> ProofTreeData {
    let goal_eval = InspectGoal::new(infcx, 0, proof_tree, None, GoalSource::Misc);
    let mut serializer = ProofTreeSerializer::new(infcx);
    serializer.serialize_goal(&goal_eval)
}

struct ProofTreeSerializer<'a, 'db> {
    infcx: &'a InferCtxt<'db>,
}

impl<'a, 'db> ProofTreeSerializer<'a, 'db> {
    fn new(infcx: &'a InferCtxt<'db>) -> Self {
        Self { infcx }
    }

    fn serialize_goal(&mut self, goal: &InspectGoal<'_, 'db>) -> ProofTreeData {
        let candidates = goal.candidates();
        let candidates_data: Vec<CandidateData> =
            candidates.iter().map(|c| self.serialize_candidate(c)).collect();

        ProofTreeData {
            goal: format!("{:?}", goal.goal()),
            result: format!("{:?}", goal.result()),
            depth: goal.depth(),
            candidates: candidates_data,
        }
    }

    fn serialize_candidate(&mut self, candidate: &InspectCandidate<'_, 'db>) -> CandidateData {
        let kind = candidate.kind();
        let impl_header = self.get_impl_header(candidate);

        let mut nested = Vec::new();
        self.infcx.probe(|_| {
            for nested_goal in candidate.instantiate_nested_goals() {
                nested.push(self.serialize_goal(&nested_goal));
            }
        });

        CandidateData {
            kind: format!("{:?}", kind),
            result: format!("{:?}", candidate.result()),
            impl_header,
            nested_goals: nested,
        }
    }

    fn get_impl_header(&self, candidate: &InspectCandidate<'_, 'db>) -> Option<String> {
        use rustc_type_ir::solve::inspect::ProbeKind;
        match candidate.kind() {
            ProbeKind::TraitCandidate { source, .. } => {
                use hir_def::{Lookup, src::HasSource};
                use rustc_type_ir::solve::CandidateSource;
                let db = self.infcx.interner.db;
                match source {
                    CandidateSource::Impl(impl_def_id) => match impl_def_id {
                        AnyImplId::ImplId(impl_def_id) => {
                            let impl_src = impl_def_id.lookup(db).source(db);
                            Some(impl_src.value.to_string())
                        }
                        AnyImplId::BuiltinDeriveImplId(impl_id) => {
                            let impl_loc = impl_id.loc(db);
                            let adt_src = match impl_loc.adt {
                                hir_def::AdtId::StructId(adt) => {
                                    adt.loc(db).source(db).value.to_string()
                                }
                                hir_def::AdtId::UnionId(adt) => {
                                    adt.loc(db).source(db).value.to_string()
                                }
                                hir_def::AdtId::EnumId(adt) => {
                                    adt.loc(db).source(db).value.to_string()
                                }
                            };
                            Some(format!("#[derive(${})]\n{}", impl_loc.trait_.name(), adt_src))
                        }
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }
}
