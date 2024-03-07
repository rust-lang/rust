use super::super::prelude::*;
use super::MutInfo;

#[derive(Debug)]
pub struct DfWalker<'a, 'tcx> {
    _info: &'a AnalysisInfo<'tcx>,
    assignments: &'a IndexVec<Local, SmallVec<[MutInfo; 2]>>,
    child: Local,
    maybe_parents: &'a [Local],
    found_parents: Vec<Local>,
    all_const: bool,
}

impl<'a, 'tcx> DfWalker<'a, 'tcx> {
    pub fn new(
        info: &'a AnalysisInfo<'tcx>,
        assignments: &'a IndexVec<Local, SmallVec<[MutInfo; 2]>>,
        child: Local,
        maybe_parents: &'a [Local],
    ) -> Self {
        Self {
            _info: info,
            assignments,
            child,
            maybe_parents,
            found_parents: vec![],
            all_const: true,
        }
    }

    pub fn walk(&mut self) {
        let mut seen = BitSet::new_empty(self.assignments.len());
        let mut stack = Vec::with_capacity(16);
        stack.push(self.child);

        while let Some(parent) = stack.pop() {
            if self.maybe_parents.contains(&parent) {
                self.found_parents.push(parent);
            }

            for assign in &self.assignments[parent] {
                match assign {
                    MutInfo::Dep(sources) | MutInfo::Ctor(sources) => {
                        stack.extend(sources.iter().filter(|local| seen.insert(**local)));
                    },
                    MutInfo::Place(from) | MutInfo::Loan(from) | MutInfo::MutRef(from) => {
                        if matches!(assign, MutInfo::MutRef(_)) {
                            self.all_const = false;
                        }

                        if seen.insert(*from) {
                            stack.push(*from);
                        }
                    },
                    MutInfo::Const => {
                        continue;
                    },
                    MutInfo::Calc | MutInfo::Arg => {
                        self.all_const = false;
                        continue;
                    },
                }
            }
        }
    }

    pub fn connection_count(&self) -> usize {
        self.found_parents.len()
    }

    pub fn found_connection(&self, maybe_parent: Local) -> bool {
        self.found_parents.contains(&maybe_parent)
    }

    pub fn all_const(&self) -> bool {
        self.all_const
    }
}
