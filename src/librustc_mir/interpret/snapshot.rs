use std::hash::{Hash, Hasher};

use rustc::ich::{StableHashingContext, StableHashingContextProvider};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, StableHasherResult};

use super::{Frame, Memory, Machine};

/// The virtual machine state during const-evaluation at a given point in time.
#[derive(Eq, PartialEq)]
pub struct EvalSnapshot<'a, 'mir, 'tcx: 'a + 'mir, M: Machine<'mir, 'tcx>> {
    machine: M,
    memory: Memory<'a, 'mir, 'tcx, M>,
    stack: Vec<Frame<'mir, 'tcx>>,
}

impl<'a, 'mir, 'tcx, M> EvalSnapshot<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
{
    pub fn new(machine: &M, memory: &Memory<'a, 'mir, 'tcx, M>, stack: &[Frame<'mir, 'tcx>]) -> Self {
        EvalSnapshot {
            machine: machine.clone(),
            memory: memory.clone(),
            stack: stack.into(),
        }
    }
}

impl<'a, 'mir, 'tcx, M> Hash for EvalSnapshot<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Implement in terms of hash stable, so that k1 == k2 -> hash(k1) == hash(k2)
        let mut hcx = self.memory.tcx.get_stable_hashing_context();
        let mut hasher = StableHasher::<u64>::new();
        self.hash_stable(&mut hcx, &mut hasher);
        hasher.finish().hash(state)
    }
}

impl<'a, 'b, 'mir, 'tcx, M> HashStable<StableHashingContext<'b>> for EvalSnapshot<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
{
    fn hash_stable<W: StableHasherResult>(&self, hcx: &mut StableHashingContext<'b>, hasher: &mut StableHasher<W>) {
        let EvalSnapshot{ machine, memory, stack } = self;
        (machine, &memory.data, stack).hash_stable(hcx, hasher);
    }
}
