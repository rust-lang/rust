use std::collections::HashMap;
use std::hash::Hash;

trait LowT: Identify {}

trait Identify {
    type Id: Clone + Hash + PartialEq + Eq;
    fn identify(&self) -> Self::Id;
}

struct MapStore<L, I>
where
    L: LowT + Identify<Id = I>,
{
    lows: HashMap<I, L>,
}

impl<L, I> MapStore<L, I>
where
    L: LowT + Identify<Id = I>,
    I: Clone + Hash + PartialEq + Eq,
{
    fn remove_low(&mut self, low: &impl LowT) {
        let _low = self.lows.remove(low.identify()).unwrap(); //~ ERROR mismatched types
    }
}

fn main() {}
