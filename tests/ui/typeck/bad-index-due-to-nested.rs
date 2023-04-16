use std::collections::HashMap;

pub struct Graph<V> {
    node_index_map: HashMap<V, usize>,
}

impl<V> Graph<V> {
    pub fn node_index(&self, node: V) -> usize {
        self.node_index_map[&node]
        //~^ ERROR the trait bound `V: Eq` is not satisfied
        //~| ERROR the trait bound `V: Hash` is not satisfied
    }
}

fn main() {}
