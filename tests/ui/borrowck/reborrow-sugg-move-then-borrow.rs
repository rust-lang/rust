//@ run-pass

struct State;

impl IntoIterator for &mut State {
    type IntoIter = std::vec::IntoIter<()>;
    type Item = ();

    fn into_iter(self) -> Self::IntoIter {
        vec![].into_iter()
    }
}

fn fill_memory_blocks_mt(state: &mut State) {
    for _ in state {}
    fill_segment(state);
}

fn fill_segment(_state: &mut State) {}

fn main() {
    fill_memory_blocks_mt(&mut State);
}
