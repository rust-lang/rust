// Tests the suggestion to reborrow the first move site
// when we move then borrow a `&mut` ref.

struct State;

impl IntoIterator for &mut State {
    type IntoIter = std::vec::IntoIter<()>;
    type Item = ();

    fn into_iter(self) -> Self::IntoIter {
        vec![].into_iter()
    }
}

fn once(f: impl FnOnce()) {}

fn fill_memory_blocks_mt(state: &mut State) {
    for _ in state {}
    //~^ HELP consider creating a fresh reborrow of `state` here
    fill_segment(state);
    //~^ ERROR borrow of moved value: `state`
}

fn fill_segment(state: &mut State) {}

fn main() {}
