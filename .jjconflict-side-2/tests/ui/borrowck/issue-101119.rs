struct State;

fn once(_: impl FnOnce()) {}

fn fill_memory_blocks_mt(state: &mut State) {
    loop {
        once(move || {
            //~^ ERROR use of moved value: `state`
            fill_segment(state);
        });
    }
}

fn fill_segment(_: &mut State) {}

fn main() {}
