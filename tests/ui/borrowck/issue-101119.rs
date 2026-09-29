//@ run-pass

struct State;

fn once(_: impl FnOnce()) {}

fn fill_memory_blocks_mt(state: &mut State) {
    for _ in 0..100 {
        once(move || {
            fill_segment(state);
        });
    }
}

fn fill_segment(_: &mut State) {}

fn main() {
    fill_memory_blocks_mt(&mut State);
}
