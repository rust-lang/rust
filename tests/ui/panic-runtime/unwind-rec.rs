//@ run-fail
//@ check-run-results:explicit panic
//@ ignore-emscripten no processes

fn build() -> Vec<isize> {
    panic!();
}

struct Blk {
    node: Vec<isize>,
}

fn main() {
    let _blk = Blk { node: build() };
}
