// run-fail
//@error-in-other-file:explicit panic
//@ignore-target-emscripten no processes

fn build() -> Vec<isize> {
    panic!();
}

struct Blk {
    node: Vec<isize>,
}

fn main() {
    let _blk = Blk { node: build() };
}
