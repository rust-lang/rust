//@ run-fail
//@ error-pattern:explicit panic
//@ needs-subprocess

fn build() -> Vec<isize> {
    panic!();
}

struct Blk {
    node: Vec<isize>,
}

fn main() {
    let _blk = Blk { node: build() };
}
