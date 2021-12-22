// aux-build:upstream.rs
// revisions: rpass1 rpass2

extern crate upstream;

struct Wrapper;

impl Wrapper {
    fn bar() {
        let val: upstream::Foo;
    }
}

fn main() {}
