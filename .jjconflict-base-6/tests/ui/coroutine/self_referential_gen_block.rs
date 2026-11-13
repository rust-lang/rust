//@ edition: 2024
#![feature(gen_blocks)]
//! This test checks that we don't allow self-referential generators

fn main() {
    let mut x = {
        let mut x = gen {
            let y = 42;
            let z = &y; //~ ERROR: borrow may still be in use when `gen` block yields
            yield 43;
            panic!("{z}");
        };
        x.next();
        Box::new(x)
    };
    x.next();
}
