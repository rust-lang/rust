#![feature(const_block_items)]

const {
    #![expect(unused)] //~ ERROR: an inner attribute is not permitted in this context
    let a = 1;
    assert!(true);
}

fn main() {}
