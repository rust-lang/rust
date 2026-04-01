#![feature(const_block_items)]

// ATTENTION: if we ever start accepting inner attributes here, make sure `rustfmt` can handle them.
//            see: https://github.com/rust-lang/rustfmt/issues/6158

const {
    #![expect(unused)] //~ ERROR: an inner attribute is not permitted in this context
    let a = 1;
    assert!(true);
}

fn main() {}
