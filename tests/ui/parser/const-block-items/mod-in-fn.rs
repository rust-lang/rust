//@ check-pass

#![feature(const_block_items)]

fn main() {
    mod foo {
        const { assert!(true) }
    }
}
