// Test for #79487
//@ check-pass

#![allow(dead_code)]

mod sha2 {
    extern "C" {
        pub(super) fn GFp_sha512_block_data_order();
    }
}

fn main() {}
