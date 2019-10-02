// Check that we apply unsizing coercions based on the return type.
//
// Also serves as a regression test for #60424.
//
// edition:2018
// check-pass

#![allow(warnings)]

use std::fmt::Debug;

const TMP: u32 = 22;

// Coerce from `Box<"asdf">` to `Box<dyn Debug>`.
fn raw_pointer_coercion() {
    fn sync_example() -> *const u32 {
        &TMP
    }

    async fn async_example() -> *const u32 {
        &TMP
    }
}

fn main() {}
