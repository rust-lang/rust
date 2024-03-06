//@ build-fail

#![feature(large_assignments)]
#![move_size_limit = "1000"]
#![deny(large_assignments)]
#![allow(unused)]

// Note: This type does not implement Copy.
struct Data([u8; 9999]);

fn main() {
    // Looking at llvm-ir output, we can see a memcpy'd into Data, so we want
    // the lint to trigger here.
    let data = Data([100; 9999]); //~ ERROR large_assignments

    // Looking at llvm-ir output, we can see that there is no memcpy involved in
    // this function call. Instead, just a pointer is passed to the function. So
    // the lint shall not trigger here.
    take_data(data);
}

fn take_data(data: Data) {}
