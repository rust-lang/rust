// edition:2018
// compile-flags:--extern alloc
// build-pass

// Test that `--extern alloc` will load from the sysroot without error.

fn main() {
    let _: Vec<i32> = alloc::vec::Vec::new();
}
