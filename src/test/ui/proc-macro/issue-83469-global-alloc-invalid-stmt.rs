// Regression test for issue #83469
// Ensures that we recover from `#[global_alloc]` on an invalid
// stmt without an ICE

fn outer() {
    #[global_allocator]
    fn inner() {} //~ ERROR allocators must be statics
}

fn main() {}
