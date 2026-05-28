// Regression test for https://github.com/rust-lang/rust/issues/157015.

fn takes_raw_ptr(_: *const u32) {}

fn main() {
    let x = 0u32;
    takes_raw_ptr(&raw x);
    //~^ ERROR expected one of
}
