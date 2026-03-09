//! regression test for https://github.com/rust-lang/rust/issues/17373
fn main() {
    *return //~ ERROR type `!` cannot be dereferenced
    ;
}
