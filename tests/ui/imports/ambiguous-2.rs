//@ aux-build: ../ambiguous-1.rs
// https://github.com/rust-lang/rust/pull/113099#issuecomment-1633574396

extern crate ambiguous_1;

fn main() {
    ambiguous_1::id(); //~ ERROR `id` is ambiguous
                       //~| WARN this was previously accepted
}
