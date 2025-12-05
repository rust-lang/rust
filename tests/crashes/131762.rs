//@ needs-rustc-debug-assertions
//@ known-bug: #131762
// ignore-tidy-linelength

#![feature(generic_assert)]
struct FloatWrapper(f64);

fn main() {
    assert!((0.0 / 0.0 >= 0.0) == (FloatWrapper(0.0 / 0.0) >= FloatWrapper(size_of::<u8>, size_of::<u16>, size_of::<usize> as fn() -> usize)))
}
