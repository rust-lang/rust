// xfail-fast
// aux-build:packed.rs

extern mod packed;

fn main() {
    assert_eq!(sys::size_of::<packed::S>(), 5);
}
