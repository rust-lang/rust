//@ run-pass
struct S<const X: u32>;

impl<const X: u32> S<X> {
    fn x() -> u32 {
        X
    }
}

fn main() {
    assert_eq!(S::<19>::x(), 19);
}
