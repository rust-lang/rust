//@ compile-flags: -Zmir-opt-level=4
//@ run-pass
fn main() {
    fn foo<const N: usize>() -> [u8; N] {
        [0; N]
    }
    let _x = foo::<1>();
}
