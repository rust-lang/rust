// check-pass
#![feature(generic_const_exprs)] //~ WARN the feature `generic_const_exprs` is incomplete


fn bind<const N: usize>(value: [u8; N + 2]) -> [u8; N * 2] {
    todo!()
}

fn main() {
    let mut arr = Default::default();
    arr = bind::<2>(arr);
}
