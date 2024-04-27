// -Zunstable-options added as test for ICE #97725 (left == right)`
// left: `Binder(<[u8; _] as std::default::Default>, [])`,
// right: `Binder(<[u8; 4] as std::default::Default>, [])

//@ compile-flags: -Zunstable-options
//@ check-pass
#![feature(generic_const_exprs)] //~ WARN the feature `generic_const_exprs` is incomplete


fn bind<const N: usize>(value: [u8; N + 2]) -> [u8; N * 2] {
    todo!()
}

fn main() {
    let mut arr = Default::default();
    arr = bind::<2>(arr);
}
