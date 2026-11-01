//@ check-pass
//@ compile-flags: -Zextra-const-ub-checks

#[repr(packed)]
pub struct Foo {
    bar: u8,
    baa: [u32; 1],
}

const FOOMP: Foo = Foo {
    bar: 0,
    baa: [69; 1],
};

fn main() {
    let _val = FOOMP;
}
