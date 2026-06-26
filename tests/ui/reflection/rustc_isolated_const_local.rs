//@ build-pass

#![feature(const_trait_impl, rustc_attrs)]

#[rustc_isolated_const]
const VAL: usize = { 4 + std::mem::size_of::<u32>() };

// irrelevant local impls that should not fail the eval of VAL
const trait Foo {
    fn bar() -> usize {
        todo!()
    }
}

const impl Foo for () {}

fn main() {
    assert_eq!(VAL, 42);
}
