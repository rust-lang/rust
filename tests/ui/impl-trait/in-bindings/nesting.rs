//@ check-pass

#![feature(impl_trait_in_bindings)]

fn main() {
    let _: impl IntoIterator<Item = impl Sized> = ["hello", " world"];
}
