// known-bug: #108491

// FIXME(inherent_associated_types): This should pass.

struct Foo {
    bar: Self::Bar,
}
impl Foo {
    pub type Bar = usize;
}
