//@ known-bug: rust-lang/rust#129209

impl<
        const N: usize = {
            static || {
                Foo([0; X]);
            }
        },
    > PartialEq for True
{
}
