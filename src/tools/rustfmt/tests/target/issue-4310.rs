#![feature(const_generics)]

fn foo<
    const N: [u8; {
        struct Inner<'a>(&'a ());
        3
    }],
>() {
}
