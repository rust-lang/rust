//@ build-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Foo {
    type Output;

    fn foo() -> Self::Output;
}

impl Foo for [u8; 3] {
    type Output = [u8; 1 + 2];

    fn foo() -> [u8; 3] {
        [1u8; 3]
    }
}

fn bug<const N: usize>()
where
    [u8; N]: Foo,
    <[u8; N] as Foo>::Output: AsRef<[u8]>,
{
    <[u8; N]>::foo().as_ref();
}

fn main() {
    bug::<3>();
}
