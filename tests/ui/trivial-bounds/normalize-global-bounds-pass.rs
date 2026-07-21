// Normalize aliases before checking trivial bounds
// Regression test for #154145 and #140309

//@ check-pass

#![allow(dead_code)]

trait Trait {
    type Assoc;
}

impl<T> Trait for T {
    type Assoc = T;
}

trait IceBaby {}
impl<const N: usize> IceBaby for [u8; N] {}

trait Hello {}
impl<const N: usize> Hello for [(); N] {}

trait Foo {
    type Assoc;
}

impl Foo for u32 {
    type Assoc = u8;
}

fn foo<T>()
where
    (): Trait<Assoc = T>,
    <() as Trait>::Assoc: Sized,
{
}

fn bar<const N: usize>()
where
    (): Trait<Assoc = [u8; N]>,
    <() as Trait>::Assoc: IceBaby,
{
}

fn baz<const N: usize>()
where
    <u32 as Foo>::Assoc: Hello,
    u32: Foo<Assoc = [(); N]>,
{
}

fn main() {}
