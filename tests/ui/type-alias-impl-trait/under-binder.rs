#![feature(type_alias_impl_trait)]

type Opaque<'a> = impl Sized + 'a;

fn test(f: fn(u8)) -> fn(Opaque<'_>) {
    f //~ ERROR E0792
}

fn main() {}
