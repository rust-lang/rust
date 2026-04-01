//@ edition:2018

#![feature(ptr_metadata)]
#![feature(type_alias_impl_trait)]

type Opaque = impl std::fmt::Debug + ?Sized;

#[define_opaque(Opaque)]
fn opaque() -> &'static Opaque {
    &[1] as &[i32]
}

fn a<T: ?Sized>() {
    is_thin::<T>();
    //~^ ERROR type mismatch resolving `<T as Pointee>::Metadata == ()`

    is_thin::<Opaque>();
    //~^ ERROR type mismatch resolving `<Opaque as Pointee>::Metadata == ()`
}

fn is_thin<T: std::ptr::Pointee<Metadata = ()> + ?Sized>() {}

fn main() {}
