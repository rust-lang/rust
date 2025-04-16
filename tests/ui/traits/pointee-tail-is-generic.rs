//@ check-pass
//@ edition:2018

#![feature(ptr_metadata)]
#![feature(type_alias_impl_trait)]

pub type Opaque = impl std::future::Future;

#[define_opaque(Opaque)]
fn opaque() -> Opaque {
    async {}
}

fn a<T>() {
    // type parameter T is known to be sized
    is_thin::<T>();
    // tail of ADT (which is a type param) is known to be sized
    is_thin::<std::cell::Cell<T>>();
    // opaque type is known to be sized
    is_thin::<Opaque>();
}

fn a2<T: Iterator>() {
    // associated type is known to be sized
    is_thin::<T::Item>();
}

fn is_thin<T: std::ptr::Pointee<Metadata = ()>>() {}

fn main() {}
