#![feature(type_alias_impl_trait)]

fn enum_upvar() {
    type T = impl Copy;
    let foo: T = Some((42, std::marker::PhantomData::<T>));
    let x = move || match foo {
        None => (),
        //~^ ERROR cannot resolve opaque type
    };
}

fn main() {}
