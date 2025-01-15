#![feature(impl_trait_in_assoc_type)]

trait Foo: Sized {
    type Bar;
    type Gat<T: Foo>;
    fn foo(self) -> (<Self as Foo>::Gat<u32>, <Self as Foo>::Gat<Self>);
}

impl Foo for u32 {
    type Bar = ();
    type Gat<T: Foo> = ();
    fn foo(self) -> (<Self as Foo>::Gat<u32>, <Self as Foo>::Gat<Self>) {
        ((), ())
    }
}

impl Foo for () {
    type Bar = impl Sized;
    //~^ ERROR: unconstrained opaque type
    type Gat<T: Foo> = <T as Foo>::Bar;
    // Because we encounter `Gat<u32>` first, we never walk into another `Gat`
    // again, thus missing the opaque type that we could be defining.
    fn foo(self) -> (<Self as Foo>::Gat<u32>, <Self as Foo>::Gat<Self>) {
        ((), ())
        //~^ ERROR: mismatched types
    }
}

fn main() {}
