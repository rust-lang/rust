// test for #110696
// failed to resolve instance for <Scope<()> as MyIndex<()>>::my_index
// ignore-tidy-linelength

#![feature(type_alias_impl_trait)]

use std::marker::PhantomData;

trait MyIndex<T> {
    type O;
    fn my_index(self) -> Self::O;
}
trait MyFrom<T>: Sized {
    type Error;
    fn my_from(value: T) -> Result<Self, Self::Error>;
}

trait F {}
impl F for () {}
type DummyT<T> = impl F;
#[define_opaque(DummyT)]
fn _dummy_t<T>() -> DummyT<T> {}

struct Phantom1<T>(PhantomData<T>);
struct Phantom2<T>(PhantomData<T>);
struct Scope<T>(Phantom2<DummyT<T>>);

impl<T> Scope<T> {
    #[define_opaque(DummyT)]
    fn new() -> Self {
        //~^ ERROR item does not constrain
        unimplemented!()
    }
}

impl<T> MyFrom<Phantom2<T>> for Phantom1<T> {
    type Error = ();
    fn my_from(_: Phantom2<T>) -> Result<Self, Self::Error> {
        unimplemented!()
    }
}

impl<T: MyFrom<Phantom2<DummyT<U>>>, U> MyIndex<DummyT<T>> for Scope<U> {
    //~^ ERROR the type parameter `T` is not constrained by the impl
    type O = T;
    #[define_opaque(DummyT)]
    fn my_index(self) -> Self::O {
        MyFrom::my_from(self.0).ok().unwrap()
    }
}

fn main() {
    let _pos: Phantom1<DummyT<()>> = Scope::new().my_index();
}
