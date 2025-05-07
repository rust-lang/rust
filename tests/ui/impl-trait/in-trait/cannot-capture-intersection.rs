use std::future::Future;
use std::pin::Pin;

trait MyTrait {
    fn foo<'a, 'b>(&'a self, x: &'b i32) -> impl Future<Output = i32>;
}

trait ErasedMyTrait {
    fn foo<'life0, 'life1, 'dynosaur>(&'life0 self, x: &'life1 i32)
    -> Pin<Box<dyn Future<Output = i32> + 'dynosaur>>
    where
        'life0: 'dynosaur,
        'life1: 'dynosaur;
}

struct DynMyTrait<T: ErasedMyTrait> {
    ptr: T,
}

impl<T: ErasedMyTrait> MyTrait for DynMyTrait<T> {
    fn foo<'a, 'b>(&'a self, x: &'b i32) -> impl Future<Output = i32> {
        self.ptr.foo(x)
        //~^ ERROR hidden type for `impl Future<Output = i32>` captures lifetime that does not appear in bounds
    }
}

fn main() {}
