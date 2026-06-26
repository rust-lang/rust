// Regression test for https://github.com/rust-lang/rust/issues/148631

struct C;

struct S<T>(T);

trait Tr {}

impl<T> Clone for S<C>
//~^ ERROR the type parameter `T` is not constrained
where
    S<T>: Tr,
{
    fn clone(&self) -> Self {
        *self
        //~^ ERROR cannot move out of `*self`
    }
}

fn main() {}
