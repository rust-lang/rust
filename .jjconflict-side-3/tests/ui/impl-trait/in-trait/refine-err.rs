#![deny(refining_impl_trait)]

trait FromRow {
    fn prepare(self) -> impl Fn() -> T;
    //~^ ERROR cannot find type `T` in this scope
}

impl<T> FromRow for T {
    fn prepare(self) -> impl Fn() -> T {
        || todo!()
    }
}

fn main() {}
