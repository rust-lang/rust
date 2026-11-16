// regression test for #164376

#[derive(Clone)]
struct S<'a, T>(&'a T);

fn require<T>() where for<'a> S<'a, T>: Clone {}

fn main() {
    require::<&'static mut u8>();
    //~^ ERROR the trait bound `&'static mut u8: Clone` is not satisfied
}
