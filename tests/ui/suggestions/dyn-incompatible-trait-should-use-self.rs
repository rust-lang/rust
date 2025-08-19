trait A: Sized {
    fn f(a: dyn A) -> dyn A;
    //~^ ERROR the trait `A` is not dyn compatible
}
trait B {
    fn f(a: dyn B) -> dyn B;
    //~^ ERROR the trait `B` is not dyn compatible
}
trait C {
    fn f(&self, a: dyn C) -> dyn C;
}

fn main() {}
