trait Bug {
    type Item: Bug;

    const FUN: fn() -> Self::Item;
}

impl Bug for &() {
    type Item = impl Bug; //~ ERROR `impl Trait` in associated types is unstable

    const FUN: fn() -> Self::Item = || ();
    //~^ ERROR the trait bound `(): Bug` is not satisfied
}

fn main() {}
