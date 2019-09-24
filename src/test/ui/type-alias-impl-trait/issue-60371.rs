trait Bug {
    type Item: Bug;

    const FUN: fn() -> Self::Item;
}

impl Bug for &() {
    type Item = impl Bug; //~ ERROR `impl Trait` in type aliases is unstable
    //~^ ERROR the trait bound `(): Bug` is not satisfied
    //~^^ ERROR could not find defining uses

    const FUN: fn() -> Self::Item = || ();
}

fn main() {}
