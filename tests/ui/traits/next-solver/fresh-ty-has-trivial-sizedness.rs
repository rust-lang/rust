// Fixes #151709
//@ compile-flags: -Znext-solver=globally

trait A {
    type Ty;
    const CT: Self::Ty;
}

fn main() {
    let _: &dyn A<Ty = i32, CT = 0> = &();
    //~^ ERROR associated const equality is incomplete
    //~| ERROR the trait `A` is not dyn compatible
    //~| ERROR the size for values of type `FreshTy(0)` cannot be known
    //~| ERROR the trait bound `FreshTy(0): A` is not satisfied
}
