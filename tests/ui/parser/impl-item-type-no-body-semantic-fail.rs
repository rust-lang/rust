fn main() {}

struct X;

impl X {
    type Y;
    //~^ ERROR associated type in `impl` without body
    //~| ERROR inherent associated types are unstable
    type Z: Ord;
    //~^ ERROR associated type in `impl` without body
    //~| ERROR bounds on `type`s in `impl`s have no effect
    //~| ERROR inherent associated types are unstable
    type W: Ord where Self: Eq;
    //~^ ERROR associated type in `impl` without body
    //~| ERROR bounds on `type`s in `impl`s have no effect
    //~| ERROR inherent associated types are unstable
    //~| ERROR `X: Eq` is not satisfied
    type W where Self: Eq;
    //~^ ERROR associated type in `impl` without body
    //~| ERROR inherent associated types are unstable
    //~| ERROR duplicate definitions
    //~| ERROR `X: Eq` is not satisfied
}
