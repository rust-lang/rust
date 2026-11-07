// Traits with bounds mentioning `Self` are dyn-incompatible.

trait X {
    type U: PartialEq<Self>;
}

fn f() -> Box<dyn X<U = u32>> {
    //~^ ERROR the trait `X` is not dyn compatible
    loop {}
}

fn main() {}
