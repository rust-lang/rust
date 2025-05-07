trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

trait Foo {}

// Even though using fulfillment in coherence allows us to figure out that
// `?T = ()`, we still treat it as incoherent because `(): Iterator` may be
// added upstream.
impl<T> Foo for T where (): Mirror<Assoc = T> {}
//~^ NOTE first implementation here
impl<T> Foo for T where T: Iterator {}
//~^ ERROR conflicting implementations of trait `Foo` for type `()`
//~| NOTE conflicting implementation for `()`
//~| NOTE upstream crates may add a new impl of trait `std::iter::Iterator` for type `()` in future versions

fn main() {}
