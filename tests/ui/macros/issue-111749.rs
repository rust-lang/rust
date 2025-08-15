macro_rules! cbor_map {
    ($key:expr) => {
        $key.signum();
    };
}

fn main() {
    cbor_map! { #[test(test)] 4};
    //~^ ERROR removing an expression is not supported in this position
    //~| ERROR attribute must be of the form `#[test]`
    //~| WARNING this was previously accepted by the compiler but is being phased out
}
