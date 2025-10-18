macro_rules! cbor_map {
    ($key:expr) => {
        $key.signum();
    };
}

fn main() {
    cbor_map! { #[test(test)] 4i32};
    //~^ ERROR the `#[test]` attribute may only be used on a free function
    //~| ERROR attribute must be of the form `#[test]`
    //~| WARNING this was previously accepted by the compiler but is being phased out
}
