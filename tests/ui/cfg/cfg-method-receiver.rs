macro_rules! cbor_map {
    ($key:expr) => {
        $key.signum();
        //~^ ERROR can't call method `signum` on ambiguous numeric type `{integer}` [E0689]
    };
}

fn main() {
    cbor_map! { #[cfg(test)] 4};
    //~^ ERROR removing an expression is not supported in this position
}
