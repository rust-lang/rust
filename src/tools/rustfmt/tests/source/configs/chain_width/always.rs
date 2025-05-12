// rustfmt-chain_width: 1
// setting an unachievable chain_width to always get chains
// on separate lines

struct Fluent {}

impl Fluent {
    fn blorp(&self) -> &Self {
        self
    }
}

fn main() {
    let test = Fluent {};

    // should be left alone
    test.blorp();

    // should be wrapped
    test.blorp().blorp();
    test.blorp().blorp().blorp();
    test.blorp().blorp().blorp().blorp();
}
