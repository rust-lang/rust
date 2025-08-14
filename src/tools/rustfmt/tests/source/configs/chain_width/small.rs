// rustfmt-chain_width: 40

struct Fluent {}

impl Fluent {
    fn blorp(&self) -> &Self {
        self
    }
}

fn main() {
    let test = Fluent {};

    // should not be wrapped
    test.blorp();
    test.blorp().blorp();
    test.blorp().blorp().blorp();
    test.blorp().blorp().blorp().blorp();

    // should be wrapped
    test.blorp().blorp().blorp().blorp().blorp();
    test.blorp().blorp().blorp().blorp().blorp().blorp();
}
