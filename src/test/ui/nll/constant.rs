// Test that MIR borrowck and NLL analysis can handle constants of
// arbitrary types without ICEs.

// compile-flags:-Zborrowck=mir
// build-pass (FIXME(62277): could be check-pass?)

const HI: &str = "hi";

fn main() {
    assert_eq!(HI, "hi");
}
