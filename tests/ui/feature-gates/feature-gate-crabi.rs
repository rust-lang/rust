// Test that the `crabi` ABI is feature-gated, and cannot be used when the `crabi` feature gate is
// not used.

extern "crabi" {
//~^ ERROR crABI is experimental and subject to change [E0658]
//~| ERROR `"crabi"` is not a supported ABI for the current target [E0570]
    fn f();
}

fn main() {
    f();
}
