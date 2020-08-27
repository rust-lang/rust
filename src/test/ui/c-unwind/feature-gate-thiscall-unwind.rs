// Test that the "thiscall-unwind" ABI is feature-gated, and cannot be used when
// the `c_unwind` feature gate is not used.

extern "thiscall-unwind" fn f() {}
//~^ ERROR thiscall-unwind ABI is experimental and subject to change [E0658]

fn main() {
    f();
}
