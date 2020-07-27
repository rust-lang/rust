// compile-flags: -Z terminal-width=20 --error-format=json

// This test checks that `-Z terminal-width` effects the JSON error output by restricting it to an
// arbitrarily low value so that the effect is visible.

fn main() {
    let _: () = 42;
    //~^ ERROR mismatched types
}
