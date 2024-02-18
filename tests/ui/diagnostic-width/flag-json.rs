//@ compile-flags: --diagnostic-width=20 --error-format=json
//@ error-pattern:expected `()`, found integer

// This test checks that `-Z output-width` effects the JSON error output by restricting it to an
// arbitrarily low value so that the effect is visible.

fn main() {
    let _: () = 42;
}
