//@ revisions: ascii unicode
//@[ascii] compile-flags: --diagnostic-width=20
//@[unicode] compile-flags: -Zunstable-options --error-format=human-unicode --diagnostic-width=20

// This test checks that `-Z output-width` effects the human error output by restricting it to an
// arbitrarily low value so that the effect is visible.

fn main() {
    let _: () = 42;
    //[ascii]~^ ERROR mismatched types
}
