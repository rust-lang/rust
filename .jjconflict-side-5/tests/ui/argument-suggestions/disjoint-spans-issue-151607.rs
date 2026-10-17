// Regression test for #151607
// The ICE was "all spans must be disjoint" when emitting diagnostics
// with overlapping suggestion spans.

struct B;
struct D;
struct F;
fn foo(g: F, y: F, e: &E) {
    //~^ ERROR cannot find type `E` in this scope
    foo(B, g, D, E, F, G)
    //~^ ERROR this function takes 3 arguments but 6 arguments were supplied
    //~| ERROR cannot find value `E` in this scope
    //~| ERROR cannot find value `G` in this scope
}

fn main() {}
