// The purpose of this test is not to validate the output of the compiler.
// Instead, it ensures the suggestion is generated without performing an arithmetic overflow.

fn main() {
    let x = not_found; //~ ERROR cannot find value `not_found` in this scope
    simd_gt::<()>(x);
    //~^ ERROR this associated function takes 0 generic arguments but 1 generic argument was supplied
    //~| ERROR cannot find function `simd_gt` in this scope
}
