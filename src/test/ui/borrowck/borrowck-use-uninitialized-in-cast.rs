// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

// Check that we detect unused values that are cast to other things.
// The problem was specified to casting to `*`, as creating unsafe
// pointers was not being fully checked. Issue #20791.

fn main() {
    let x: &i32;
    let y = x as *const i32; //[ast]~ ERROR use of possibly uninitialized variable: `*x` [E0381]
                             //[mir]~^ ERROR [E0381]
}
