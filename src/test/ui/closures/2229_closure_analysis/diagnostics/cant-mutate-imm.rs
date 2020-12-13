#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete

// Ensure that diagnostics for mutability error (because the root variable
// isn't mutable) work with `capture_disjoint_fields` enabled.

fn main() {
    let x = (10, 10);
    let y = (x, 10);
    let z = (y, 10);

    let mut c = || {
        z.0.0.0 = 20;
        //~^ ERROR: cannot assign to `z`, as it is not declared as mutable
    };

    c();
}
