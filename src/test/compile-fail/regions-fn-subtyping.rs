// Here, `f` is a function that takes a pointer `x` and a function
// `g`, where `g` requires its argument `y` to be in the same region
// that `x` is in.
fn has_same_region(f: fn(x: &a/int, g: fn(y: &a/int))) {
    // Somewhat counterintuitively, this fails because, in
    // `wants_two_regions`, the `g` argument needs to be able to
    // accept any region.  That is, the type that `has_same_region`
    // expects is *not* a subtype of the type that `wants_two_regions`
    // expects.
    wants_two_regions(f); //~ ERROR mismatched types
}

fn wants_two_regions(_f: fn(x: &int, g: fn(y: &int))) {
    // Suppose we were to write code here that passed some arbitrary
    // &int and some arbitrary fn(&int) to whatever's passed in as _f.
    // This would be fine as far as the type annotation on the formal
    // parameter _f goes, but if _f were `f` we'd be in trouble since
    // `f` can't handle those arguments.
}

fn main() {
}


