// xfail-test

// Issue #2263.

// Here, `f` is a function that takes a pointer `x` and a function
// `g`, where `g` requires its argument `y` to be in the same region
// that `x` is in.
fn has_same_region(f: fn(x: &a.int, g: fn(y: &a.int))) {
    // `f` should be the type that `wants_same_region` wants, but
    // right now the compiler complains that it isn't.
    wants_same_region(f);
}

fn wants_same_region(_f: fn(x: &b.int, g: fn(y: &b.int))) { 
}

fn main() {
}


