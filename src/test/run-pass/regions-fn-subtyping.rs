// Issue #2263.

// Should pass region checking.
fn ok(f: fn@(x: &uint)) {
    // Here, g is a function that can accept a uint pointer with
    // lifetime r, and f is a function that can accept a uint pointer
    // with any lifetime.  The assignment g = f should be OK (i.e., f
    // should be a subtype of g), because f can be used in any context
    // that expects g's type.  But this currently fails.
    let mut g: fn@(y: &r.uint) = fn@(x: &r.uint) { };
    g = f;
}

// This version is the same as above, except that here, g's type is
// inferred.
fn ok_inferred(f: fn@(x: &uint)) {
    let mut g = fn@(x: &r.uint) { };
    g = f;
}

fn main() {
}
