//@ run-pass
// Test that param substitutions from the correct environment are
// used when codegenning unboxed closure calls.


pub fn inside<F: Fn()>(c: F) {
    c();
}

// Use different number of type parameters and closure type to trigger
// an obvious ICE when param environments are mixed up
pub fn outside<A,B>() {
    inside(|| {});
}

fn main() {
    outside::<(),()>();
}
