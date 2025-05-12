pub struct P;
pub struct Q;
pub struct R<T>(T);

// Checks that tuple and unit both work
pub fn side_effect() {}

// Check a non-tuple
pub fn not_tuple() -> P {
    loop {}
}

// Check a 1-tuple
pub fn one() -> (P,) {
    loop {}
}

// Check a 2-tuple
pub fn two() -> (P, P) {
    loop {}
}

// Check a nested tuple
pub fn nest() -> (Q, R<(u32,)>) {
    loop {}
}
