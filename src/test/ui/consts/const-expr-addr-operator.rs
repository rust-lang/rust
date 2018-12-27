// Encountered while testing #44614.
// compile-pass

pub fn main() {
    // Constant of generic type (int)
    const X: &'static u32 = &22;
    assert_eq!(0, match &22 {
        X => 0,
        _ => 1,
    });
}
