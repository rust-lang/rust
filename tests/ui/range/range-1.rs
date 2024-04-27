// Test range syntax - type errors.

pub fn main() {
    // Mixed types.
    let _ = 0u32..10i32;
    //~^ ERROR mismatched types

    // Bool => does not implement iterator.
    for i in false..true {}
    //~^ ERROR `bool: Step` is not satisfied

    // Unsized type.
    let arr: &[_] = &[1, 2, 3];
    let range = *arr..;
    //~^ ERROR the size for values of type
}
