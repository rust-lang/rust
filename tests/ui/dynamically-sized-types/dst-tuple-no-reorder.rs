//@ run-pass

#![feature(unsized_tuple_coercion)]

// Ensure that unsizable fields that might be accessed don't get reordered

fn nonzero_size() {
    let sized: (u8, [u32; 2]) = (123, [456, 789]);
    let unsize: &(u8, [u32]) = &sized;
    assert_eq!(unsize.0, 123);
    assert_eq!(unsize.1.len(), 2);
    assert_eq!(unsize.1[0], 456);
    assert_eq!(unsize.1[1], 789);
}

fn zst() {
    let sized: (u8, [u32; 0]) = (123, []);
    let unsize: &(u8, [u32]) = &sized;
    assert_eq!(unsize.0, 123);
    assert_eq!(unsize.1.len(), 0);
}

pub fn main() {
    nonzero_size();
    zst();
}
