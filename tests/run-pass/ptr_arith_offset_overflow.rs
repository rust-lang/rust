use std::ptr;

fn main() {
    let v = [1i16, 2];
    let x = &mut ptr::null(); // going through memory as there are more sanity checks along that path
    *x = v.as_ptr().wrapping_offset(1); // ptr to the 2nd element
    // Adding 2*isize::max and then 1 is like substracting 1
    *x = x.wrapping_offset(isize::MAX);
    *x = x.wrapping_offset(isize::MAX);
    *x = x.wrapping_offset(1);
    assert_eq!(unsafe { **x }, 1);
}
