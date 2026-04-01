use std::mem::{align_of, size_of};

// See <https://github.com/rust-lang/rust/issues/134713>

#[repr(C)]
struct Foo(usize, u8);

fn main() {
    let buf1: [usize; 2] = [1000, 2000];
    let buf2: [usize; 2] = [3000, 4000];

    // Foo and [usize; 2] have the same size and alignment,
    // so swap_nonoverlapping should treat them the same
    assert_eq!(size_of::<Foo>(), size_of::<[usize; 2]>());
    assert_eq!(align_of::<Foo>(), align_of::<[usize; 2]>());

    let mut b1 = buf1;
    let mut b2 = buf2;
    // Safety: b1 and b2 are distinct local variables,
    // with the same size and alignment as Foo.
    unsafe {
        std::ptr::swap_nonoverlapping(
            b1.as_mut_ptr().cast::<Foo>(),
            b2.as_mut_ptr().cast::<Foo>(),
            1,
        );
    }
    assert_eq!(b1, buf2);
    assert_eq!(b2, buf1);
}
