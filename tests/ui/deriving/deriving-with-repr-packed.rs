// run-pass
// check that derive on a packed struct does not call field
// methods with a misaligned field.

use std::mem;

#[derive(Copy, Clone)]
struct Aligned(usize);

#[inline(never)]
fn check_align(ptr: *const Aligned) {
    assert_eq!(ptr as usize % mem::align_of::<Aligned>(),
               0);
}

impl PartialEq for Aligned {
    fn eq(&self, other: &Self) -> bool {
        check_align(self);
        check_align(other);
        self.0 == other.0
    }
}

#[repr(packed)]
#[derive(Copy, Clone, PartialEq)]
struct Packed(Aligned, Aligned);

#[derive(PartialEq)]
#[repr(C)]
struct Dealigned<T>(u8, T);

fn main() {
    let d1 = Dealigned(0, Packed(Aligned(1), Aligned(2)));
    let ck = d1 == d1;
    assert!(ck);
}
