// run-pass
use std::cell::Cell;
use std::mem;

struct Aligned<'a> {
    drop_count: &'a Cell<usize>
}

#[inline(never)]
fn check_align(ptr: *const Aligned) {
    assert_eq!(ptr as usize % mem::align_of::<Aligned>(),
               0);
}

impl<'a> Drop for Aligned<'a> {
    fn drop(&mut self) {
        check_align(self);
        self.drop_count.set(self.drop_count.get() + 1);
    }
}

#[repr(packed)]
struct Packed<'a>(u8, Aligned<'a>);

fn main() {
    let drop_count = &Cell::new(0);
    {
        let mut p = Packed(0, Aligned { drop_count });
        p.1 = Aligned { drop_count };
        assert_eq!(drop_count.get(), 1);
    }
    assert_eq!(drop_count.get(), 2);
}
