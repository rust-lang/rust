// run-pass
#![feature(generators)]
#![feature(generator_trait)]
use std::cell::Cell;
use std::mem;
use std::ops::Generator;
use std::pin::Pin;

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

#[repr(transparent)]
struct NotCopy(u8);

#[repr(packed)]
struct Packed<'a>(NotCopy, Aligned<'a>);

fn main() {
    let drop_count = &Cell::new(0);
    {
        let mut p = Packed(NotCopy(0), Aligned { drop_count });
        p.1 = Aligned { drop_count };
        assert_eq!(drop_count.get(), 1);
    }
    assert_eq!(drop_count.get(), 2);

    let drop_count = &Cell::new(0);
    let mut g = || {
        let mut p = Packed(NotCopy(0), Aligned { drop_count });
        let _ = &p;
        p.1 = Aligned { drop_count };
        assert_eq!(drop_count.get(), 1);
        // Test that a generator drop function moves a value from a packed
        // struct to a separate local before dropping it. We move out the
        // first field to generate and open drop for the second field.
        drop(p.0);
        yield;
    };
    Pin::new(&mut g).resume(());
    assert_eq!(drop_count.get(), 1);
    drop(g);
    assert_eq!(drop_count.get(), 2);
}
