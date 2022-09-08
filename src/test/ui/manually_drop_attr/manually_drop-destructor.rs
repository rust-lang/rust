//! A test of `#[manually_drop]` on a type that *does* have a `Drop` impl.
//!
//! The mirror image of `manually_drop-nodestructor.rs`
#![feature(manually_drop_attr)]
// run-pass
extern crate core;
use core::cell::Cell;

struct DropCounter<'a>(&'a Cell<isize>);
impl<'a> Drop for DropCounter<'a> {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[manually_drop]
struct ManuallyDropped<'a> {
    field_1: DropCounter<'a>,
    field_2: DropCounter<'a>,
}

impl<'a> Drop for ManuallyDropped<'a> {
    fn drop(&mut self) {
        // just do a LITTLE dropping.
        unsafe {
            core::ptr::drop_in_place(&mut self.field_1)
        }
    }
}

#[manually_drop]
enum ManuallyDroppedEnum<'a> {
    A(DropCounter<'a>, DropCounter<'a>),
}

impl<'a> Drop for ManuallyDroppedEnum<'a> {
    fn drop(&mut self) {
        // just do a LITTLE dropping.
        let ManuallyDroppedEnum::A(a, _) = self;
        unsafe {
            core::ptr::drop_in_place(a);
        }
    }
}

/// Dropping a `#[manually_drop]` struct does not implicitly drop its fields.
///
/// (Though it does run `Drop`, which can choose to drop them explicitly.)
fn test_destruction() {
    let counter = Cell::new(0);
    core::mem::drop(ManuallyDropped {
        field_1: DropCounter(&counter),
        field_2: DropCounter(&counter),
    });
    // We only run the drop specifically requested in the Drop impl.
    assert_eq!(counter.get(), 1);
    assert!(core::mem::needs_drop::<ManuallyDropped>());

    core::mem::drop(ManuallyDroppedEnum::A(DropCounter(&counter), DropCounter(&counter)));
    assert_eq!(counter.get(), 2);
    assert!(core::mem::needs_drop::<ManuallyDroppedEnum>());

}

/// Assignment does still drop the fields.
fn test_assignment() {
    let counter = Cell::new(0);
    let mut manually_dropped = ManuallyDropped {
        field_1: DropCounter(&counter),
        field_2: DropCounter(&counter),
    };
    assert_eq!(counter.get(), 0);
    manually_dropped.field_1 = DropCounter(&counter);
    manually_dropped.field_2 = DropCounter(&counter);
    assert_eq!(counter.get(), 2);
}

fn main() {
    test_destruction();
    test_assignment();
}
