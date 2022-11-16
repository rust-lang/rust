//! A test of `#[manually_drop]` on a type that *doesn't* have a `Drop` impl.
//!
//! The mirror image of `manually_drop-destructor.rs`

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

#[manually_drop]
enum ManuallyDroppedEnum<'a> {
    _A,
    B(DropCounter<'a>, DropCounter<'a>),
}

/// Dropping a `#[manually_drop]` type does not implicitly drop its fields.
fn test_destruction() {
    let counter = Cell::new(0);
    core::mem::drop(ManuallyDropped {
        field_1: DropCounter(&counter),
        field_2: DropCounter(&counter),
    });
    assert_eq!(counter.get(), 0);
    assert!(!core::mem::needs_drop::<ManuallyDropped>());

    core::mem::drop(ManuallyDroppedEnum::B(DropCounter(&counter), DropCounter(&counter)));
    assert_eq!(counter.get(), 0);
    assert!(!core::mem::needs_drop::<ManuallyDroppedEnum>());
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
