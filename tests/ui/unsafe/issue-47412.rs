#[derive(Copy, Clone)]
enum Void {}

// Tests that we detect unsafe places (specifically, union fields and
// raw pointer dereferences), even when they're matched on while having
// an uninhabited type (equivalent to `std::intrinsics::unreachable()`).

fn union_field() {
    union Union { unit: (), void: Void }
    let u = Union { unit: () };
    match u.void {}
    //~^ ERROR access to union field is unsafe
}

fn raw_ptr_deref() {
    let ptr = std::ptr::null::<Void>();
    match *ptr {}
    //~^ ERROR dereference of raw pointer is unsafe
}

fn main() {}
