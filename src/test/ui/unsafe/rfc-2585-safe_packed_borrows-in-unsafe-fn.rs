#![feature(unsafe_block_in_unsafe_fn)]

#[repr(packed)]
pub struct Packed {
    data: &'static u32,
}

const PACKED: Packed = Packed { data: &0 };

#[allow(safe_packed_borrows)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn allow_allow() {
    &PACKED.data; // allowed
}

#[allow(safe_packed_borrows)]
#[warn(unsafe_op_in_unsafe_fn)]
unsafe fn allow_warn() {
    &PACKED.data; // allowed
}

#[allow(safe_packed_borrows)]
#[deny(unsafe_op_in_unsafe_fn)]
unsafe fn allow_deny() {
    &PACKED.data; // allowed
}

#[warn(safe_packed_borrows)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn warn_allow() {
    &PACKED.data; // allowed
}

#[warn(safe_packed_borrows)]
#[warn(unsafe_op_in_unsafe_fn)]
unsafe fn warn_warn() {
    &PACKED.data; //~ WARN
    //~| WARNING this was previously accepted by the compiler but is being phased out
}

#[warn(safe_packed_borrows)]
#[deny(unsafe_op_in_unsafe_fn)]
unsafe fn warn_deny() {
    &PACKED.data; //~ WARN
    //~| WARNING this was previously accepted by the compiler but is being phased out
}

#[deny(safe_packed_borrows)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn deny_allow() {
    &PACKED.data; // allowed
}

#[deny(safe_packed_borrows)]
#[warn(unsafe_op_in_unsafe_fn)]
unsafe fn deny_warn() {
    &PACKED.data; //~ WARN
}

#[deny(safe_packed_borrows)]
#[deny(unsafe_op_in_unsafe_fn)]
unsafe fn deny_deny() {
    &PACKED.data; //~ ERROR
    //~| WARNING this was previously accepted by the compiler but is being phased out
}

fn main() {}
