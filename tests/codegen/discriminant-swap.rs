// The lowering of the function below initially reads and writes to the entire pointee, even
// though it only needs to do a store to the discriminant.
// This test ensures that std::hint::unreachable_unchecked does not prevent the desired
// optimization.

//@ compile-flags: -O

#![crate_type = "lib"]

use std::hint::unreachable_unchecked;
use std::ptr::{read, write};

type T = [u8; 753];

pub enum State {
    A(T),
    B(T),
}

// CHECK-LABEL: @init(ptr {{.*}}s)
// CHECK-NEXT: start
// CHECK-NEXT: store i8 1, ptr %s, align 1
// CHECK-NEXT: ret void
#[no_mangle]
unsafe fn init(s: *mut State) {
    let State::A(v) = read(s) else { unreachable_unchecked() };
    write(s, State::B(v));
}
