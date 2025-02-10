//@ compile-flags: -Copt-level=3
//@ min-llvm-version: 19

#![crate_type = "lib"]

pub enum State {
    A([u8; 753]),
    B([u8; 753]),
}

// CHECK-LABEL: @update
#[no_mangle]
pub unsafe fn update(s: *mut State) {
    // CHECK-NEXT: start:
    // CHECK-NEXT: store i8
    // CHECK-NEXT: ret
    let State::A(v) = s.read() else { std::hint::unreachable_unchecked() };
    s.write(State::B(v));
}
