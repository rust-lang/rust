//@ compile-flags: -Copt-level=3
//@ revisions: new old
//@ [old] max-llvm-major-version: 21
//@ [new] min-llvm-version: 22

#![crate_type = "lib"]

// The bug here was that it was loading and storing the whole value.
// It's ok for it to load the discriminant,
// to preserve the UB from `unreachable_unchecked`,
// but it better only store the constant discriminant of `B`.

pub enum State {
    A([u8; 753]),
    B([u8; 753]),
}

// CHECK-LABEL: @update
#[no_mangle]
pub unsafe fn update(s: *mut State) {
    // CHECK-NOT: alloca

    // CHECK-NOT: load
    // CHECK-NOT: store
    // CHECK-NOT: memcpy
    // CHECK-NOT: 75{{3|4}}

    // old: %[[TAG:.+]] = load i8, ptr %s, align 1
    // old-NEXT: trunc nuw i8 %[[TAG]] to i1

    // CHECK-NOT: load
    // CHECK-NOT: store
    // CHECK-NOT: memcpy
    // CHECK-NOT: 75{{3|4}}

    // CHECK: store i8 1, ptr %s, align 1

    // CHECK-NOT: load
    // CHECK-NOT: store
    // CHECK-NOT: memcpy
    // CHECK-NOT: 75{{3|4}}

    // CHECK: ret
    let State::A(v) = s.read() else { std::hint::unreachable_unchecked() };
    s.write(State::B(v));
}
