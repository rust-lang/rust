//@ compile-flags: -C panic=unwind -Copt-level=0
//@ needs-unwind
//@ only-wasm32

#![crate_type = "lib"]

// Test that `panic_in_cleanup` is called on webassembly targets when a panic
// occurs in a destructor during unwinding.

extern "Rust" {
    fn may_panic();
}

struct PanicOnDrop;

impl Drop for PanicOnDrop {
    fn drop(&mut self) {
        unsafe { may_panic() }
    }
}

// CHECK-LABEL: @double_panic
// CHECK: invoke void @may_panic()
// CHECK: invoke void @{{.+}}drop_in_place{{.+}}
// CHECK: unwind label %[[TERMINATE:.*]]
//
// CHECK: [[TERMINATE]]:
// CHECK: call void @{{.*panic_in_cleanup}}
// CHECK: unreachable
#[no_mangle]
pub fn double_panic() {
    let _guard = PanicOnDrop;
    unsafe { may_panic() }
}
