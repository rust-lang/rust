//@ compile-flags: -C no-prepopulate-passes -C opt-level=0

#![crate_type = "lib"]

// String formating macros without any arguments should compile
// to a `memcpy` followed by a call to `std::io::stdio::_print`.

#[no_mangle]
pub fn code() {
    // CHECK-LABEL: @code
    // CHECK-NOT: getelementptr
    // CHECK-NOT: store
    // CHECK-NOT: ; call core::fmt::Arguments::new_const
    // CHECK: call void @llvm.memcpy
    // CHECK-NEXT: ; call std::io::stdio::_print
    println!("hello world");
}
