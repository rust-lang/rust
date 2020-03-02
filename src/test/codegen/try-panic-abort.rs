// compile-flags: -C panic=abort -O

#![crate_type = "lib"]
#![feature(unwind_attributes, core_intrinsics)]

extern "C" {
    #[unwind(allow)]
    fn bar(data: *mut u8);
}
extern "Rust" {
    fn catch(data: *mut u8, exception: *mut u8);
}

// CHECK-LABEL: @foo
#[no_mangle]
pub unsafe fn foo() -> i32 {
    // CHECK: call void @bar
    // CHECK: ret i32 0
    std::intrinsics::r#try(|x| bar(x), 0 as *mut u8, |x, y| catch(x, y))
}
