#![no_std]

#[unsafe(no_mangle)]
pub fn panic_noarg() {
    // CHECK-LABEL: @panic_noarg(
    // CHECK-NEXT: start:
    // CHECK-NEXT: tail call void @llvm.trap()
    panic!();
}

#[unsafe(no_mangle)]
pub fn panic_str() {
    // CHECK-LABEL: @panic_str(
    // CHECK-NEXT: start:
    // CHECK-NEXT: tail call void @llvm.trap()
    panic!("ouch");
}

#[unsafe(no_mangle)]
pub fn bounds_check(x: &[u8], idx: usize) -> &u8 {
    // CHECK-LABEL: @bounds_check(
    // CHECK-NEXT: start:
    // CHECK-NEXT: icmp ult
    // CHECK-NEXT: br i1
    // CHECK: bb1:
    // CHECK-NEXT: getelementptr inbounds nuw i8
    // CHECK-NEXT: ret ptr
    // CHECK: panic:
    // CHECK-NEXT: tail call void @llvm.trap()
    &x[idx]
}

#[unsafe(no_mangle)]
pub fn str_bounds_check(x: &str, idx: usize) -> &str {
    // CHECK-LABEL: @str_bounds_check(
    // CHECK-NOT: call
    // CHECK: tail call void @llvm.trap()
    // CHECK-NOT: call
    &x[idx..]
}

#[unsafe(no_mangle)]
pub fn unsigned_integer_div(x: u16, y: u16) -> u16 {
    // CHECK-LABEL: @unsigned_integer_div(
    // CHECK-NEXT: start:
    // CHECK-NEXT: icmp eq i16
    // CHECK-NEXT: br i1
    // CHECK: bb1:
    // CHECK-NEXT: udiv i16
    // CHECK-NEXT: ret i16
    // CHECK: panic:
    // CHECK-NEXT: tail call void @llvm.trap()
    x / y
}

#[unsafe(no_mangle)]
pub fn refcell_already_borrowed() {
    // CHECK-LABEL: @refcell_already_borrowed(
    // CHECK-NOT: call
    // CHECK: tail call void @llvm.trap()
    // CHECK-NOT: call
    let r = core::cell::RefCell::new(0u8);
    let _guard = r.borrow_mut();
    r.borrow_mut();
}
