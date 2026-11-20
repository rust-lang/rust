//@ compile-flags: -Copt-level=3
//@ needs-unwind
#![feature(c_unwind)]
#![crate_type = "lib"]

pub struct Foo([u8; 1000]);

extern "C" {
    fn init(p: *mut Foo);
}

pub fn new_from_uninit() -> Foo {
    // CHECK-LABEL: new_from_uninit
    // CHECK-NOT: call void @llvm.memcpy.
    let mut x = std::mem::MaybeUninit::uninit();
    unsafe {
        init(x.as_mut_ptr());
        x.assume_init()
    }
}

extern "C-unwind" {
    fn init_unwind(p: *mut Foo);
}

pub fn new_from_uninit_unwind() -> Foo {
    // CHECK-LABEL: new_from_uninit_unwind
    // CHECK-NOT: call void @llvm.memcpy.
    let mut x = std::mem::MaybeUninit::uninit();
    unsafe {
        init_unwind(x.as_mut_ptr());
        x.assume_init()
    }
}
