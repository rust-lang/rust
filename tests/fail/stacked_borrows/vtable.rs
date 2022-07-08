//@error-pattern: vtable pointer does not have permission
#![feature(ptr_metadata)]

trait Foo {}

impl Foo for u32 {}

fn uwu(thin: *const (), meta: &'static ()) -> *const dyn Foo {
    core::ptr::from_raw_parts(thin, unsafe { core::mem::transmute(meta) })
}

fn main() {
    unsafe {
        let orig = 1_u32;
        let x = &orig as &dyn Foo;
        let (ptr, meta) = (x as *const dyn Foo).to_raw_parts();
        let _ = uwu(ptr, core::mem::transmute(meta));
    }
}
