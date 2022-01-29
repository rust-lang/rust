// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(const_raw_ptr_deref)]
mod Y {
    pub type X = usize;
    extern "C" {
        pub static x: *const usize;
    }
    pub fn foo(value: *const X) -> *const X {
        value
    }
}

static foo: &Y::X = &*Y::foo(Y::x as *const Y::X);
//~^ ERROR dereference of raw pointer
//~| ERROR E0015
//~| ERROR use of extern static is unsafe and requires

fn main() {}
