#![allow(safe_extern_statics)]

mod Y {
    pub type X = usize;
    extern {
        pub static x: *const usize;
    }
    pub fn foo(value: *const X) -> *const X {
        value
    }
}

static foo: *const Y::X = Y::foo(Y::x as *const Y::X);
//~^ ERROR `*const usize` cannot be shared between threads safely [E0277]
//~| ERROR E0015

fn main() {}
