// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

mod y {
    pub type X = usize;
    extern "C" {
        pub static x: *const usize;
    }
    pub fn foo(value: *const X) -> *const X {
        value
    }
}

static FOO: &y::X = &*y::foo(y::x as *const y::X);
//~^ ERROR dereference of raw pointer
//~| ERROR E0015
//~| ERROR use of extern static is unsafe and requires

fn main() {}
