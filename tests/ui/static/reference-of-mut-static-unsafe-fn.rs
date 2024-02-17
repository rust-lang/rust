//@ compile-flags: --edition 2024 -Z unstable-options

fn main() {}

unsafe fn _foo() {
    static mut X: i32 = 1;
    static mut Y: i32 = 1;

    let _y = &X;
    //~^ ERROR reference of mutable static is disallowed

    let ref _a = X;
    //~^ ERROR reference of mutable static is disallowed

    let (_b, _c) = (&X, &Y);
    //~^ ERROR reference of mutable static is disallowed
    //~^^ ERROR reference of mutable static is disallowed

    foo(&X);
    //~^ ERROR reference of mutable static is disallowed
}

fn foo<'a>(_x: &'a i32) {}
