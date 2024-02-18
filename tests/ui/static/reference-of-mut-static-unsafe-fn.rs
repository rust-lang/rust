//@ compile-flags: --edition 2024 -Z unstable-options

fn main() {}

unsafe fn _foo() {
    static mut X: i32 = 1;
    static mut Y: i32 = 1;

    let _y = &X;
    //~^ ERROR creating a shared reference to a mutable static [E0796]

    let ref _a = X;
    //~^ ERROR creating a shared reference to a mutable static [E0796]

    let ref mut _a = X;
    //~^ ERROR creating a mutable reference to a mutable static [E0796]

    let (_b, _c) = (&X, &mut Y);
    //~^ ERROR creating a shared reference to a mutable static [E0796]
    //~^^ ERROR creating a mutable reference to a mutable static [E0796]

    foo(&X);
    //~^ ERROR creating a shared reference to a mutable static [E0796]
}

fn foo<'a>(_x: &'a i32) {}
