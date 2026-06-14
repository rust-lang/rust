// Regression test for #143174.

#![crate_type = "lib"]

type Fun = unsafe extern "C" fn();

struct Foo(Fun);

static FOO: &Foo = &Foo(BAR);
//~^ ERROR cannot access extern static `BAR` [E0080]
//~| ERROR use of extern static is unsafe and requires unsafe function or block [E0133]

// Formerly part of `tests/mir-opt/const_promotion_extern_static.rs`.
// Promotion of extern statics is now rejected entirely.
static mut BAZ: *const &Fun = [unsafe { &BAR }].as_ptr();
//~^ ERROR dangling pointer

unsafe extern "C" {
    static BAR: Fun;
}
