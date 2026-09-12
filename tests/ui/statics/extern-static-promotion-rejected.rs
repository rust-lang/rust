// previously part of tests/mir-opt/const_promotion_extern_static.rs
// promotion of extern statics is now rejected entirely, even if we're not trying to read its value

unsafe extern "C" {
    static X: i32;
}

static mut FOO: *const &i32 = [unsafe { &X }].as_ptr();
//~^ ERROR dangling pointer

fn main() {}
