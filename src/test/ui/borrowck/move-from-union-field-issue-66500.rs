// Moving from a reference/raw pointer should be an error, even when they're
// the field of a union.

union Pointers {
    a: &'static String,
    b: &'static mut String,
    c: *const String,
    d: *mut String,
}

unsafe fn move_ref(u: Pointers) -> String {
    *u.a
    //~^ ERROR cannot move out of `*u.a`
}
unsafe fn move_ref_mut(u: Pointers) -> String {
    *u.b
    //~^ ERROR cannot move out of `*u.b`
}
unsafe fn move_ptr(u: Pointers) -> String {
    *u.c
    //~^ ERROR cannot move out of `*u.c`
}
unsafe fn move_ptr_mut(u: Pointers) -> String {
    *u.d
    //~^ ERROR cannot move out of `*u.d`
}

fn main() {}
