const fn bad_const_fn_deref_raw(x: *mut usize) -> &'static usize { unsafe { &*x } }
//~^ dereferencing raw pointers in constant functions

const unsafe fn bad_const_unsafe_deref_raw(x: *mut usize) -> usize { *x }
//~^ dereferencing raw pointers in constant functions

const unsafe fn bad_const_unsafe_deref_raw_ref(x: *mut usize) -> &'static usize { &*x }
//~^ dereferencing raw pointers in constant functions

fn main() {}

const unsafe fn no_union() {
    union Foo { x: (), y: () }
    Foo { x: () }.y
    //~^ unions in const fn
}
