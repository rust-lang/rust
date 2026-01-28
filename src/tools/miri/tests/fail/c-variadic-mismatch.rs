#![feature(c_variadic)]

unsafe extern "C" fn helper(_: i32, _: ...) {}

fn main() {
    unsafe {
        let f = helper as *const ();
        let f = std::mem::transmute::<_, unsafe extern "C" fn(_: i32, _: i64)>(f);

        f(1i32, 1i64);
        //~^ ERROR: Undefined Behavior: calling a function where the caller and callee disagree on whether the function is C-variadic
    }
}
