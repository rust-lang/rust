#![feature(c_variadic)]

unsafe extern "C" fn helper(_: i32, _: ...) {}

fn main() {
    unsafe {
        let f = helper as *const ();
        let f = std::mem::transmute::<_, unsafe extern "C" fn(...)>(f);

        f(1);
        //~^ ERROR: Undefined Behavior
    }
}
