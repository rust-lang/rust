// Regression test for issue #89497.

//@ run-rustfix

fn main() {
    let pointer: usize = &1_i32 as *const i32 as usize;
    let _reference: &'static i32 = unsafe { pointer as *const i32 as &'static i32 };
    //~^ ERROR: non-primitive cast
    //~| HELP: consider borrowing the value
}
