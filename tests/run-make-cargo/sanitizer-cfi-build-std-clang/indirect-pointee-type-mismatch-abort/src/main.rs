// This example demonstrates redirecting control flow using an indirect
// branch/call to a function with different return and parameter (i.e., pointee)
// types than the return type expected and arguments intended/passed at the
// call/branch site.

use std::mem;

fn add_one(x: *const i32) -> i32 {
    unsafe { *x + 1 }
}

fn add_two(x: *const i64) -> i32 {
    unsafe { (*x + 2) as i32 }
}

fn do_twice(f: fn(*const i32) -> i32, arg: *const i32) -> i32 {
    f(arg) + f(arg)
}

fn main() {
    let value: i32 = 5;
    let answer = do_twice(add_one, &value);

    println!("The answer is: {}", answer);

    println!("With CFI enabled, you should not see the next answer");
    let f: fn(*const i32) -> i32 =
        unsafe { mem::transmute::<*const u8, fn(*const i32) -> i32>(add_two as *const u8) };
    let next_answer = do_twice(f, &value);

    println!("The next answer is: {}", next_answer);
}
