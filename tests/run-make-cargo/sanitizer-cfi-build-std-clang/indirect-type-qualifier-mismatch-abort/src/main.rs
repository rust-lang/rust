// This example demonstrates redirecting control flow using an indirect
// branch/call to a function with parameter type qualifiers than the argument
// type qualifiers intended/passed at the call/branch site.

use std::mem;

fn add_one(x: &i32) -> i32 {
    *x + 1
}

fn add_two(x: &mut i32) -> i32 {
    *x + 2
}

fn do_twice(f: fn(&i32) -> i32, arg: &i32) -> i32 {
    f(arg) + f(arg)
}

fn main() {
    let value: i32 = 5;
    let answer = do_twice(add_one, &value);

    println!("The answer is: {}", answer);

    println!("With CFI enabled, you should not see the next answer");
    let f: fn(&i32) -> i32 =
        unsafe { mem::transmute::<*const u8, fn(&i32) -> i32>(add_two as *const u8) };
    let next_answer = do_twice(f, &value);

    println!("The next answer is: {}", next_answer);
}
