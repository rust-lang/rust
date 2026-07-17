//@ edition: 2021
#![warn(clippy::block_scrutinee)]
#![allow(clippy::blocks_in_conditions)]
#![allow(clippy::let_and_return)]

fn my_function() -> Option<i32> {
    Some(1)
}

fn main() {
    if let Some(x) = { my_function() } {
        //~^ ERROR: this scrutinee is wrapped in a block
        let _ = x;
    }

    match { my_function() } {
        //~^ ERROR: this scrutinee is wrapped in a block
        Some(1) => println!("one"),
        Some(_) => println!("other"),
        None => println!("none"),
    }

    let mut v = vec![1, 2, 3];
    while let Some(x) = { v.pop() } {
        //~^ ERROR: this scrutinee is wrapped in a block
        let _ = x;
    }

    if let Some(x) = my_function() {
        let _ = x;
    }

    if let Some(x) = {
        let _y = 2;
        my_function()
    } {
        let _ = x;
    }

    //~v ERROR: this scrutinee is wrapped in a block
    if let Some(x) = {
        // We are popping a value
        v.pop()
    } {
        let _ = x;
    }

    macro_rules! get_val {
        () => {{ my_function() }};
    }
    if let Some(x) = get_val!() {
        let _ = x;
    }

    // Test that `unsafe` blocks are ignored
    unsafe fn my_unsafe_fn() -> Option<i32> {
        Some(1)
    }

    if let Some(x) = unsafe { my_unsafe_fn() } {
        let _ = x;
    }
}
