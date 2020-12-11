#![allow(dead_code, clippy::eval_order_dependence)]
#![deny(clippy::if_same_then_else, clippy::shared_code_in_if_blocks)]

// This tests the shared_code_in_if_blocks lint at the start of blocks

fn simple_examples() {
    let x = 0;

    // Simple
    if true {
        println!("Hello World!");
        println!("I'm branch nr: 1");
    } else {
        println!("Hello World!");
        println!("I'm branch nr: 2");
    }

    // Else if
    if x == 0 {
        let y = 9;
        println!("The value y was set to: `{}`", y);
        let _z = y;

        println!("I'm the true start index of arrays");
    } else if x == 1 {
        let y = 9;
        println!("The value y was set to: `{}`", y);
        let _z = y;

        println!("I start counting from 1 so my array starts from `1`");
    } else {
        let y = 9;
        println!("The value y was set to: `{}`", y);
        let _z = y;

        println!("Ha, Pascal allows you to start the array where you want")
    }

    // Return a value
    let _ = if x == 7 {
        let y = 16;
        println!("What can I say except: \"you're welcome?\"");
        let _ = y;
        x
    } else {
        let y = 16;
        println!("Thank you");
        y
    };
}

/// Simple examples where the move can cause some problems due to moved values
fn simple_but_suggestion_is_invalid() {
    let x = 10;

    // Can't be automatically moved because used_value_name is getting used again
    let used_value_name = 19;
    if x == 10 {
        let used_value_name = "Different type";
        println!("Str: {}", used_value_name);
        let _ = 1;
    } else {
        let used_value_name = "Different type";
        println!("Str: {}", used_value_name);
        let _ = 2;
    }
    let _ = used_value_name;

    // This can be automatically moved as `can_be_overridden` is not used again
    let can_be_overridden = 8;
    let _ = can_be_overridden;
    if x == 11 {
        let can_be_overridden = "Move me";
        println!("I'm also moveable");
        let _ = 111;
    } else {
        let can_be_overridden = "Move me";
        println!("I'm also moveable");
        let _ = 222;
    }
}

fn main() {}
