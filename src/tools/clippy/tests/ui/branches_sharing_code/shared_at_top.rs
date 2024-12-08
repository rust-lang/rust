#![deny(clippy::branches_sharing_code, clippy::if_same_then_else)]
#![allow(dead_code)]
#![allow(clippy::mixed_read_write_in_expression, clippy::uninlined_format_args)]
//@no-rustfix
// This tests the branches_sharing_code lint at the start of blocks

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
    //~^^^^^^^ ERROR: all if blocks contain the same code at the start

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
    //~^^^^^^^^^^^^^^^^^^^ ERROR: all if blocks contain the same code at the start

    // Return a value
    let _ = if x == 7 {
        //~^ ERROR: all if blocks contain the same code at the start
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
    //~^^^^^^^^^ ERROR: all if blocks contain the same code at the start
    let _ = used_value_name;

    // This can be automatically moved as `can_be_overridden` is not used again
    let can_be_overridden = 8;
    let _ = can_be_overridden;
    if x == 11 {
        //~^ ERROR: all if blocks contain the same code at the start
        let can_be_overridden = "Move me";
        println!("I'm also moveable");
        let _ = 111;
    } else {
        let can_be_overridden = "Move me";
        println!("I'm also moveable");
        let _ = 222;
    }
}

/// This function tests that the `IS_SAME_THAN_ELSE` only covers the lint if it's enabled.
fn check_if_same_than_else_mask() {
    let x = 2021;

    #[allow(clippy::if_same_then_else)]
    if x == 2020 {
        //~^ ERROR: all if blocks contain the same code at the start
        println!("This should trigger the `SHARED_CODE_IN_IF_BLOCKS` lint.");
        println!("Because `IF_SAME_THEN_ELSE` is allowed here");
    } else {
        println!("This should trigger the `SHARED_CODE_IN_IF_BLOCKS` lint.");
        println!("Because `IF_SAME_THEN_ELSE` is allowed here");
    }

    if x == 2019 {
        println!("This should trigger `IS_SAME_THAN_ELSE` as usual");
    } else {
        println!("This should trigger `IS_SAME_THAN_ELSE` as usual");
    }
    //~^^^^^ ERROR: this `if` has identical blocks
}

#[allow(clippy::vec_init_then_push)]
fn pf_local_with_inferred_type_issue7053() {
    if true {
        let mut v = Vec::new();
        v.push(0);
    } else {
        let mut v = Vec::new();
        v.push("");
    };
}

fn main() {}
