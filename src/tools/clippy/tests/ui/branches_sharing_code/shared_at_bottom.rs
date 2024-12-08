#![deny(clippy::if_same_then_else, clippy::branches_sharing_code)]
#![allow(
    clippy::equatable_if_let,
    clippy::uninlined_format_args,
    clippy::redundant_pattern_matching,
    dead_code
)]
//@no-rustfix
// This tests the branches_sharing_code lint at the end of blocks

fn simple_examples() {
    let x = 1;

    let _ = if x == 7 {
        println!("Branch I");
        let start_value = 0;
        println!("=^.^=");

        // Same but not moveable due to `start_value`
        let _ = start_value;

        // The rest is self contained and moveable => Only lint the rest
        let result = false;
        println!("Block end!");
        result
    } else {
        println!("Branch II");
        let start_value = 8;
        println!("xD");

        // Same but not moveable due to `start_value`
        let _ = start_value;

        // The rest is self contained and moveable => Only lint the rest
        let result = false;
        //~^ ERROR: all if blocks contain the same code at the end
        //~| NOTE: the end suggestion probably needs some adjustments to use the expressio
        println!("Block end!");
        result
    };

    // Else if block
    if x == 9 {
        println!("The index is: 6");

        println!("Same end of block");
    } else if x == 8 {
        println!("The index is: 4");

        // We should only get a lint trigger for the last statement
        println!("This is also eq with the else block");
        println!("Same end of block");
    } else {
        println!("This is also eq with the else block");
        println!("Same end of block");
        //~^ ERROR: all if blocks contain the same code at the end
    }

    // Use of outer scope value
    let outer_scope_value = "I'm outside the if block";
    if x < 99 {
        let z = "How are you";
        println!("I'm a local because I use the value `z`: `{}`", z);

        println!(
            "I'm moveable because I know: `outer_scope_value`: '{}'",
            outer_scope_value
        );
    } else {
        let z = 45678000;
        println!("I'm a local because I use the value `z`: `{}`", z);

        println!(
            //~^ ERROR: all if blocks contain the same code at the end
            "I'm moveable because I know: `outer_scope_value`: '{}'",
            outer_scope_value
        );
    }

    if x == 9 {
        if x == 8 {
            // No parent!!
            println!("---");
            println!("Hello World");
        } else {
            println!("Hello World");
            //~^ ERROR: all if blocks contain the same code at the end
        }
    }
}

/// Simple examples where the move can cause some problems due to moved values
fn simple_but_suggestion_is_invalid() {
    let x = 16;

    // Local value
    let later_used_value = 17;
    if x == 9 {
        let _ = 9;
        let later_used_value = "A string value";
        println!("{}", later_used_value);
    } else {
        let later_used_value = "A string value";
        //~^ ERROR: all if blocks contain the same code at the end
        println!("{}", later_used_value);
        // I'm expecting a note about this
    }
    println!("{}", later_used_value);

    // outer function
    if x == 78 {
        let simple_examples = "I now identify as a &str :)";
        println!("This is the new simple_example: {}", simple_examples);
    } else {
        println!("Separator print statement");

        let simple_examples = "I now identify as a &str :)";
        //~^ ERROR: all if blocks contain the same code at the end
        println!("This is the new simple_example: {}", simple_examples);
    }
    simple_examples();
}

/// Tests where the blocks are not linted due to the used value scope
fn not_moveable_due_to_value_scope() {
    let x = 18;

    // Using a local value in the moved code
    if x == 9 {
        let y = 18;
        println!("y is: `{}`", y);
    } else {
        let y = "A string";
        println!("y is: `{}`", y);
    }

    // Using a local value in the expression
    let _ = if x == 0 {
        let mut result = x + 1;

        println!("1. Doing some calculations");
        println!("2. Some more calculations");
        println!("3. Setting result");

        result
    } else {
        let mut result = x - 1;

        println!("1. Doing some calculations");
        println!("2. Some more calculations");
        println!("3. Setting result");

        result
    };

    let _ = if x == 7 {
        let z1 = 100;
        println!("z1: {}", z1);

        let z2 = z1;
        println!("z2: {}", z2);

        z2
    } else {
        let z1 = 300;
        println!("z1: {}", z1);

        let z2 = z1;
        println!("z2: {}", z2);

        z2
    };
}

/// This should add a note to the lint msg since the moved expression is not `()`
fn added_note_for_expression_use() -> u32 {
    let x = 9;

    let _ = if x == 7 {
        x << 2
    } else {
        let _ = 6;
        x << 2
        //~^ ERROR: all if blocks contain the same code at the end
        //~| NOTE: the end suggestion probably needs some adjustments to use the expressio
    };

    if x == 9 {
        x * 4
    } else {
        let _ = 17;
        x * 4
        //~^ ERROR: all if blocks contain the same code at the end
        //~| NOTE: the end suggestion probably needs some adjustments to use the expressio
    }
}

#[rustfmt::skip]
fn test_suggestion_with_weird_formatting() {
    let x = 9;
    let mut a = 0;
    let mut b = 0;

    // The error message still looks weird tbh but this is the best I can do
    // for weird formatting
    if x == 17 { b = 1; a = 0x99; } else { a = 0x99; }
    //~^ ERROR: all if blocks contain the same code at the end
}

fn fp_test() {
    let x = 17;

    if x == 18 {
        let y = 19;
        if y < x {
            println!("Trigger")
        }
    } else {
        let z = 166;
        if z < x {
            println!("Trigger")
        }
    }
}

fn fp_if_let_issue7054() {
    // This shouldn't trigger the lint
    let string;
    let _x = if let true = true {
        ""
    } else if true {
        string = "x".to_owned();
        &string
    } else {
        string = "y".to_owned();
        &string
    };
}

fn main() {}
