#![deny(clippy::branches_sharing_code, clippy::if_same_then_else)]
#![allow(dead_code)]
#![allow(
    clippy::mixed_read_write_in_expression,
    clippy::uninlined_format_args,
    clippy::needless_else
)]

// This tests valid if blocks that shouldn't trigger the lint

// Tests with value references are includes in "shared_code_at_bottom.rs"

fn valid_examples() {
    let x = 2;

    // The edge statements are different
    if x == 9 {
        let y = 1 << 5;

        println!("This is the same: vvv");
        let _z = y;
        println!("The block expression is different");

        println!("Different end 1");
    } else {
        let y = 1 << 7;

        println!("This is the same: vvv");
        let _z = y;
        println!("The block expression is different");

        println!("Different end 2");
    }

    // No else
    if x == 2 {
        println!("Hello world!");
        println!("Hello back, how are you?");

        // This is different vvvv
        println!("Howdy stranger =^.^=");

        println!("Bye Bye World");
    } else if x == 9 {
        println!("Hello world!");
        println!("Hello back, how are you?");

        // This is different vvvv
        println!("Hello reviewer :D");

        println!("Bye Bye World");
    }

    // Overlapping statements only in else if blocks -> Don't lint
    if x == 0 {
        println!("I'm important!")
    } else if x == 17 {
        println!("I share code in else if");

        println!("x is 17");
    } else {
        println!("I share code in else if");

        println!("x is nether x nor 17");
    }

    // Mutability is different
    if x == 13 {
        let mut y = 9;
        println!("Value y is: {}", y);
        y += 16;
        let _z1 = y;
    } else {
        let y = 9;
        println!("Value y is: {}", y);
        let _z2 = y;
    }

    // Same blocks but at start and bottom so no `if_same_then_else` lint
    if x == 418 {
        let y = 9;
        let z = 8;
        let _ = (x, y, z);
        // Don't tell the programmer, my code is also in the else block
    } else if x == 419 {
        println!("+-----------+");
        println!("|           |");
        println!("|  O     O  |");
        println!("|     °     |");
        println!("|  \\_____/  |");
        println!("|           |");
        println!("+-----------+");
    } else {
        let y = 9;
        let z = 8;
        let _ = (x, y, z);
        // I'm so much better than the x == 418 block. Trust me
    }

    let x = 1;
    if true {
        println!("{}", x);
    } else {
        let x = 2;
        println!("{}", x);
    }

    // Let's test empty blocks
    if false {
    } else {
    }
}

/// This makes sure that the `if_same_then_else` masks the `shared_code_in_if_blocks` lint
fn trigger_other_lint() {
    let x = 0;
    let y = 1;

    // Same block
    if x == 0 {
        let u = 19;
        println!("How are u today?");
        let _ = "This is a string";
    } else {
        let u = 19;
        println!("How are u today?");
        let _ = "This is a string";
    }

    // Only same expression
    let _ = if x == 6 { 7 } else { 7 };

    // Same in else if block
    let _ = if x == 67 {
        println!("Well I'm the most important block");
        "I'm a pretty string"
    } else if x == 68 {
        println!("I'm a doppelgänger");
        // Don't listen to my clone below

        if y == 90 { "=^.^=" } else { ":D" }
    } else {
        // Don't listen to my clone above
        println!("I'm a doppelgänger");

        if y == 90 { "=^.^=" } else { ":D" }
    };

    if x == 0 {
        println!("I'm single");
    } else if x == 68 {
        println!("I'm a doppelgänger");
        // Don't listen to my clone below
    } else {
        // Don't listen to my clone above
        println!("I'm a doppelgänger");
    }
}

fn main() {}
