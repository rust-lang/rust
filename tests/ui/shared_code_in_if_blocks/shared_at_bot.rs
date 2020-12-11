#![allow(dead_code)]
#![deny(clippy::if_same_then_else, clippy::shared_code_in_if_blocks)]

// This tests the shared_code_in_if_blocks lint at the end of blocks

fn simple_examples() {
    // TODO xFrednet 2021-01-06: Test with const literals at the end
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
        println!("Block end!");
        result
    };
}

/// Simple examples where the move can cause some problems due to moved values
fn simple_but_suggestion_is_invalid() {
    // TODO xFrednet 2021-01-12: This
}

/// Tests where the blocks are not linted due to the used value scope
fn not_moveable_due_to_value_scope() {
    // TODO xFrednet 2021-01-12: This
}

fn main() {}
