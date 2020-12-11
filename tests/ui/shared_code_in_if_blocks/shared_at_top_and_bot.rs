#![allow(dead_code)]
#![deny(clippy::if_same_then_else, clippy::shared_code_in_if_blocks)]

// shared_code_in_if_blocks at the top and bottom of the if blocks

fn main() {
    // TODO xFrednet 2021-01-12: This
}

// General TODOs By xFrednet:

// 
// * Make a test with overlapping eq regions (else ifs)
// * Test if as function parameter, tuple constructor, index, while loop condition
// * Test where only the expression is the same
// * Test where the block only has an expression
// * Test with let on a previous line let _ = \n if...
// * Tests with unreadable formatting (Inline if, Not indented)
// * Test multiline condition if x == 9 \n x == 8 {}
// * Test if for return/break (Only move to front)
// * Test in inline closures
// * Test with structs and tuples
