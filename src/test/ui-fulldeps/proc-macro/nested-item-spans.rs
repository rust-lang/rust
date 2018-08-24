// aux-build:nested-item-spans.rs

extern crate nested_item_spans;

use nested_item_spans::foo;

#[foo]
fn another() {
    fn bar() {
        let x: u32 = "x"; //~ ERROR: mismatched types
    }

    bar();
}

fn main() {
    #[foo]
    fn bar() {
        let x: u32 = "x"; //~ ERROR: mismatched types
    }

    bar();
    another();
}
