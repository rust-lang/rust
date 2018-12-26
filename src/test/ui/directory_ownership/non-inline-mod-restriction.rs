// Test that non-inline modules are not allowed inside blocks.

fn main() {
    mod foo; //~ ERROR Cannot declare a non-inline module inside a block
}
