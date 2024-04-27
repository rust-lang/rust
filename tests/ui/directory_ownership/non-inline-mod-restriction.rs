// Test that non-inline modules are not allowed inside blocks.

fn main() {
    mod foo; //~ ERROR cannot declare a non-inline module inside a block
}
