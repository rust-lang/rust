// Test that file modules are not allowed inside blocks.

fn main() {
    mod foo; //~ ERROR cannot declare a file module inside a block unless it has a path attribute
}
