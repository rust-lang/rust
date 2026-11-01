// Regression test for issue #75062
// Tests that we don't ICE on a privacy error for a fieldless tuple struct.

mod foo {
    struct Bar();
}

fn main() {
    foo::Bar(); //~ ERROR tuple struct
}
