// Check that we report an error if an upcast box is moved twice.

fn consume(_: Box<[i32]>) {
}

fn foo(b: Box<[i32;5]>) {
    consume(b);
    consume(b); //~ ERROR use of moved value
}

fn main() {
}
