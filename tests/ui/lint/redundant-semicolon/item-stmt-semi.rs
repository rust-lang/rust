#![deny(redundant_semicolons)]

fn main() {
    fn inner() {}; //~ ERROR unnecessary
    struct Bar {}; //~ ERROR unnecessary
}
