fn main() {
    extern crate test; //~ ERROR use of unstable
    use test::*; //~ ERROR unresolved import
}
