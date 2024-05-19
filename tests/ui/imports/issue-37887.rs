fn main() {
    extern crate libc; //~ ERROR use of unstable
    use libc::*; //~ ERROR unresolved import
}
