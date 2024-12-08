//@ aux-build:issue-36881-aux.rs

fn main() {
    extern crate issue_36881_aux;
    use issue_36881_aux::Foo; //~ ERROR unresolved import
}
