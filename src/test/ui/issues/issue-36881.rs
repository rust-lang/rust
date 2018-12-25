// aux-build:issue-36881-aux.rs

fn main() {
    #[allow(unused_extern_crates)]
    extern crate issue_36881_aux;
    use issue_36881_aux::Foo; //~ ERROR unresolved import
}
