// When mutably indexing a type that implements `Index` and `IndexMut` but
// `Index::index` is being used specifically, the normal special help message
// should not mention a missing `IndexMut` impl.

fn main() {
    use std::ops::Index;

    let v = String::from("dinosaur");
    Index::index(&v, 1..2).make_ascii_uppercase(); //~ ERROR
}
