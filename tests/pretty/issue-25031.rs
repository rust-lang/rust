//
// Testing that type items with where clauses output correctly.

//@ pp-exact

fn main() {
    type Foo<T> where T: Copy = Box<T>;
}
