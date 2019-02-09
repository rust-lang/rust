// pp-exact

// Test that type items with where-clauses output correctly.

fn main() {
    type Foo<T> where T: Copy = Box<T>;
}
