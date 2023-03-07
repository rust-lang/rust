// Test that we can parse a unit struct with a where clause, even if
// it leads to an error later on since `T` is unused.

struct Foo<T> where T: Copy; //~ ERROR parameter `T` is never used

fn main() {}
