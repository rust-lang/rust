// error-pattern: mismatched types

enum blah { a(int, int, uint), b(int, int), }

fn main() { match a(1, 1, 2u) { a(_, x, y) | b(x, y) => { } } }
