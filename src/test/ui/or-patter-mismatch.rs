// error-pattern: mismatched types

enum blah { a(isize, isize, usize), b(isize, isize), }

fn main() { match blah::a(1, 1, 2) { blah::a(_, x, y) | blah::b(x, y) => { } } }
