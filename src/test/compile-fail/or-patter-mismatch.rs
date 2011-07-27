// error-pattern: mismatched types

tag blah { a(int, int, uint); b(int, int); }

fn main() { alt a(1, 1, 2u) { a(_, x, y) | b(x, y) { } } }