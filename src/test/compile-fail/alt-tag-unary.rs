// error-pattern: mismatched types

tag a { A(int); }
tag b { B(int); }

fn main() { let x: a = A(0); alt x { B(y) { } } }

