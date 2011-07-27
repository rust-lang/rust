// error-pattern: mismatched types

tag a { A; }
tag b { B; }

fn main() { let x: a = A; alt x { B. { } } }

