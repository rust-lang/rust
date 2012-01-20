// error-pattern: mismatched types

enum a { A, }
enum b { B, }

fn main() { let x: a = A; alt x { B { } } }

