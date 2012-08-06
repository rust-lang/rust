// error-pattern: mismatched types

enum a { A, }
enum b { B, }

fn main() { let x: a = A; match x { B => { } } }

