// error-pattern: mismatched types

enum a { A(int), }
enum b { B(int), }

fn main() { let x: a = A(0); match x { B(y) => { } } }

