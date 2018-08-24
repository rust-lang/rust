// error-pattern: mismatched types

enum a { A(isize), }
enum b { B(isize), }

fn main() { let x: a = a::A(0); match x { b::B(y) => { } } }
