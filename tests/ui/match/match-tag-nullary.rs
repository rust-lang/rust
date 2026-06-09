enum A { A }
enum B { B }

fn main() { let x: A = A::A; match x { B::B => { } } } //~ ERROR mismatched types
