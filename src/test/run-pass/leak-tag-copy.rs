

enum t { a, b(@int), }

fn main() { let mut x = b(@10); x = a; }
