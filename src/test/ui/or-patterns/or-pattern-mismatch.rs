enum Blah { A(isize, isize, usize), B(isize, isize) }

fn main() { match Blah::A(1, 1, 2) { Blah::A(_, x, y) | Blah::B(x, y) => { } } }
//~^ ERROR mismatched types
