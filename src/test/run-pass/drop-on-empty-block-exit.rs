

tag t { foo(@int); }

fn main() { auto tt = foo(@10); alt (tt) { case (foo(?z)) { } } }