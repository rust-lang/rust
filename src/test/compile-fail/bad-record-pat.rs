// error-pattern:expected a record with 2 fields, found one with 1

fn main() { alt {x: 1, y: 2} { {x: x} { } } }