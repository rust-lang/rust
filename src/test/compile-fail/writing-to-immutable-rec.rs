// error-pattern: assigning to immutable field
fn main() { let r: {x: int} = {x: 1}; r.x = 6; }
