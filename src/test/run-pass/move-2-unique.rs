

fn main() { let x = ~{x: 1, y: 2, z: 3}; let y = move x; assert (y.y == 2); }
