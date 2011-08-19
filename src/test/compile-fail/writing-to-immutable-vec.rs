// error-pattern:assignment to immutable vec content
fn main() { let v: [int] = [1, 2, 3]; v[1] = 4; }
