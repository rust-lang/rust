// error-pattern:assigning to immutable vec content
fn main() { let v: [int] = [1, 2, 3]; v[1] = 4; }
