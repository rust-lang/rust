// error-pattern: cast from nil: () as u32
fn main() { let u = (assert true) as u32; }