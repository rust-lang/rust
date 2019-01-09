// error-pattern: non-primitive cast: `()` as `u32`
fn main() { let u = (assert!(true) as u32); }
