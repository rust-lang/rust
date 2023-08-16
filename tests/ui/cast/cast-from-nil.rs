//@error-in-other-file: non-primitive cast: `()` as `u32`
fn main() { let u = (assert!(true) as u32); }
