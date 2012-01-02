// error-pattern:binary operation + cannot be applied to type `*int`

fn die() -> *int { (0 as *int) + (0 as *int) }
fn main() { }