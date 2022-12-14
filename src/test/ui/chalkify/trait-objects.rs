// check-pass
// compile-flags: -Z chalk

use std::fmt::Display;

fn main() {
    let d: &dyn Display = &mut 3;
    d.to_string();
    (&d).to_string();
    let f: &dyn Fn(i32) -> _ = &|x| x + x;
    f(2);
}
