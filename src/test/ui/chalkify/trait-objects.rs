// check-pass
// compile-flags: -Z chalk

use std::fmt::Display;

fn main() {
    let d: &dyn Display = &mut 3;
    // FIXME(chalk) should be able to call d.to_string() as well, but doing so
    // requires Chalk to be able to prove trait object well-formed goals.
    (&d).to_string();
    let f: &dyn Fn(i32) -> _ = &|x| x + x;
    f(2);
}
