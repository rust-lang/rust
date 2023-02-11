// run-pass

use std::fmt::Display;

fn foo(f: impl Display + Clone) -> String {
    let g = f.clone();
    format!("{} + {}", f, g)
}

fn main() {
    let sum = foo(format!("22"));
    assert_eq!(sum, r"22 + 22");
}
