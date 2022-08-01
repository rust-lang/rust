// run-rustfix

#![allow(dead_code)]

struct S;
struct Y;

trait Trait {}

impl Trait for S {}
impl Trait for Y {}

fn baz() -> impl Trait {
    if true {
        S
    } else {
        Y //~ ERROR `if` and `else` have incompatible types
    }
}

fn main() {}
