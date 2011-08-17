// error-pattern:duplicate definition of f

import m1::{f};
import m2::{f};

mod m1 {
    fn f() {}
}

mod m2 {
    fn f() {}
}

fn main() {}