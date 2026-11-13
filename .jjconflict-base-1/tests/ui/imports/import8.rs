//@ run-pass

use foo::x;
use foo::x as z;

mod foo {
    pub fn x(y: isize) { println!("{}", y); }
}

pub fn main() { x(10); z(10); }
