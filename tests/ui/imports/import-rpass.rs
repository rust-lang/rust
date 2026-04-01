//@ run-pass
mod foo {
    pub fn x(y: isize) { println!("{}", y); }
}

mod bar {
    use crate::foo::x;
    use crate::foo::x as z;
    pub fn thing() { x(10); z(10); }
}

pub fn main() { bar::thing(); }
