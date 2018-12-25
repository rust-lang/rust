pub mod foo {
    pub fn x() { ::bar::x(); }
}

pub mod bar {
    pub fn x() { println!("x"); }
}

pub fn main() { foo::x(); }
