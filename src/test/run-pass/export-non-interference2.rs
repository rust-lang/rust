mod foo {
    pub mod bar {
        pub fn y() { super::super::foo::x(); }
    }

    pub fn x() { println!("x"); }
}

pub fn main() { self::foo::bar::y(); }
