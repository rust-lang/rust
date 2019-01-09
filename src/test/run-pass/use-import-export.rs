// pretty-expanded FIXME #23616

mod foo {
    pub fn x() -> isize { return 1; }
}

mod bar {
    pub fn y() -> isize { return 1; }
}

pub fn main() { foo::x(); bar::y(); }
