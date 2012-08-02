

mod foo {
    fn x() -> int { return 1; }
}

mod bar {
    fn y() -> int { return 1; }
}

fn main() { foo::x(); bar::y(); }
