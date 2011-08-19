

mod foo {
    fn x() -> int { ret 1; }
}

mod bar {
    fn y() -> int { ret 1; }
}

fn main() { foo::x(); bar::y(); }
