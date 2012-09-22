

mod foo {
    #[legacy_exports];
    fn x() -> int { return 1; }
}

mod bar {
    #[legacy_exports];
    fn y() -> int { return 1; }
}

fn main() { foo::x(); bar::y(); }
