mod foo {
    #[legacy_exports];
    fn x(y: int) { log(debug, y); }
}

mod bar {
    #[legacy_exports];
    use foo::x;
    use z = foo::x;
    fn thing() { x(10); z(10); }
}

fn main() { bar::thing(); }
