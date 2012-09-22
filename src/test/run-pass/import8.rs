
use foo::x;
use z = foo::x;

mod foo {
    #[legacy_exports];
    fn x(y: int) { log(debug, y); }
}

fn main() { x(10); z(10); }
