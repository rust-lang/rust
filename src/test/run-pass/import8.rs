
import foo::x;
import z = foo::x;

mod foo {
    fn x(y: int) { log y; }
}

fn main() { x(10); z(10); }
