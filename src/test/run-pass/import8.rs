
import foo::x;
import z = foo::x;

mod foo {
    fn x(int y) { log y; }
}

fn main() { x(10); z(10); }