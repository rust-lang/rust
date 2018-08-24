#[derive(rustfmt::skip)] //~ ERROR expected a macro, found tool attribute
struct S;

fn main() {
    rustfmt::skip!(); //~ ERROR expected a macro, found tool attribute
}
