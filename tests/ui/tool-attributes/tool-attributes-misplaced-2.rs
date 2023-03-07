#[derive(rustfmt::skip)] //~ ERROR expected derive macro, found tool attribute `rustfmt::skip`
struct S;

fn main() {
    rustfmt::skip!(); //~ ERROR expected macro, found tool attribute `rustfmt::skip`
}
