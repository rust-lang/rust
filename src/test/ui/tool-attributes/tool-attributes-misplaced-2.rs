#[derive(rustfmt::skip)] //~ ERROR macro `rustfmt::skip` may not be used for derive attributes
struct S;

fn main() {
    rustfmt::skip!(); //~ ERROR `rustfmt::skip` can only be used in attributes
}
