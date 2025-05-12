#![feature(no_sanitize)]

#[no_sanitize(brontosaurus)] //~ ERROR invalid argument
fn main() {
}
