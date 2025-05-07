//@ known-bug: #131048

impl<A> std::ops::CoerceUnsized<A> for A {}

fn main() {
    format_args!("Hello, world!");
}
