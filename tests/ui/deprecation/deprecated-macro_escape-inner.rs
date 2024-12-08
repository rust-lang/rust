//@ run-pass

mod foo {
    #![macro_escape] //~ WARN `#[macro_escape]` is a deprecated synonym for `#[macro_use]`
}

fn main() {
}
