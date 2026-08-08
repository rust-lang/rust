//@ known-bug: rust-lang/rust#147208
//@ edition: 2021
use bar::foo;
use foo::bar;
fn main() {
    mod bar;
    use bar::foo;
}
