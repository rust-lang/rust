//@ edition: 2024

// Regression test for issue https://github.com/rust-lang/rust/issues/147208
// Fix by https://github.com/rust-lang/rust/pull/149681

use bar::foo;
    //~^ ERROR unresolved import `bar`
use foo::bar;
fn main() {
    mod bar;
    //~^ ERROR cannot declare a file module inside a block unless it has a path attribute
    use bar::foo;
}
