//@ edition: 2024

// Regression test for issue https://github.com/rust-lang/rust/issues/147208
// Fix by https://github.com/rust-lang/rust/pull/149681

use foo::bar::E::*;
  //~^ ERROR cannot find module or crate `foo` in this scope
use foo::bar::test_use::io as std_io;
  //~^ ERROR cannot find module or crate `foo` in this scope
  //~| ERROR unresolved import `foo::bar::test_use::io`
fn main() {
    Foo(());
    //~^ ERROR cannot find function, tuple struct or tuple variant `Foo` in this scope
    {
        use ::std::io as std_io;
        use std_io::stdout;
    }
}
