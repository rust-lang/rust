//@ compile-flags: --crate-type=lib

//@ pp-exact
fn f() {} /*
          The next line should not be indented.

          That one. It shouldn't have been indented.
          */
