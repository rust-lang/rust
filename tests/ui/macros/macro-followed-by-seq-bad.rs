// Regression test for issue #25436: check that things which can be
// followed by any token also permit X* to come afterwards.

#![allow(unused_macros)]

macro_rules! foo {
  ( $a:expr $($b:tt)* ) => { }; //~ ERROR not allowed for `expr` fragments
  ( $a:ty $($b:tt)* ) => { };   //~ ERROR not allowed for `ty` fragments
}

fn main() { }
