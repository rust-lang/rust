//! regression test for <https://github.com/rust-lang/rust/issues/16725>
//@ aux-build:private-extern-fn.rs

extern crate private_extern_fn as foo;

fn main() {
    unsafe {
        foo::bar();
        //~^ ERROR: function `bar` is private
    }
}
