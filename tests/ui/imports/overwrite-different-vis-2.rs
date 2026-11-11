// Regression test for issue #152347.

//@ check-pass
//@ edition: 2018..

use outer::*; // must be before `mod outer`
mod outer {
    mod inner {
        pub fn f() {}
    }

    use inner::*;
    pub use inner::*;
}

fn main() {
    f();
}
