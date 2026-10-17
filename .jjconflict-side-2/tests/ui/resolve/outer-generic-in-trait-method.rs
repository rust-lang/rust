//! Regression test for <https://github.com/rust-lang/rust/issues/3021>.
//! Test we can't use outer generic param in trait method defined in fn body.

fn siphash<T>() {

    trait U {
        fn g(&self, x: T) -> T;  //~ ERROR can't use generic parameters from outer item
        //~^ ERROR can't use generic parameters from outer item
    }
}

fn main() {}
