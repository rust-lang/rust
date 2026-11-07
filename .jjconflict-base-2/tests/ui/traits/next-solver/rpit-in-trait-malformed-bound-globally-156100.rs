//@ compile-flags: -Znext-solver=globally

// Regression test for <https://github.com/rust-lang/rust/issues/156100>.

trait X {
    fn into_iter(&self) -> impl Iterator<X> {
        //~^ ERROR trait takes 0 generic arguments but 1 generic argument was supplied
        //~| ERROR trait takes 0 generic arguments but 1 generic argument was supplied
        //~| ERROR trait takes 0 generic arguments but 1 generic argument was supplied
        //~| ERROR trait takes 0 generic arguments but 1 generic argument was supplied
        //~| ERROR overflow evaluating the requirement
        todo!()
    }
}

fn main() {}
