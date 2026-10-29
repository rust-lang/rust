//! Regression test for https://github.com/rust-lang/rust/issues/159680.
//@ compile-flags: -Znext-solver=globally

trait Crash {
    fn build<'a>(&mut self, commands: impl Iterator + 'a) -> impl Iterator + 'a {
        //~^ ERROR type mismatch resolving `impl Iterator == impl Iterator`

        let further_commands = self;
        self.build(further_commands)
        //~^ ERROR `Self` is not an iterator
    }
}

fn main() {}
