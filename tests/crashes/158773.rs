//@ known-bug: #158773
//@ needs-rustc-debug-assertions
//@ compile-flags: -Znext-solver=globally
trait HasLifetime {
    type AtLifetime<'a>;
}

pub struct ExistentialLifetime<S: HasLifetime>(S::AtLifetime<'static>);

impl<S: HasLifetime> ExistentialLifetime<S> {
    fn new() -> ExistentialLifetime<S> {
        ExistentialLifetime(ExistentialLifetime(()))
    }
}

fn main() {}
