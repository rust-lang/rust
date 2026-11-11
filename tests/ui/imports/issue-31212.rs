// This checks that a path that cannot be resolved because of an indeterminate import
// does not trigger an ICE.

mod foo {
    pub use self::*; //~ ERROR unresolved
}

fn main() {
    foo::f(); // cannot find function `f` in module `foo`, but silenced
}
