// Check that imports with naked super and self don't fail during parsing
// FIXME: this shouldn't fail during name resolution either

mod a {
    mod b {
        use self as A;
        //~^ ERROR `self` imports are only allowed within a { } list
        use super as B;
        //~^ ERROR unresolved import `super` [E0432]
        //~| no `super` in the root
        use super::{self as C};
        //~^ ERROR unresolved import `super` [E0432]
        //~| no `super` in the root
    }
}

fn main() {}
