//@ check-pass
//@ revisions: first last
//
// `Foo` is in scope both via an ambiguous named import and an unambiguous
// `as _` one. The lint reports regardless of the order of the two `pub use`s.
//
// Issue: #160742

mod alias {
    pub type Foo = u8;
}

mod def {
    pub trait Foo {
        fn method(self);
    }

    impl Foo for i32 {
        fn method(self) {}
    }
}

#[cfg(first)]
mod export {
    pub use crate::def::Foo as _;
    pub use crate::def::Foo;
}

#[cfg(last)]
mod export {
    pub use crate::def::Foo;
    pub use crate::def::Foo as _;
}

#[allow(unused_imports)] // needed only to make the trait ambiguous
use alias::*;
use export::*;

fn main() {
    1i32.method();
    //~^ WARNING Use of ambiguously glob imported trait `Foo` [ambiguous_glob_imported_traits]
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
