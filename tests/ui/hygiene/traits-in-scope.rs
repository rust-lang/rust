// Macros with def-site hygiene still bring traits into scope.
// It is not clear whether this is desirable behavior or not.
// It is also not clear how to prevent it if it is not desirable.

//@ check-pass

#![feature(decl_macro)]
#![feature(trait_alias)]

mod traits {
    pub trait Trait1 {
        fn simple_import(&self) {}
    }
    pub trait Trait2 {
        fn renamed_import(&self) {}
    }
    pub trait Trait3 {
        fn underscore_import(&self) {}
    }
    pub trait Trait4 {
        fn trait_alias(&self) {}
    }

    impl Trait1 for () {}
    impl Trait2 for () {}
    impl Trait3 for () {}
    impl Trait4 for () {}
}

macro m1() {
    use traits::Trait1;
}
macro m2() {
    use traits::Trait2 as Alias;
}
macro m3() {
    use traits::Trait3 as _;
}
macro m4() {
    trait Alias = traits::Trait4;
}

fn main() {
    m1!();
    m2!();
    m3!();
    m4!();

    ().simple_import();
    ().renamed_import();
    ().underscore_import();
    ().trait_alias();
}
