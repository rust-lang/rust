// Semantically, an associated constant cannot use `_` as a name.

fn main() {}

const _: () = {
    pub trait A {
        const _: () = (); //~ ERROR `const` items in this context need a name
    }
    impl A for () {
        const _: () = (); //~ ERROR `const` items in this context need a name
        //~^ ERROR const `_` is not a member of trait `A`
    }
    struct B;
    impl B {
        const _: () = (); //~ ERROR `const` items in this context need a name
    }
};
