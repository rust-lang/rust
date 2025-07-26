macro_rules! foo {
    ($ty:ty) => {
        fn foo(_: $ty, _: $ty) {}
    }
}

foo!(_);
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR the placeholder `_` is not allowed within types on item signatures for functions

fn main() {}
