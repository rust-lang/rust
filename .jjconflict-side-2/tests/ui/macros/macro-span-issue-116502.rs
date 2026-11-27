#![allow(dead_code)]
#![allow(unused_variables)]

fn bug() {
    macro_rules! m {
        () => {
            _ //~ ERROR the placeholder `_` is not allowed within types on item signatures for structs
            //~^ ERROR the placeholder `_` is not allowed within types on item signatures for structs
            //~| ERROR the placeholder `_` is not allowed within types on item signatures for structs
        };
    }
    struct S<T = m!()>(m!(), T)
    where
        T: Trait<m!()>;
}
trait Trait<T> {}

fn main() {}
