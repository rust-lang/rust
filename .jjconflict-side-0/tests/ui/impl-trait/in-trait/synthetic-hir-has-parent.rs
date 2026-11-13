// Don't panic when iterating through the `hir::Map::parent_iter` of an RPITIT.

pub trait Foo {
    fn demo() -> impl Foo
    //~^ ERROR the trait bound `String: Copy` is not satisfied
    where
        String: Copy;
    //~^ ERROR the trait bound `String: Copy` is not satisfied
}

fn main() {}
