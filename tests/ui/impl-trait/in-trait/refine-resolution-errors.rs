// This is a non-regression test for issue #126670 where RPITIT refinement checking encountered
// errors during resolution and ICEd.

//@ edition: 2018

pub trait Mirror {
    type Assoc;
}
impl<T: ?Sized> Mirror for () {
    //~^ ERROR the type parameter `T` is not constrained
    type Assoc = T;
}

pub trait First {
    async fn first() -> <() as Mirror>::Assoc;
}

impl First for () {
    async fn first() {}
}

fn main() {}
