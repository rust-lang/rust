// issue: 113903

use std::ops::Deref;

pub trait Tr {
    fn w() -> impl Deref<Target = Missing<impl Sized>>;
    //~^ ERROR cannot find type `Missing` in this scope
}

impl Tr for () {
    #[expect(refining_impl_trait)]
    fn w() -> &'static () {
        &()
    }
}

fn main() {}
