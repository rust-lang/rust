use std::marker::PhantomData;

type Component = fn(&());

struct Wrapper {
    router: Router<(Component, Box<Self>)>,
}

struct Match<C>(PhantomData<C>);

struct Router<T>(PhantomData<T>);

impl<T> Router<T> {
    pub fn at(&self) -> Result<Match<&T>, ()> {
        todo!()
    }
}

impl Wrapper {
    fn at(&self, path: &str) -> Result<(Component, Box<Self>), ()> {
        let (cmp, router) = self.router.at()?;
        //~^ ERROR mismatched types
        todo!()
    }
}

fn main() {}
