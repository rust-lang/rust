// test for #122098 ICE snapshot_vec.rs: index out of bounds: the len is 4 but the index is 4

trait LendingIterator {
    type Item<'q>: 'a;
    //~^ ERROR use of undeclared lifetime name `'a`

    fn for_each(mut self, mut f: Box<dyn FnMut(Self::Item<'_>) + 'static>) {}
    //~^ ERROR the size for values of type `Self` cannot be known at compilation time
}

struct Query<'q> {}
//~^ ERROR lifetime parameter `'q` is never used

impl<'static> Query<'q> {
//~^ ERROR invalid lifetime parameter name: `'static`
//~^^ ERROR use of undeclared lifetime name `'q`
    pub fn new() -> Self {}
}

fn data() {
    LendingIterator::for_each(Query::new(&data), Box::new);
    //~^ ERROR this function takes 0 arguments but 1 argument was supplied
}

pub fn main() {}
