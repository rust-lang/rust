#![feature(const_trait_impl, const_mut_refs)]

struct Foo<'a> {
    bar: &'a mut Vec<usize>,
}

impl<'a> Foo<'a> {
    const fn spam(&mut self, baz: &mut Vec<u32>) {
        self.bar[0] = baz.len();
        //~^ the trait bound `Vec<usize>: ~const Index<_>` is not satisfied
        //~| the trait bound `Vec<usize>: ~const IndexMut<usize>` is not satisfied
    }
}

fn main() {}
