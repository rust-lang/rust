#![feature(const_trait_impl, const_mut_refs)]

struct Foo<'a> {
    bar: &'a mut Vec<usize>,
}

impl<'a> Foo<'a> {
    const fn spam(&mut self, baz: &mut Vec<u32>) {
        self.bar[0] = baz.len();
        //~^ ERROR cannot call non-const fn `Vec::<u32>::len` in constant functions
        //~| ERROR the trait bound `Vec<usize>: ~const IndexMut<usize>` is not satisfied
        //~| ERROR cannot call non-const operator in constant functions
    }
}

fn main() {}
