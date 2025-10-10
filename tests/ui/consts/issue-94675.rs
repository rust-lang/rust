//@ ignore-backends: gcc

#![feature(const_trait_impl)]

struct Foo<'a> {
    bar: &'a mut Vec<usize>,
}

impl<'a> Foo<'a> {
    const fn spam(&mut self, baz: &mut Vec<u32>) {
        self.bar[0] = baz.len();
        //~^ ERROR: `Vec<usize>: [const] Index<_>` is not satisfied
        //~| ERROR: `Vec<usize>: [const] Index<usize>` is not satisfied
        //~| ERROR: `Vec<usize>: [const] IndexMut<usize>` is not satisfied
    }
}

fn main() {}
