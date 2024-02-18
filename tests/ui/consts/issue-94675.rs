//@ known-bug: #103507

#![feature(const_trait_impl, const_mut_refs)]

struct Foo<'a> {
    bar: &'a mut Vec<usize>,
}

impl<'a> Foo<'a> {
    const fn spam(&mut self, baz: &mut Vec<u32>) {
        self.bar[0] = baz.len();
        //FIXME ~^ ERROR: cannot call
        //FIXME ~| ERROR: cannot call
        //FIXME ~| ERROR: the trait bound
    }
}

fn main() {}
