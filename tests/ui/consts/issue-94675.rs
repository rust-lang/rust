#![feature(const_trait_impl, const_vec_string_slice)]

struct Foo<'a> {
    bar: &'a mut Vec<usize>,
}

impl<'a> Foo<'a> {
    const fn spam(&mut self, baz: &mut Vec<u32>) {
        self.bar[0] = baz.len();
        //~^ ERROR: cannot call
    }
}

fn main() {}
