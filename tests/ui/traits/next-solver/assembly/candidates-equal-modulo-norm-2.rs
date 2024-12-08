//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#86. This previously
// failed with ambiguity due to multiple candidates with different
// normalization.

trait Bar {
    type Item;
    type Assoc: AsRef<[Self::Item]>;
}

struct Foo<T: Bar> {
    t: <T as Bar>::Assoc,
}

impl<T: Bar<Item = u32>> Foo<T>
where
    <T as Bar>::Assoc: AsRef<[u32]>,
{
    fn hello(&self) {
        println!("{}", self.t.as_ref().len());
    }
}

fn main() {}
