fn main() {}

struct MyType;

const impl Iterator for MyType { //~ ERROR: const trait impls are experimental [E0658]
    //~^ ERROR: use of unstable const library feature `const_iter` [E0658]
    type Item = ();
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

const impl ExactSizeIterator for MyType {} //~ ERROR: const trait impls are experimental [E0658]
//~^ ERROR: use of unstable const library feature `const_exact_size_iterator` [E0658]
