//@ compile-flags: -Znext-solver

// Regression test for trait-system-refactor-initiative#267. This recursively
// changing opaque type used to overflow the stack while instantiating a
// canonical response.

trait Distribution<T> {}

impl<A, B> Distribution<(A, B)> for u32
where
    u32: Distribution<A>,
    u32: Distribution<B>,
{
}

fn require_distribution<U: Distribution<T>, T>(_: *mut T) {}

fn random_paulis() -> Option<*mut impl Sized> {
    if false {
        let r = random_paulis().unwrap();
        //~^ ERROR type annotations needed
        require_distribution::<u32, _>(r);
    }

    None
}

fn main() {}
