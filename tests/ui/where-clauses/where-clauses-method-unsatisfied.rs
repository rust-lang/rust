//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
// Test that a where clause attached to a method allows us to add
// additional constraints to a parameter out of scope.

struct Foo<T> {
    value: T,
}

struct Bar; // does not implement Eq

impl<T> Foo<T> {
    fn equals(&self, u: &Foo<T>) -> bool
    where
        T: Eq,
    {
        self.value == u.value
    }
}

fn main() {
    let x = Foo { value: Bar };
    x.equals(&x);
    //~^ ERROR `Bar: Eq` is not satisfied
}
