enum StackList<'a, T: 'a> {
    Nil,
    Cons(T, &'a Self)
    //~^ ERROR cannot find type `Self` in this scope
    //~| `Self` is only available in traits and impls
}

fn main() {}
