fn foo<T: Iterator>() {
    T::Item; //~ ERROR no associated item named `Item` found for type `T` in the current scope
}

fn main() { }
