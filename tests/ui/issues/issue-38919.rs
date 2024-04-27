fn foo<T: Iterator>() {
    T::Item; //~ ERROR no associated item named `Item` found
}

fn main() { }
