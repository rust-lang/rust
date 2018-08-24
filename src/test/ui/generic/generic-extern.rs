extern {
    fn foo<T>(); //~ ERROR foreign items may not have type parameters
}

fn main() {
    foo::<i32>();
}
