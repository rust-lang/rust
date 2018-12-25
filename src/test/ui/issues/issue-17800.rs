enum MyOption<T> {
    MySome(T),
    MyNone,
}

fn main() {
    match MyOption::MySome(42) {
        MyOption::MySome { x: 42 } => (),
        //~^ ERROR variant `MyOption::MySome` does not have a field named `x`
        _ => (),
    }
}
