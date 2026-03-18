// https://github.com/rust-lang/rust/issues/17800

enum MyOption<T> {
    MySome(T),
    MyNone,
}

fn main() {
    match MyOption::MySome(42) {
        MyOption::MySome { x: 42 } => (),
        //~^ ERROR tuple variant `MyOption::MySome` written as struct variant
        _ => (),
    }
}
