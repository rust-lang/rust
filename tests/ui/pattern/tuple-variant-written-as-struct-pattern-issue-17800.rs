// https://github.com/rust-lang/rust/issues/17800
// Ensure that tuple enum variants cannot be matched using struct pattern syntax.
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
