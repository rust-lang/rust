//Test for https://github.com/rust-lang/rust/issues/52049
//Tests that a non-promotable temp variable cannot be used as a static reference.
fn foo(_: &'static u32) {}

fn unpromotable<T>(t: T) -> T { t }

fn main() {
    foo(&unpromotable(5u32));
}
//~^^ ERROR temporary value dropped while borrowed
