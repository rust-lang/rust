fn foo() {}

fn main() {
    while let Some(foo) = Some(1) { break }
    foo();
}
