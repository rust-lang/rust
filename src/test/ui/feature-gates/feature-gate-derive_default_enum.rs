#[derive(Default)] //~ ERROR deriving `Default` on enums is experimental
enum Foo {
    #[default]
    Alpha,
}

fn main() {}
