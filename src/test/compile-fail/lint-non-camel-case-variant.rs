#[forbid(non_camel_case_types)]
enum Foo {
    bar //~ ERROR type, variant, or trait must be camel case
}

fn main() {
}
