#[forbid(non_camel_case_types)];

struct foo { //~ ERROR type, variant, or trait should have a camel case identifier
    bar: int,
}

enum foo2 { //~ ERROR type, variant, or trait should have a camel case identifier
    Bar
}

struct foo3 { //~ ERROR type, variant, or trait should have a camel case identifier
    bar: int
}

type foo4 = int; //~ ERROR type, variant, or trait should have a camel case identifier

enum Foo5 {
    bar //~ ERROR type, variant, or trait should have a camel case identifier
}

trait foo6 { //~ ERROR type, variant, or trait should have a camel case identifier
}

fn main() { }