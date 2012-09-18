#[forbid(non_camel_case_types)];

struct foo { //~ ERROR type, variant, or trait must be camel case
    bar: int,
}

enum foo2 { //~ ERROR type, variant, or trait must be camel case
    Bar
}

struct foo3 { //~ ERROR type, variant, or trait must be camel case
    bar: int
}

type foo4 = int; //~ ERROR type, variant, or trait must be camel case

enum Foo5 {
    bar //~ ERROR type, variant, or trait must be camel case
}

trait foo6 { //~ ERROR type, variant, or trait must be camel case
}

fn main() { }