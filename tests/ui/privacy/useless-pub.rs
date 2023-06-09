struct A { pub i: isize }

pub trait E {
    fn foo(&self);
}

impl E for A {
    pub fn foo(&self) {} //~ ERROR: visibility qualifiers are not permitted here
}

enum Foo {
    V1 { pub f: i32 }, //~ ERROR visibility qualifiers are not permitted here
    V2(pub i32), //~ ERROR visibility qualifiers are not permitted here
}

fn main() {}
