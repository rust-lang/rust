struct A { pub i: isize }

pub trait E {
    fn foo(&self);
}

impl E for A {
    pub fn foo(&self) {} //~ ERROR: unnecessary visibility qualifier
}

enum Foo {
    V1 { pub f: i32 }, //~ ERROR unnecessary visibility qualifier
    V2(pub i32), //~ ERROR unnecessary visibility qualifier
}

fn main() {}
