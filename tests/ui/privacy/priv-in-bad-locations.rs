pub extern "C" { //~ ERROR visibility qualifiers are not permitted here
    pub fn bar();
}

trait A {
    fn foo(&self) {}
}

struct B;

pub impl B {} //~ ERROR visibility qualifiers are not permitted here

pub impl A for B { //~ ERROR visibility qualifiers are not permitted here
    pub fn foo(&self) {} //~ ERROR visibility qualifiers are not permitted here
}

pub fn main() {}
