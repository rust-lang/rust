pub extern { //~ ERROR unnecessary visibility qualifier
    pub fn bar();
}

trait A {
    fn foo(&self) {}
}

struct B;

pub impl B {} //~ ERROR unnecessary visibility qualifier

pub impl A for B { //~ ERROR unnecessary visibility qualifier
    pub fn foo(&self) {} //~ ERROR unnecessary visibility qualifier
}

pub fn main() {}
