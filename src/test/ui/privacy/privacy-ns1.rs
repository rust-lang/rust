// Check we do the correct privacy checks when we import a name and there is an
// item with that name in both the value and type namespaces.

#![allow(dead_code)]
#![allow(unused_imports)]


// public type, private value
pub mod foo1 {
    pub trait Bar {
    }
    pub struct Baz;

    fn Bar() { }
}

fn test_glob1() {
    use foo1::*;

    Bar();  //~ ERROR expected function, tuple struct or tuple variant, found trait `Bar`
}

// private type, public value
pub mod foo2 {
    trait Bar {
    }
    pub struct Baz;

    pub fn Bar() { }
}

fn test_glob2() {
    use foo2::*;

    let _x: Box<Bar>;  //~ ERROR wrong number of const arguments: expected 0, found 1
    //~^ ERROR wrong number of type arguments: expected at least 1, found 0
}

// neither public
pub mod foo3 {
    trait Bar {
    }
    pub struct Baz;

    fn Bar() { }
}

fn test_glob3() {
    use foo3::*;

    Bar();  //~ ERROR cannot find function, tuple struct or tuple variant `Bar` in this scope
    let _x: Box<Bar>;  //~ ERROR cannot find type `Bar` in this scope
}

fn main() {
}
