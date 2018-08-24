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

fn test_single1() {
    use foo1::Bar;

    Bar(); //~ ERROR expected function, found trait `Bar`
}

fn test_list1() {
    use foo1::{Bar,Baz};

    Bar(); //~ ERROR expected function, found trait `Bar`
}

// private type, public value
pub mod foo2 {
    trait Bar {
    }
    pub struct Baz;

    pub fn Bar() { }
}

fn test_single2() {
    use foo2::Bar;

    let _x : Box<Bar>; //~ ERROR expected type, found function `Bar`
}

fn test_list2() {
    use foo2::{Bar,Baz};

    let _x: Box<Bar>; //~ ERROR expected type, found function `Bar`
}

// neither public
pub mod foo3 {
    trait Bar {
    }
    pub struct Baz;

    fn Bar() { }
}

fn test_unused3() {
    use foo3::Bar;  //~ ERROR `Bar` is private
}

fn test_single3() {
    use foo3::Bar;  //~ ERROR `Bar` is private

    Bar();
    let _x: Box<Bar>;
}

fn test_list3() {
    use foo3::{Bar,Baz};  //~ ERROR `Bar` is private

    Bar();
    let _x: Box<Bar>;
}

fn main() {
}
