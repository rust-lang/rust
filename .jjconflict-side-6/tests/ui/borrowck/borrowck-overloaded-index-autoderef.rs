// Test that we still see borrowck errors of various kinds when using
// indexing and autoderef in combination.

use std::ops::{Index, IndexMut};



struct Foo {
    x: isize,
    y: isize,
}

impl<'a> Index<&'a String> for Foo {
    type Output = isize;

    fn index(&self, z: &String) -> &isize {
        if *z == "x" {
            &self.x
        } else {
            &self.y
        }
    }
}

impl<'a> IndexMut<&'a String> for Foo {
    fn index_mut(&mut self, z: &String) -> &mut isize {
        if *z == "x" {
            &mut self.x
        } else {
            &mut self.y
        }
    }
}

fn test1(mut f: Box<Foo>, s: String) {
    let p = &mut f[&s];
    let q = &f[&s]; //~ ERROR cannot borrow
    p.use_mut();
}

fn test2(mut f: Box<Foo>, s: String) {
    let p = &mut f[&s];
    let q = &mut f[&s]; //~ ERROR cannot borrow
    p.use_mut();
}

struct Bar {
    foo: Foo
}

fn test3(mut f: Box<Bar>, s: String) {
    let p = &mut f.foo[&s];
    let q = &mut f.foo[&s]; //~ ERROR cannot borrow
    p.use_mut();
}

fn test4(mut f: Box<Bar>, s: String) {
    let p = &f.foo[&s];
    let q = &f.foo[&s];
    p.use_ref();
}

fn test5(mut f: Box<Bar>, s: String) {
    let p = &f.foo[&s];
    let q = &mut f.foo[&s]; //~ ERROR cannot borrow
    p.use_ref();
}

fn test6(mut f: Box<Bar>, g: Foo, s: String) {
    let p = &f.foo[&s];
    f.foo = g; //~ ERROR cannot assign
    p.use_ref();
}

fn test7(mut f: Box<Bar>, g: Bar, s: String) {
    let p = &f.foo[&s];
    *f = g; //~ ERROR cannot assign
    p.use_ref();
}

fn test8(mut f: Box<Bar>, g: Foo, s: String) {
    let p = &mut f.foo[&s];
    f.foo = g; //~ ERROR cannot assign
    p.use_mut();
}

fn test9(mut f: Box<Bar>, g: Bar, s: String) {
    let p = &mut f.foo[&s];
    *f = g; //~ ERROR cannot assign
    p.use_mut();
}

fn main() {
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
