//@ run-rustfix
#![allow(unused)]
enum Foo {
    Foo1(Box<u32>, Box<u32>),
    Foo2(Box<u32>),
    Foo3,
}



fn blah() {
    let f = &Foo::Foo1(Box::new(1), Box::new(2));
    match *f { //~ ERROR cannot move out of
        Foo::Foo1(num1,
                  num2) => (),
        Foo::Foo2(num) => (),
        Foo::Foo3 => ()
    }
}

struct S {
    f: String,
    g: String
}
impl Drop for S {
    fn drop(&mut self) { println!("{}", self.f); }
}

fn move_in_match() {
    match (S {f: "foo".to_string(), g: "bar".to_string()}) {
        //~^ ERROR cannot move out of type `S`, which implements the `Drop` trait
        S {
            f: _s,
            g: _t
        } => {}
    }
}

// from issue-8064
struct A {
    a: Box<isize>,
}

fn free<T>(_: T) {}

fn blah2() {
    let a = &A { a: Box::new(1) };
    match a.a { //~ ERROR cannot move out of
        n => {
            free(n)
        }
    }
    free(a)
}

fn main() {}
