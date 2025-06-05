//@ edition: 2015
//@ run-pass

#![deny(warnings)]
#![allow(unused_imports)]

pub enum Foo { A }
mod bar {
    pub fn normal(x: ::Foo) {
        use Foo::A;
        match x {
            A => {}
        }
    }
    pub fn wrong(x: ::Foo) {
        match x {
            ::Foo::A => {}
        }
    }
}

pub fn main() {
    bar::normal(Foo::A);
    bar::wrong(Foo::A);
}
