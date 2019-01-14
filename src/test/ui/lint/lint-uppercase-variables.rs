#![warn(unused)]
#![allow(dead_code)]
#![deny(non_snake_case)]

mod foo {
    pub enum Foo { Foo }
}

struct Something {
    X: usize //~ ERROR structure field `X` should have a snake case name
}

fn test(Xx: usize) { //~ ERROR variable `Xx` should have a snake case name
    println!("{}", Xx);
}

fn main() {
    let Test: usize = 0; //~ ERROR variable `Test` should have a snake case name
    println!("{}", Test);

    match foo::Foo::Foo {
        Foo => {}
//~^ ERROR variable `Foo` should have a snake case name
//~^^ WARN `Foo` is named the same as one of the variants of the type `foo::Foo`
//~^^^ WARN unused variable: `Foo`
    }

    test(1);

    let _ = Something { X: 0 };
}
