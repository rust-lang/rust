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
    //~^^ ERROR `Foo` is named the same as one of the variants of the type `foo::Foo`
    //~^^^ WARN unused variable: `Foo`
    }

    let Foo = foo::Foo::Foo;
    //~^ ERROR variable `Foo` should have a snake case name
    //~^^ ERROR `Foo` is named the same as one of the variants of the type `foo::Foo`
    //~^^^ WARN unused variable: `Foo`

    fn in_param(Foo: foo::Foo) {}
    //~^ ERROR variable `Foo` should have a snake case name
    //~^^ ERROR `Foo` is named the same as one of the variants of the type `foo::Foo`
    //~^^^ WARN unused variable: `Foo`

    let _: fn(CamelCase: i32);
    //~^ ERROR variable `CamelCase` should have a snake case name

    test(1);

    let _ = Something { X: 0 };
}
