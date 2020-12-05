// Also works as a test for #14564

#[allow(non_snake_case)]
mod Foo { }

#[allow(dead_code)]
struct Foo;
//~^ ERROR the name `Foo` is defined multiple times

#[allow(non_snake_case)]
mod Bar { }

#[allow(dead_code)]
struct Bar(i32);
//~^ ERROR the name `Bar` is defined multiple times


#[allow(dead_code)]
struct Baz(i32);

#[allow(non_snake_case)]
mod Baz { }
//~^ ERROR the name `Baz` is defined multiple times


#[allow(dead_code)]
struct Qux { x: bool }

#[allow(non_snake_case)]
mod Qux { }
//~^ ERROR the name `Qux` is defined multiple times


#[allow(dead_code)]
struct Quux;

#[allow(non_snake_case)]
mod Quux { }
//~^ ERROR the name `Quux` is defined multiple times


#[allow(dead_code)]
enum Corge { A, B }

#[allow(non_snake_case)]
mod Corge { }
//~^ ERROR the name `Corge` is defined multiple times

fn main() { }
