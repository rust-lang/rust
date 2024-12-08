enum Enum<'a, T, 'b> {
//~^ ERROR lifetime parameters must be declared prior to type and const parameters
    A(&'a &'b T)
}

struct Struct<'a, T, 'b> {
//~^ ERROR lifetime parameters must be declared prior to type and const parameters
    x: &'a &'b T
}

trait Trait<'a, T, 'b> {}
//~^ ERROR lifetime parameters must be declared prior to type and const parameters

fn foo<'a, T, 'b>(x: &'a T) {}
//~^ ERROR lifetime parameters must be declared prior to type and const parameters

struct Y<T>(T);
impl<'a, T, 'b> Y<T> {}
//~^ ERROR lifetime parameters must be declared prior to type and const parameters

mod bar {
    pub struct X<'a, 'b, 'c, T> {
        a: &'a str,
        b: &'b str,
        c: &'c str,
        t: T,
    }
}

fn bar<'a, 'b, 'c, T>(x: bar::X<'a, T, 'b, 'c>) {}
//~^ ERROR type provided when a lifetime was expected

fn main() {}
