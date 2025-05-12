#![deny(elided_named_lifetimes)]

fn ampersand<'a>(x: &'a u8) -> &u8 {
    //~^ ERROR elided lifetime has a name
    x
}

struct Brackets<'a>(&'a u8);

fn brackets<'a>(x: &'a u8) -> Brackets {
    //~^ ERROR elided lifetime has a name
    Brackets(x)
}

struct Comma<'a, T>(&'a T);

fn comma<'a>(x: &'a u8) -> Comma<u8> {
    //~^ ERROR elided lifetime has a name
    Comma(x)
}

fn underscore<'a>(x: &'a u8) -> &'_ u8 {
    //~^ ERROR elided lifetime has a name
    x
}

fn main() {}
