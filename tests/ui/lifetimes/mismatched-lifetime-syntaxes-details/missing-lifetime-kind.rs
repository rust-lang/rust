#![deny(mismatched_lifetime_syntaxes)]

fn ampersand<'a>(x: &'a u8) -> &u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    x
}

struct Brackets<'a>(&'a u8);

fn brackets<'a>(x: &'a u8) -> Brackets {
    //~^ ERROR lifetime flowing from input to output with different syntax
    Brackets(x)
}

struct Comma<'a, T>(&'a T);

fn comma<'a>(x: &'a u8) -> Comma<u8> {
    //~^ ERROR lifetime flowing from input to output with different syntax
    Comma(x)
}

fn underscore<'a>(x: &'a u8) -> &'_ u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    x
}

fn main() {}
