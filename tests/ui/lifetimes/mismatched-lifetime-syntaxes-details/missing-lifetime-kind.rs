#![deny(mismatched_lifetime_syntaxes)]

fn ampersand<'a>(x: &'a u8) -> &u8 {
    //~^ ERROR eliding a lifetime that's named elsewhere is confusing
    x
}

struct Brackets<'a>(&'a u8);

fn brackets<'a>(x: &'a u8) -> Brackets {
    //~^ ERROR hiding a lifetime that's named elsewhere is confusing
    Brackets(x)
}

struct Comma<'a, T>(&'a T);

fn comma<'a>(x: &'a u8) -> Comma<u8> {
    //~^ ERROR hiding a lifetime that's named elsewhere is confusing
    Comma(x)
}

fn underscore<'a>(x: &'a u8) -> &'_ u8 {
    //~^ ERROR eliding a lifetime that's named elsewhere is confusing
    x
}

fn main() {}
