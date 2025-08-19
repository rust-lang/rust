#![deny(mismatched_lifetime_syntaxes)]

use std::borrow::Cow;

const A: &[u8] = &[];
static B: &str = "hello";

trait Trait {
    const C: &u8 = &0;
}

impl Trait for () {
    const C: &u8 = &1;
}

fn ampersand(x: &'static u8) -> &u8 {
    //~^ ERROR eliding a lifetime that's named elsewhere is confusing
    x
}

struct Brackets<'a>(&'a u8);

fn brackets(x: &'static u8) -> Brackets {
    //~^ ERROR hiding a lifetime that's named elsewhere is confusing
    Brackets(x)
}

struct Comma<'a, T>(&'a T);

fn comma(x: &'static u8) -> Comma<u8> {
    //~^ ERROR hiding a lifetime that's named elsewhere is confusing
    Comma(x)
}

fn underscore(x: &'static u8) -> &'_ u8 {
    //~^ ERROR eliding a lifetime that's named elsewhere is confusing
    x
}

const NESTED: &Vec<&Box<Cow<str>>> = &vec![];

fn main() {
    const HELLO: &str = "Hello";
    static WORLD: &str = "world";
    println!("{HELLO}, {WORLD}!")
}
