#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code)]
#![allow(single_match)]
#![allow(unused_variables, similar_names)]
#![deny(blacklisted_name)]

fn test(foo: ()) {} //~ERROR use of a blacklisted/placeholder name `foo`

fn main() {
    let foo = 42; //~ERROR use of a blacklisted/placeholder name `foo`
    let bar = 42; //~ERROR use of a blacklisted/placeholder name `bar`
    let baz = 42; //~ERROR use of a blacklisted/placeholder name `baz`

    let barb = 42;
    let barbaric = 42;

    match (42, Some(1337), Some(0)) {
        (foo, Some(bar), baz @ Some(_)) => (),
        //~^ ERROR use of a blacklisted/placeholder name `foo`
        //~| ERROR use of a blacklisted/placeholder name `bar`
        //~| ERROR use of a blacklisted/placeholder name `baz`
        _ => (),
    }
}
