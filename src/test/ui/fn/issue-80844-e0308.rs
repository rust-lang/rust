#![feature(generators)]

// Functions with a type placeholder `_` as the return type should
// not suggest returning the unnameable type of generators.
// This is a regression test of #80844

struct Container<T>(T);

fn expected_unit_got_generator() {
//~^ NOTE possibly return type missing here?
    || yield 0i32
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected unit type `()`
    //~| NOTE expected `()`, found generator
}

fn expected_unit_got_closure() {
//~^ NOTE possibly return type missing here?
    || 0i32
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected unit type `()`
    //~| NOTE expected `()`, found closure
}

fn expected_unit_got_option_closure() {
//~^ NOTE possibly return type missing here?
    Some(|| 0i32)
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected unit type `()`
    //~| NOTE expected `()`, found enum `Option`
    //~| HELP try adding a semicolon
}

fn expected_unit_got_option_i32() {
//~^ NOTE possibly return type missing here?
    Some(0i32)
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected unit type `()`
    //~| NOTE expected `()`, found enum `Option`
    //~| HELP try adding a semicolon
}

fn expected_unit_got_container_closure() {
//~^ NOTE possibly return type missing here?
    Container(|| 0i32)
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected unit type `()`
    //~| NOTE expected `()`, found struct `Container`
    //~| HELP try adding a semicolon
}

fn expected_unit_got_container_i32() {
//~^ NOTE possibly return type missing here?
    Container(0i32)
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected unit type `()`
    //~| NOTE expected `()`, found struct `Container`
    //~| HELP try adding a semicolon
}

fn main() {}
