fn foo(x: bool | i32) -> i32 | f64 {
//~^ ERROR anonymous enums are not supported
//~| ERROR anonymous enums are not supported
    match x {
        x: i32 => x, //~ ERROR expected
        true => 42.,
        false => 0.333,
    }
}

fn main() {
    match foo(true) {
        42: i32 => (), //~ ERROR expected
        _: f64 => (), //~ ERROR expected
        x: i32 => (), //~ ERROR expected
    }
}
