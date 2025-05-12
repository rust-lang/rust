fn foo(x: bool) -> i32 {
    match x { //~ ERROR struct literals are not allowed here
        x: i32 => x, //~ ERROR expected
        true => 42., //~ ERROR expected identifier
        false => 0.333, //~ ERROR expected identifier
    }
} //~ ERROR expected one of

fn main() {
    match foo(true) {
        42: i32 => (), //~ ERROR expected
        _: f64 => (), //~ ERROR expected
        x: i32 => (), //~ ERROR expected
    }
}
