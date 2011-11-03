
use std;
import std::uint;

fn test(foo: @{a: int, b: int, c: int}) -> @{a: int, b: int, c: int} {
    let foo = foo;
    let bar <- foo;
    let baz <- bar;
    let quux <- baz;
    ret quux;
}

fn main() { let x = @{a: 1, b: 2, c: 3}; let y = test(x); assert (y.c == 3); }
