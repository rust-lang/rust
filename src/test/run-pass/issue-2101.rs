// xfail-test
use std;
import std::arena;
import std::arena::arena;

enum hold { s(str) }

fn init(ar: &a.arena::arena, str: str) -> &a.hold {
    new(*ar) s(str)
}

fn main(args: ~[str]) {
    let ar = arena::arena();
    let leak = init(&ar, args[0]);
    match *leak {
        s(astr) {
            io::println(fmt!{"%?", astr});
        }
    };
}
