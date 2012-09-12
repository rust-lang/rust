// xfail-test
extern mod std;
use std::arena;
use std::arena::Arena;

enum hold { s(str) }

fn init(ar: &a.arena::Arena, str: str) -> &a.hold {
    new(*ar) s(str)
}

fn main(args: ~[str]) {
    let ar = arena::Arena();
    let leak = init(&ar, args[0]);
    match *leak {
        s(astr) {
            io::println(fmt!("%?", astr));
        }
    };
}
