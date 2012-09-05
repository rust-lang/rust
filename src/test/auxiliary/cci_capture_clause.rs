export foo;

use comm::*;

fn foo<T: send copy>(x: T) -> Port<T> {
    let p = Port();
    let c = Chan(p);
    do task::spawn() |copy c, copy x| {
        c.send(x);
    }
    p
}
