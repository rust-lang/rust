#[legacy_exports];

export foo;

use comm::*;

fn foo<T: Send Copy>(x: T) -> Port<T> {
    let p = Port();
    let c = Chan(&p);
    do task::spawn() |copy c, copy x| {
        c.send(x);
    }
    p
}
