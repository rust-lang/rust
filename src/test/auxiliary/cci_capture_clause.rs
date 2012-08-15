export foo;

import comm::*;

fn foo<T: send copy>(x: T) -> Port<T> {
    let p = port();
    let c = chan(p);
    do task::spawn() |copy c, copy x| {
        c.send(x);
    }
    p
}
