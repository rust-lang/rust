


// -*- rust -*-
fn id<T>(x: &T) -> T { ret x; }

type triple = {x: int, y: int, z: int};

fn main() {
    let x = 62;
    let y = 63;
    let a = 'a';
    let b = 'b';
    let p: triple = {x: 65, y: 66, z: 67};
    let q: triple = {x: 68, y: 69, z: 70};
    y = id::<int>(x);
    log y;
    assert (x == y);
    b = id::<char>(a);
    log b;
    assert (a == b);
    q = id::<triple>(p);
    x = p.z;
    y = q.z;
    log y;
    assert (x == y);
}
