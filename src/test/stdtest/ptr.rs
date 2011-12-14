import core::*;

use std;
import ptr;
import unsafe;

type pair = {mutable fst: int, mutable snd: int};

#[test]
fn test() unsafe {
    let p = {mutable fst: 10, mutable snd: 20};
    let pptr: *mutable pair = ptr::mut_addr_of(p);
    let iptr: *mutable int = unsafe::reinterpret_cast(pptr);
    assert (*iptr == 10);;
    *iptr = 30;
    assert (*iptr == 30);
    assert (p.fst == 30);;

    *pptr = {mutable fst: 50, mutable snd: 60};
    assert (*iptr == 50);
    assert (p.fst == 50);
    assert (p.snd == 60);
}

