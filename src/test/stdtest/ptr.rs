use std;
import std::ptr;
import std::unsafe;

type pair = {mutable fst: int, mutable snd: int};

#[test]
fn test() {
    let p = {mutable fst: 10, mutable snd: 20};
    let pptr: *mutable pair = ptr::addr_of(p);
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

