// xfail-stage0

use std;
import std::ptr;
import std::unsafe;

type pair = rec(mutable int fst, mutable int snd);

fn main() {
    auto p = rec(mutable fst=10, mutable snd=20);
    let *mutable pair pptr = ptr::addr_of(p);
    let *mutable int iptr = unsafe::reinterpret_cast(pptr);
    assert (*iptr == 10);
    *iptr = 30;
    assert (*iptr == 30);
    assert (p.fst == 30);

    *pptr = rec(mutable fst=50, mutable snd=60);
    assert (*iptr == 50);
    assert (p.fst == 50);
    assert (p.snd == 60);
}

