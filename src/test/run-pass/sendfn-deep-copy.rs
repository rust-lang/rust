use std;

import std::comm;
import std::comm::chan;
import std::comm::send;

fn main() { test05(); }

fn mk_counter<copy A>() -> sendfn(A) -> (A,uint) {
    // The only reason that the counter is generic is so that it closes
    // over both a type descriptor and some data.
    let v = [mutable 0u];
    ret sendfn(a: A) -> (A,uint) {
        let n = v[0];
        v[0] = n + 1u;
        (a, n)
    };
}

fn test05() {
    let fp0 = mk_counter::<float>();

    assert (5.3f, 0u) == fp0(5.3f);
    assert (5.5f, 1u) == fp0(5.5f);

    let fp1 = copy fp0;

    assert (5.3f, 2u) == fp0(5.3f);
    assert (5.3f, 2u) == fp1(5.3f);
    assert (5.5f, 3u) == fp0(5.5f);
    assert (5.5f, 3u) == fp1(5.5f);
}
