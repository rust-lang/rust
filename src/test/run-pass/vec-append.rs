

// -*- rust -*-
use std;
import std::str;
import std::vec;

// FIXME: import std::dbg::const_refcount. Currently
// cross-crate const references don't work.
const const_refcount: uint = 0x7bad_face_u;

fn fast_growth() {
    let v: [int] = ~[1, 2, 3, 4, 5];
    v += ~[6, 7, 8, 9, 0];
    log v.(9);
    assert (v.(0) == 1);
    assert (v.(7) == 8);
    assert (v.(9) == 0);
}

fn slow_growth() {
    let v: [int] = ~[];
    let u: [int] = v;
    v += ~[17];
    log v.(0);
    assert (v.(0) == 17);
}

fn slow_growth2_helper(s: str) { // ref up: s

    obj acc(mutable v: [str]) {
        fn add(s: &str) { v += ~[s]; }
    }
    let ss: str = s; // ref up: s

    let mumble: str = "mrghrm"; // ref up: mumble

    {
        /**
         * Within this block, mumble goes into a vec that is referenced
         * both by the local slot v and the acc's v data field.  When we
         * add(s) on the acc, its v undergoes a slow append (allocate a
         * new vec, copy over existing elements).  Here we're testing to
         * see that this slow path goes over well.  In particular, the
         * copy of existing elements should increment the ref count of
         * mumble, the existing str in the originally- shared vec.
         */

        let v: [str] = ~[mumble]; // ref up: mumble

        let a: acc = acc(v);

        a.add(s); // ref up: mumble, s

        log str::refcount(s);
        log str::refcount(mumble);
        assert (str::refcount(s) == const_refcount);
        assert (str::refcount(mumble) == const_refcount);
        log v.(0);
        log vec::len[str](v);
        assert (str::eq(v.(0), mumble));
        assert (vec::len[str](v) == 1u);
    } // ref down: mumble, s,

    log str::refcount(s);
    log str::refcount(mumble);
    assert (str::refcount(s) == const_refcount);
    assert (str::refcount(mumble) == const_refcount);
    log mumble;
    log ss;
}

// ref down
fn slow_growth2() {
    let s: str = "hi"; // ref up: s

    slow_growth2_helper(s);
    log str::refcount(s);
    assert (str::refcount(s) == const_refcount);
}

fn main() { fast_growth(); slow_growth(); slow_growth2(); }