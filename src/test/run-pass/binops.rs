// Binop corner cases

use std;
import std::unsafe::reinterpret_cast;
import std::task;
import std::comm;

fn test_nil() {
    assert (() == ());
    assert (!(() != ()));
    assert (!(() < ()));
    assert (() <= ());
    assert (!(() > ()));
    assert (() >= ());
}

fn test_bool() {
    assert (!(true < false));
    assert (!(true <= false));
    assert (true > false);
    assert (true >= false);

    assert (false < true);
    assert (false <= true);
    assert (!(false > true));
    assert (!(false >= true));

    // Bools support bitwise binops
    assert (false & false == false);
    assert (true & false == false);
    assert (true & true == true);
    assert (false | false == false);
    assert (true | false == true);
    assert (true | true == true);
    assert (false ^ false == false);
    assert (true ^ false == true);
    assert (true ^ true == false);
}

fn test_char() {
    let ch10 = 10 as char;
    let ch4 = 4 as char;
    let ch2 = 2 as char;
    assert (ch10 + ch4 == 14 as char);
    assert (ch10 - ch4 == 6 as char);
    assert (ch10 * ch4 == 40 as char);
    assert (ch10 / ch4 == ch2);
    assert (ch10 % ch4 == ch2);
    assert (ch10 >> ch2 == ch2);
    assert (ch10 >>> ch2 == ch2);
    assert (ch10 << ch4 == 160 as char);
    assert (ch10 | ch4 == 14 as char);
    assert (ch10 & ch2 == ch2);
    assert (ch10 ^ ch2 == 8 as char);
}

fn test_box() {
    assert (@10 == @10);
    assert (@{a: 1, b: 3} < @{a: 1, b: 4});
    assert (@{a: 'x'} != @{a: 'y'});
}

fn test_port() {
    // FIXME: Re-enable this once we can compare resources.
    /*
    let p1 = comm::port::<int>();
    let p2 = comm::port::<int>();

    assert (p1 == p1);
    assert (p1 != p2);
    */
}

fn test_chan() {
    let p: comm::port<int> = comm::port();
    let ch1 = comm::chan(p);
    let ch2 = comm::chan(p);

    assert (ch1 == ch1);
    // Chans are equal because they are just task:port addresses.
    assert (ch1 == ch2);
}

fn test_ptr() {
    // FIXME: Don't know what binops apply to pointers. Don't know how
    // to make or use pointers
}

fn test_task() {
    fn f() { }
    let f1 = f, f2 = f;
    let t1 = task::spawn(f1);
    let t2 = task::spawn(f2);

    assert (t1 == t1);
    assert (t1 != t2);
}

fn test_fn() {
    fn f() { }
    fn g() { }
    fn h(i: int) { }
    let f1 = f;
    let f2 = f;
    let g1 = g;
    let h1 = h;
    let h2 = h;
    assert (f1 == f2);
    assert (f1 == f);

    assert (f1 != g1);
    assert (h1 == h2);
    assert (!(f1 != f2));
    assert (!(h1 < h2));
    assert (h1 <= h2);
    assert (!(h1 > h2));
    assert (h1 >= h2);
}

native "rust" mod native_mod = "" {
    fn str_byte_len(s: str) -> uint;
    // This isn't actually the signature of str_alloc, but since
    // we're not calling it that shouldn't matter
    fn str_alloc(s: str) -> uint;
}

// FIXME: comparison of native fns
fn test_native_fn() {
    assert (native_mod::str_byte_len == native_mod::str_byte_len);
    assert (native_mod::str_byte_len != native_mod::str_alloc);
}

fn test_obj() {
    let o1 = obj () { };
    let o2 = obj () { };

    assert (o1 == o1);

    // FIXME (#815): This doesn't work on linux only. Wierd.
    //assert (o1 != o2);
    //assert (!(o1 == o2));

    obj constr1(i: int) { }
    obj constr2(i: int) { }

    let o5 = constr1(10);
    let o6 = constr1(10);
    let o7 = constr1(11);
    let o8 = constr2(11);

    assert (o5 != o6);
    assert (o6 != o7);
    assert (o7 != o8);
}

fn main() {
    test_nil();
    test_bool();
    test_char();
    test_box();
    test_port();
    test_chan();
    test_ptr();
    test_task();
    test_fn();
    test_native_fn();
    test_obj();
}
