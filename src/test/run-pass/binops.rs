// Binop corner cases

use std;
import unsafe::reinterpret_cast;
import task;
import comm;

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
    let p1 = comm::port::<int>();
    let p2 = comm::port::<int>();

    assert (p1 == p1);
    assert (p1 != p2);
}

fn test_chan() {
    let p: comm::port<int> = comm::port();
    let ch1 = comm::chan(p);
    let ch2 = comm::chan(p);

    assert (ch1 == ch1);
    // Chans are equal because they are just task:port addresses.
    assert (ch1 == ch2);
}

fn test_ptr() unsafe {
    let p1: *u8 = unsafe::reinterpret_cast(0);
    let p2: *u8 = unsafe::reinterpret_cast(0);
    let p3: *u8 = unsafe::reinterpret_cast(1);

    assert p1 == p2;
    assert p1 != p3;
    assert p1 < p3;
    assert p1 <= p3;
    assert p3 > p1;
    assert p3 >= p3;
    assert p1 <= p2;
    assert p1 >= p2;
}

fn test_fn() {
    fn f() { }
    fn g() { }
    fn h(_i: int) { }
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

#[abi = "cdecl"]
#[nolink]
native mod test {
    fn unsupervise();
    fn get_task_id();
}

fn test_native_fn() {
    assert test::unsupervise != test::get_task_id;
    assert test::unsupervise == test::unsupervise;
}

class p {
  let mut x: int;
  let mut y: int;
  new(x: int, y: int) { self.x = x; self.y = y; }
}

fn test_class() {
  let q = p(1, 2);
  let r = p(1, 2);
  
  unsafe {
  #error("q = %x, r = %x",
         (unsafe::reinterpret_cast::<*p, uint>(ptr::addr_of(q))),
         (unsafe::reinterpret_cast::<*p, uint>(ptr::addr_of(r))));
  }
  assert(q == r);
  r.y = 17;
  assert(r.y != q.y);
  assert(r.y == 17);
  assert(q != r);
}

fn main() {
    test_nil();
    test_bool();
    test_char();
    test_box();
    test_port();
    test_chan();
    test_ptr();
    test_fn();
    test_native_fn();
    test_class();
}
