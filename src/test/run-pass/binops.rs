// Binop corner cases

use std;
use unsafe::reinterpret_cast;

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
}

fn test_ptr() unsafe {
    let p1: *u8 = unsafe::reinterpret_cast(&0);
    let p2: *u8 = unsafe::reinterpret_cast(&0);
    let p3: *u8 = unsafe::reinterpret_cast(&1);

    assert p1 == p2;
    assert p1 != p3;
    assert p1 < p3;
    assert p1 <= p3;
    assert p3 > p1;
    assert p3 >= p3;
    assert p1 <= p2;
    assert p1 >= p2;
}

#[abi = "cdecl"]
#[nolink]
extern mod test {
    fn rust_get_sched_id() -> libc::intptr_t;
    fn get_task_id() -> libc::intptr_t;
}

struct p {
  let mut x: int;
  let mut y: int;
}

fn p(x: int, y: int) -> p {
    p {
        x: x,
        y: y
    }
}

impl p : cmp::Eq {
    pure fn eq(&&other: p) -> bool {
        self.x == other.x && self.y == other.y
    }
}

fn test_class() {
  let q = p(1, 2);
  let r = p(1, 2);
  
  unsafe {
  error!("q = %x, r = %x",
         (unsafe::reinterpret_cast::<*p, uint>(&ptr::addr_of(q))),
         (unsafe::reinterpret_cast::<*p, uint>(&ptr::addr_of(r))));
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
    test_ptr();
    test_class();
}
