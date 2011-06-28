// Binop corner cases

fn test_nil() {
  assert () == ();
  assert !(() != ());
  // FIXME (#576): The current implementation of relational ops on nil
  // is nonsensical
  assert () < ();
  assert () <= ();
  assert !(() > ());
  assert !(() >= ());
}

fn test_bool() {
  assert !(true < false);
  assert !(true <= false);
  assert (true > false);
  assert (true >= false);

  assert (false < true);
  assert (false <= true);
  assert !(false > true);
  assert !(false >= true);

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
  auto ch10 = 10 as char;
  auto ch4 = 4 as char;
  auto ch2 = 2 as char;
  assert ch10 + ch4 == 14 as char;
  assert ch10 - ch4 == 6 as char;
  assert ch10 * ch4 == 40 as char;
  assert ch10 / ch4 == ch2;
  assert ch10 % ch4 == ch2;
  assert ch10 >> ch2 == ch2;
  assert ch10 >>> ch2 == ch2;
  assert ch10 << ch4 == 160 as char;
  assert ch10 | ch4 == 14 as char;
  assert ch10 & ch2 == ch2;
  assert ch10 ^ ch2 == 8 as char;
}

fn test_box() {
  assert @10 == 10;
  assert 0xFF & @0xF0 == 0xF0;
  assert tup(1, 3) < @tup(1, 4);
  assert @rec(a = 'x') != @rec(a = 'y');
}

fn test_port() {
  let port[int] p1 = port();
  let port[int] p2 = port();

  // FIXME (#577) comparison of ports
  // assert (p1 != p2);
  // assert !(p1 < p2);
  // etc
}

fn test_chan() {
  let port[int] p = port();
  auto ch1 = chan(p);
  auto ch2 = chan(p);

  // FIXME (#577) comparison of channels
  // assert (ch1 != ch2);
  // etc
}

fn test_ptr() {
  // FIXME: Don't know what binops apply to pointers. Don't know how
  // to make or use pointers
}

fn test_task() {
  fn f() {}
  auto t1 = spawn f();
  auto t2 = spawn f();
  // FIXME (#577) comparison of tasks
  //assert t1 != t2;
}

fn test_fn() {
  fn f() {}
  fn g() {}
  fn h(int i) {}
  auto f1 = f;
  auto f2 = f;
  auto g1 = g;
  auto h1 = h;
  auto h2 = h;
  assert (f1 == f2);
  assert (f1 == f);

  assert (f1 == g1);
  assert (h1 == h2);
  assert !(f1 != f2);
  assert !(h1 < h2);
  assert (h1 <= h2);
  assert !(h1 > h2);
  assert (h1 >= h2);
}

native "rust" mod native_mod = "" {
  fn str_byte_len(str s) -> vec[u8];
  fn str_alloc(uint n_bytes) -> str;
}

// FIXME: comparison of native fns
fn test_native_fn() {
  /*assert (native_mod::str_byte_len == native_mod::str_byte_len);
    assert (native_mod::str_byte_len != native_mod::str_alloc);*/
}

fn test_obj() {
  auto o1 = obj() { };
  auto o2 = obj() { };

  assert (o1 == o2);
  assert !(o1 != o2);
  assert !(o1 < o2);
  assert (o1 <= o2);
  assert !(o1 > o2);
  assert (o1 >= o2);

  obj constr1(int i) { };
  obj constr2(int i) { };

  auto o5 = constr1(10);
  auto o6 = constr1(10);
  auto o7 = constr1(11);
  auto o8 = constr2(11);

  assert (o5 == o6);
  assert (o6 == o7);
  assert (o7 == o8);
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