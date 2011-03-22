// -*- rust -*-

// Tests of ports and channels on various types

impure fn test_rec() {
  type r = rec(int val0, u8 val1, char val2);

  let port[r] po = port();
  let chan[r] ch = chan(po);
  let r r0 = rec(val0 = 0, val1 = 1u8, val2 = '2');

  ch <| r0;

  let r r1;
  r1 <- po;

  check (r1.val0 == 0);
  check (r1.val1 == 1u8);
  check (r1.val2 == '2');
}

impure fn test_vec() {
  let port[vec[int]] po = port();
  let chan[vec[int]] ch = chan(po);
  let vec[int] v0 = vec(0, 1, 2);

  ch <| v0;

  let vec[int] v1;
  v1 <- po;

  check (v1.(0) == 0);
  check (v1.(1) == 1);
  check (v1.(2) == 2);
}

impure fn test_tup() {
  type t = tup(int, u8, char);

  let port[t] po = port();
  let chan[t] ch = chan(po);
  let t t0 = tup(0, 1u8, '2');

  ch <| t0;

  let t t1;
  t1 <- po;

  check (t0._0 == 0);
  check (t0._1 == 1u8);
  check (t0._2 == '2');
}

impure fn test_tag() {
  tag t {
    tag1;
    tag2(int);
    tag3(int, u8, char);
  }

  let port[t] po = port();
  let chan[t] ch = chan(po);

  ch <| tag1;
  ch <| tag2(10);
  ch <| tag3(10, 11u8, 'A');

  let t t1;

  t1 <- po;
  check (t1 == tag1);
  t1 <- po;
  check (t1 == tag2(10));
  t1 <- po;
  check (t1 == tag3(10, 11u8, 'A'));
}

impure fn test_chan() {
  let port[chan[int]] po = port();
  let chan[chan[int]] ch = chan(po);

  let port[int] po0 = port();
  let chan[int] ch0 = chan(po0);

  ch <| ch0;

  let chan[int] ch1;
  ch1 <- po;

  // Does the transmitted channel still work?
  ch1 <| 10;

  let int i;
  i <- po0;

  check (i == 10);
}

impure fn main() {
  test_rec();
  test_vec();
  test_tup();
  test_tag();
  test_chan();
}
