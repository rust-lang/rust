// -*- rust -*-

// Tests of ports and channels on various types

fn test_rec() {
  type r = rec(int val0, u8 val1, char val2);

  let port[r] po = port();
  let chan[r] ch = chan(po);
  let r r0 = rec(val0 = 0, val1 = 1u8, val2 = '2');

  ch <| r0;

  let r r1;
  po |> r1;

  assert (r1.val0 == 0);
  assert (r1.val1 == 1u8);
  assert (r1.val2 == '2');
}

fn test_vec() {
  let port[vec[int]] po = port();
  let chan[vec[int]] ch = chan(po);
  let vec[int] v0 = [0, 1, 2];

  ch <| v0;

  let vec[int] v1;
  po |> v1;

  assert (v1.(0) == 0);
  assert (v1.(1) == 1);
  assert (v1.(2) == 2);
}

fn test_str() {
  let port[str] po = port();
  let chan[str] ch = chan(po);
  let str s0 = "test";

  ch <| s0;

  let str s1;
  po |> s1;

  assert (s1.(0) as u8 == 't' as u8);
  assert (s1.(1) as u8 == 'e' as u8);
  assert (s1.(2) as u8 == 's' as u8);
  assert (s1.(3) as u8 == 't' as u8);
}

fn test_tup() {
  type t = tup(int, u8, char);

  let port[t] po = port();
  let chan[t] ch = chan(po);
  let t t0 = tup(0, 1u8, '2');

  ch <| t0;

  let t t1;
  po |> t1;

  assert (t0._0 == 0);
  assert (t0._1 == 1u8);
  assert (t0._2 == '2');
}

fn test_tag() {
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

  po |> t1;
  assert (t1 == tag1);
  po |> t1;
  assert (t1 == tag2(10));
  po |> t1;
  assert (t1 == tag3(10, 11u8, 'A'));
}

fn test_chan() {
  let port[chan[int]] po = port();
  let chan[chan[int]] ch = chan(po);

  let port[int] po0 = port();
  let chan[int] ch0 = chan(po0);

  ch <| ch0;

  let chan[int] ch1;
  po |> ch1;

  // Does the transmitted channel still work?
  ch1 <| 10;

  let int i;
  po0 |> i;

  assert (i == 10);
}

fn main() {
  test_rec();
  test_vec();
  test_str();
  test_tup();
  test_tag();
  test_chan();
}
