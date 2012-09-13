enum foo {
  a(uint),
  b(~str),
  c,
}

enum bar {
  d, e, f
}

fn main() {
    assert ~"a(22)" == fmt!("%?", a(22u));
    assert ~"b(~\"hi\")" == fmt!("%?", b(~"hi"));
    assert ~"c" == fmt!("%?", c);
    assert ~"d" == fmt!("%?", d);
}
