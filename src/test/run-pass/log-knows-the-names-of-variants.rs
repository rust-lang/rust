tag foo {
  a(uint);
  b(str);
  c;
}

fn main() {
    assert "a(22)" == #fmt["%?", a(22u)];
    assert "b(\"hi\")" == #fmt["%?", b("hi")];
    assert "c" == #fmt["%?", c];
}
