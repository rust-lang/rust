enum blah { a; b; }

fn or_alt(q: blah) -> int {
  alt q { a | b { 42 } }
}

fn main() {
    assert (or_alt(a) == 42);
    assert (or_alt(b) == 42);
}
