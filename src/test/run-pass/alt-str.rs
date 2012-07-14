// Issue #53

fn main() {
    alt check ~"test" { ~"not-test" { fail; } ~"test" { } _ { fail; } }

    enum t { tag1(~str), tag2, }


    alt tag1(~"test") {
      tag2 { fail; }
      tag1(~"not-test") { fail; }
      tag1(~"test") { }
      _ { fail; }
    }

    let x = alt check ~"a" { ~"a" { 1 } ~"b" { 2 } };
    assert (x == 1);

    alt check ~"a" { ~"a" { } ~"b" { } }

}
