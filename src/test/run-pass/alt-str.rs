// Issue #53

fn main() {
    alt "test" { "not-test" { fail; } "test" { } _ { fail; } }

    tag t { tag1(str); tag2; }


    alt tag1("test") {
      tag2. { fail; }
      tag1("not-test") { fail; }
      tag1("test") { }
      _ { fail; }
    }
}
