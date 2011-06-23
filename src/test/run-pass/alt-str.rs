// Issue #53

fn main() {
  alt ("test") {
    case ("not-test") { fail; }
    case ("test") { }
    case (_) { fail; }
  }

  tag t {
    tag1(str);
    tag2;
  }

  alt (tag1("test")) {
    case (tag2) { fail; }
    case (tag1("not-test")) { fail; }
    case (tag1("test")) { }
    case (_) { fail; }
  }
}