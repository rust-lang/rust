// xfail-boot
// xfail-stage0
// error-pattern: unresolved name

// Tag variants are not exported with their tags. This allows for a
// simple sort of ADT.

mod foo {
  export t;

  tag t {
    t1;
  }
}

fn main() {
  auto x = foo.t1;
}
