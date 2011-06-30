// xfail-stage0

#[cfg(bogus)]
const bool b = false;

const bool b = true;

#[cfg(bogus)]
native "rust" mod rustrt {
  // This symbol doesn't exist and would be a link error if this
  // module was translated
  fn bogus();
}

native "rust" mod rustrt {
}

#[cfg(bogus)]
type t = int;

type t = bool;

#[cfg(bogus)]
tag tg {
  foo;
}

tag tg {
  bar;
}

#[cfg(bogus)]
obj o() {
  fn f() { ret bogus; }
}

obj o() {
}

#[cfg(bogus)]
resource r(int i) {}

resource r(int i) {}

#[cfg(bogus)]
mod m {
  // This needs to parse but would fail in typeck. Since it's not in
  // the current config it should not be typechecked.
  fn bogus() { ret 0; }
}

mod m {

  // Submodules have slightly different code paths than the top-level
  // module, so let's make sure this jazz works here as well
  #[cfg(bogus)]
  fn f() {}

  fn f() {}
}

// Since the bogus configuration isn't defined main will just be
// parsed, but nothing further will be done with it
#[cfg(bogus)]
fn main() { fail }

fn main() {
  // Exercise some of the configured items in ways that wouldn't be possible
  // if they had the bogus definition
  assert b;
  let t x = true;
  let tg y = bar;

  test_in_fn_ctxt();
}

fn test_in_fn_ctxt() {
  #[cfg(bogus)]
  fn f() { fail }
  fn f() {}
  f();

  #[cfg(bogus)]
  const int i = 0;
  const int i = 1;
  assert i == 1;
}
