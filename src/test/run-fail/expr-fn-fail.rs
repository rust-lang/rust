// xfail-stage0
// error-pattern:explicit failure

fn f() -> ! { fail }

fn main() {
  f();
}
