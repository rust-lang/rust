// xfail-stage0

tag maybe[T] {
  nothing;
  just(T);
}

fn foo(maybe[int] x) {
  alt (x) {
    case (nothing) {log_err "A";}
    case (just(?a)) {log_err "B";}
  }
}

fn main() {}
