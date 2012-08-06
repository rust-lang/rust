// xfail-test
enum pat { pat_ident(option<uint>) }

fn f(pat: pat) -> bool { true }

fn num_bindings(pat: pat) -> uint {
    match pat {
      pat_ident(_) if f(pat) { 0 }
      pat_ident(none) { 1 }
      pat_ident(some(sub)) { sub }
    }
}

fn main() {}