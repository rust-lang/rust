// run-pass

struct V;
struct S;

impl Drop for S {
    fn drop(&mut self) {
        std::process::exit(1);
    }
}

enum E {
    A,
    B((V, S)),
}

fn foo(v: &mut E) {
    *v = E::B((V, S));
}

fn bar() {
    let mut v = E::A;
    match v {
        E::A => (),
        _ => unreachable!(),
    }
    foo(&mut v);
    match v {
        E::B((_x, _)) => {}
        _ => {}
    }
    // v is `E::B((V, S))`, so `S::drop` should be called and this shouldn't return.
}

pub fn main() {
    bar();
    unreachable!();
}
