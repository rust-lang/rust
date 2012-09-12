extern mod std;
use std::list;

enum foo {
  a(uint),
  b(~str),
}

fn check_log<T>(exp: ~str, v: T) {
    assert exp == fmt!("%?", v);
}

fn main() {
    let x = list::from_vec(~[a(22u), b(~"hi")]);
    let exp = ~"@Cons(a(22), @Cons(b(~\"hi\"), @Nil))";
    assert fmt!("%?", x) == exp;
    check_log(exp, x);
}
