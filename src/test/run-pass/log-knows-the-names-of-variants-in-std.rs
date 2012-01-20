use std;
import std::list;

enum foo {
  a(uint),
  b(str),
}

fn check_log<T>(exp: str, v: T) {
    assert exp == #fmt["%?", v];
}

fn main() {
    let x = list::from_vec([a(22u), b("hi")]);
    let exp = "cons(a(22), @cons(b(\"hi\"), @nil))";
    assert #fmt["%?", x] == exp;
    check_log(exp, x);
}
