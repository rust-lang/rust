// -*- rust -*-
fn foo(c: [int]) {
    let a: int = 5;
    let b: [int] = ~[];


    alt none[int] {
      some[int](_) { for i: int in c { log a; let a = 17; b += ~[a]; } }
      _ {}
    }
}

tag t[T] { none; some(T); }

fn main() {
    let x = 10;
    let x = x + 20;
    assert x == 30;
    foo(~[]);
}
