// -*- rust -*-
// error-pattern:src/test/compile-fail/shadow.rs
fn foo(c: vec[int]) {
    let a: int = 5;
    let b: vec[int] = [];


    alt none[int] {
      some[int](_) { for i: int  in c { log a; let a = 17; b += [a]; } }
    }
}

tag t[T] { none; some(T); }

fn main() { foo([]); }