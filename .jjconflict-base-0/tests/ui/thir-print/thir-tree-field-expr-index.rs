//@ check-pass
//@ compile-flags: -Zunpretty=thir-tree

struct S {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    e: u32,
}

fn update(x: u32) {
  let s = S { a: x, b: x, c: x, d: x, e: x };

  S { a: x , ..s };
  S { b: x , ..s };
  S { c: x , ..s };
  S { d: x , ..s };
  S { e: x , ..s };

  S { b: x, d: x, ..s };
  S { a: x, c: x, e: x, ..s };
}

fn main() {}
