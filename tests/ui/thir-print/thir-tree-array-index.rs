//@ check-pass
//@ compile-flags: -Zunpretty=thir-tree

fn index(x: usize) -> usize { x }

fn indexing(x: usize) -> usize {
  let a1: [usize; 5] = [1, 2, 3, 4, 5];
  let a2: [usize; 5] = [x; 5];

  a1[0];
  a2[1];

  a1[x];
  a2[x + 2];

  a1[a2[x] + a2[x - 3]];
  a2[index (x + 1) - x];

  0
}

fn main() {}
