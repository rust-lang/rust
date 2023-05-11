// compile-flags: -Zmaximal-hir-to-mir-coverage
// run-pass

// Just making sure this flag is accepted and doesn't crash the compiler

fn main() {
  let x = 1;
  let y = x + 1;
  println!("{y}");
}
