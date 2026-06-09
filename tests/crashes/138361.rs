//@ known-bug: #138361

fn main() {
  [0; loop{}];
  std::mem::transmute(4)
}
