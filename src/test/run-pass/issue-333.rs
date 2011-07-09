fn quux[T](&T x) -> T{
  auto f = id[T];
  ret f(x);
}

fn id[T](&T x) -> T {
  ret x;
}

fn main() {
  assert quux(10) == 10;
}