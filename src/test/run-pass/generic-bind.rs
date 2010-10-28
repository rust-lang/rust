fn id[T](&T t) -> T {
  ret t;
}

fn main() {
  auto f = bind id[int](_);
  check (f(10) == 10);
}
