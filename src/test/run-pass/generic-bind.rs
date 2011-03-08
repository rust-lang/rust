fn id[T](&T t) -> T {
  ret t;
}

fn main() {
  auto t = tup(1,2,3,4,5,6,7);
  check (t._5 == 6);
  // FIXME: this needs to work.
  // auto f0 = bind id[tup(int,int,int,int,int,int,int)](t);
  auto f1 = bind id[tup(int,int,int,int,int,int,int)](_);
  // check (f0()._5 == 6);
  check (f1(t)._5 == 6);
}
