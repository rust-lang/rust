fn main() {
  let v = vec![1i32, 2, 3];
  for _ in v[1..] {
    //~^ ERROR `[i32]` is not an iterator [E0277]
    //~| ERROR `[i32]` is not an iterator [E0277]
  }
  struct K {
    n: i32,
  }
  let mut v2 = vec![K { n: 1 }, K { n: 1 }, K { n: 1 }];
  for i2 in v2[1..] {
    //~^ ERROR `[K]` is not an iterator [E0277]
    //~| ERROR `[K]` is not an iterator [E0277]
    i2.n = 2;
  }
}
